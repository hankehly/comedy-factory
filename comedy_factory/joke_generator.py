"""Joke generator workflow (see README.md for the flowchart).

Step 1 — Find topic: agent step that uses Gemini's native Google Search
grounding to find a real, current news story and return it as a plain factual
one-sentence topic with its source URL.

Step 2 — Generate subtext: LLM step that turns a topic into a subtext — the
writer's opinion, the idea the eventual joke communicates.

Step 3 — Grade subtext: evaluation gate. If the subtext breaks a rule, the
workflow re-runs step 2 with the grader's feedback.

Step 4 — Generate joke: LLM step that filters the subtext through a "funny
filter" (irony, character, shock, hyperbole) into a short joke.

Step 5 — Grade joke: evaluation gate. If the joke breaks a rule, the workflow
re-runs step 4 with the grader's feedback.

Step 6 — Rephrase joke: LLM step that writes N alternative phrasings of the
approved joke for later human review and possible recaptioning. No grading
gate — the human review is the evaluation.

Step 7 — Write image prompt: LLM step that writes a text-to-image prompt for a
text-free image that plays the joke straight; the joke text is rendered onto
the image later as a caption.

Step 8 — Generate image: renders the image prompt with the configured provider
— a Gemini image model ("Nano Banana") or FLUX.1-schnell on Cloudflare Workers
AI — and returns the image bytes.

Step 9 — Render caption: system step that word-wraps the joke text into a
white caption bar beneath the image and returns the combined JPEG bytes; also
renders one captioned image per rephrasing from the same base image.

Step 10 — Describe image: vision LLM step that writes a factual visual
description of the generated (uncaptioned) image. The posting alt text is
this description with the caption sentence templated on, so a recaption can
rebuild the alt text without another vision call.

Step 11 — Evaluate joke holistically: evaluation of the finished asset as a
whole. Placeholder for now — the criteria are undecided, so everything passes;
the verdict is recorded in the asset bundle rather than blocking it.

Step 12 — Save asset: system step that writes the run's artifacts (original
image, captioned images including the rephrased versions, and metadata) to a
timestamped output directory.

Each LLM step also surfaces its token usage and estimated cost (priced via
genai-prices); main() accumulates them per step and the totals are recorded
in the asset bundle's metadata.
"""

import base64
import json
import re
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TypeVar, cast

import httpx
from loguru import logger
from pydantic import BaseModel, Field
from pydantic_ai import Agent, AgentRunResult, NativeOutput, PromptedOutput
from pydantic_ai.capabilities import WebSearch
from pydantic_ai.messages import (
    BinaryImage,
    ModelMessage,
    ModelResponse,
    UserContent,
)
from pydantic_ai.models import Model, infer_model
from pydantic_ai.output import OutputSpec

from comedy_factory.captioning import render_caption
from comedy_factory.prompts import load_prompt
from comedy_factory.settings import StepConfig, settings
from comedy_factory.utils import compose_alt_text, image_media_type

OutputT = TypeVar("OutputT")


class Grade(BaseModel):
    """Result of an evaluation gate: pass, or fail with corrective feedback."""

    passed: bool = Field(description="Whether the graded text satisfies every rule.")
    feedback: str = Field(
        default="",
        description=(
            "When failed: a bullet list naming each violated rule and giving"
            " specific feedback on what to correct. Empty when passed."
        ),
    )


class Topic(BaseModel):
    """A news topic — the raw material a joke is built from."""

    text: str = Field(
        description="A single factual sentence summarizing one real news story."
    )
    source_url: str | None = Field(
        default=None,
        description="URL of the news story the topic summarizes, when available.",
    )


class Subtext(BaseModel):
    """A subtext — the writer's opinion that a joke communicates."""

    text: str = Field(
        description=(
            "The subtext sentence and nothing else — no preamble, no"
            " explanation, no surrounding quotation marks."
        )
    )


class Joke(BaseModel):
    """A joke and the reasoning behind its construction."""

    text: str = Field(
        description=(
            "The joke and nothing else — no preamble, no explanation, no"
            " surrounding quotation marks."
        )
    )
    rationale: str = Field(
        description=(
            "A short description of how the joke was created: which funny"
            " filter(s) were used and how the joke reveals the subtext to the"
            " reader."
        )
    )


class ImagePrompt(BaseModel):
    """A text-to-image prompt for the image that accompanies a joke."""

    text: str = Field(
        description=(
            "The image prompt and nothing else — no preamble and no"
            " explanation."
        )
    )


class StepCost(BaseModel):
    """Token usage and estimated cost of a pipeline step's model calls.

    `usd` is a genai-prices estimate; None means no price is known (models
    missing from the price table, or providers that report no usage). Adding
    StepCosts sums the tokens and the known usd components; the sum's usd is
    None only when every component is unknown.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    usd: float | None = None

    def __add__(self, other: "StepCost") -> "StepCost":
        if self.usd is None and other.usd is None:
            usd = None
        else:
            usd = (self.usd or 0.0) + (other.usd or 0.0)
        return StepCost(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            usd=usd,
        )


def _response_cost(response: ModelResponse) -> StepCost:
    """Token usage and estimated cost of a single model response."""
    try:
        usd = float(response.cost().total_price)
    except Exception:
        # genai-prices has no entry for every model (image models, test stubs).
        usd = None
    return StepCost(
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        usd=usd,
    )


def _run_cost(result: AgentRunResult[object]) -> StepCost:
    """Token usage and estimated cost of an agent run. A run may take several
    model responses (a tool loop, an output-validation retry); sum them all,
    leaving out the earlier turns passed in as message history."""
    return sum(
        (
            _response_cost(message)
            for message in result.new_messages()
            if isinstance(message, ModelResponse)
        ),
        StepCost(),
    )


class Rephrasings(BaseModel):
    """Alternative phrasings of the approved joke, for human review and
    possible recaptioning."""

    texts: list[str] = Field(
        description=(
            "The alternative phrasings of the joke, each one complete joke"
            " text and nothing else — no numbering, no explanation, no"
            " surrounding quotation marks."
        )
    )


class ImageDescription(BaseModel):
    """A factual visual description of the generated image — the part of the
    posting alt text that precedes the templated caption sentence."""

    text: str = Field(
        description=(
            "The visual description and nothing else — no preamble, no"
            " explanation, no surrounding quotation marks."
        )
    )


# One agent per LLM step. None is bound to a model: each run passes its
# step's StepConfig from settings (see _run_step), so the model and model
# settings are read at call time. Finding a topic is the one step that loops
# (search tool); the rest answer in a single structured response.
_find_topic_agent = Agent(name="find_topic", capabilities=[WebSearch()])
_generate_subtext_agent = Agent(name="generate_subtext", output_type=Subtext)
_grade_subtext_agent = Agent(name="grade_subtext", output_type=Grade)
_generate_joke_agent = Agent(name="generate_joke", output_type=Joke)
_grade_joke_agent = Agent(name="grade_joke", output_type=Grade)
_rephrase_joke_agent = Agent(name="rephrase_joke", output_type=Rephrasings)
_write_image_prompt_agent = Agent(name="write_image_prompt", output_type=ImagePrompt)
_google_image_agent = Agent(name="generate_image", output_type=BinaryImage)
_describe_image_agent = Agent(name="describe_image", output_type=ImageDescription)


def _run_step(
    agent: Agent[None, OutputT],
    step: StepConfig,
    prompt: str | Sequence[UserContent],
    message_history: list[ModelMessage] | None = None,
) -> AgentRunResult[OutputT]:
    """Run a step's agent with the step's configured model and settings."""
    model = infer_model(step.model)
    return agent.run_sync(
        prompt,
        model=model,
        model_settings=step.model_settings,
        message_history=message_history,
        output_type=_output_spec(agent.output_type, model),
    )


def _output_spec(output_type: OutputSpec[OutputT], model: Model) -> OutputSpec[OutputT]:
    """How the model should return a step's structured output: natively
    (JSON-schema constrained) where it supports that, as with Gemini and
    Claude Sonnet/Opus, otherwise with the schema in the prompt (e.g. Claude
    Haiku). Tool-call output is avoided: Anthropic rejects forced tool use
    while thinking is enabled. Non-structured outputs pass through as is."""
    # Narrowing a separate name keeps `output_type`'s declared type intact for
    # the passthrough return; mypy can't carry OutputT through issubclass().
    candidate: object = output_type
    if isinstance(candidate, type) and issubclass(candidate, BaseModel):
        native = model.profile.get("supports_json_schema_output", False)
        wrapper = NativeOutput if native else PromptedOutput
        return cast(OutputSpec[OutputT], wrapper(candidate))
    return output_type


def _recent_topics() -> list[str]:
    """Topic sentences of the newest asset bundles, newest first."""
    if not settings.output_dir.is_dir():
        return []
    topics = []
    for bundle_dir in sorted(settings.output_dir.iterdir(), reverse=True):
        metadata_path = bundle_dir / "metadata.json"
        if not metadata_path.is_file():
            continue
        try:
            topic = json.loads(metadata_path.read_text()).get("topic")
        except json.JSONDecodeError:
            continue
        # Older bundles stored the topic as a bare string.
        text = topic.get("text") if isinstance(topic, dict) else topic
        if text:
            topics.append(text)
        if len(topics) >= settings.max_recent_topics:
            break
    return topics


def find_topic() -> tuple[Topic, StepCost]:
    """Return a factual news topic suitable for joke writing, with its source
    URL when the model provides one, along with the step's usage and cost.
    Topics of recent asset bundles are passed to the model as stories to
    avoid."""
    recent = "\n".join(f"* {topic}" for topic in _recent_topics()) or "(none)"
    prompt = load_prompt("find-topic.md", RECENT_TOPICS=recent)

    result = _run_step(_find_topic_agent, settings.find_topic, prompt)

    # The prompt demands the bare topic line followed by the source URL; strip
    # any numbering/bullets or extra lines that slip through anyway.
    lines = [
        re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        for line in result.output.splitlines()
    ]
    text = next(
        (line for line in lines if line and not line.startswith("http")), None
    )
    if text is None:
        raise RuntimeError("News scan returned no topic")

    url_match = re.search(r"https?://\S+", result.output)
    topic = Topic(text=text, source_url=url_match.group() if url_match else None)
    return topic, _run_cost(result)


def generate_subtext(
    topic: str,
    history: list[ModelMessage] | None = None,
    feedback: str | None = None,
) -> tuple[Subtext, list[ModelMessage], StepCost]:
    """Return a subtext — the writer's opinion about a news topic — the
    conversation history that produced it, and the call's usage and cost.

    To request a rewrite after a failed grading, pass back the returned
    `history` along with the grader's `feedback`; the model then sees its
    previous attempts as prior turns of the conversation.
    """
    if history is None:
        prompt = load_prompt("generate-subtext.md", TOPIC=topic)
    else:
        prompt = load_prompt("rewrite-with-feedback.md", FEEDBACK=feedback)

    result = _run_step(
        _generate_subtext_agent, settings.generate_subtext, prompt, history
    )
    return result.output, result.all_messages(), _run_cost(result)


def generate_joke(
    subtext: str,
    history: list[ModelMessage] | None = None,
    feedback: str | None = None,
) -> tuple[Joke, list[ModelMessage], StepCost]:
    """Return a joke communicating the subtext — along with the rationale
    behind its construction — the conversation history that produced it, and
    the call's usage and cost.

    To request a rewrite after a failed grading, pass back the returned
    `history` along with the grader's `feedback`; the model then sees its
    previous attempts as prior turns of the conversation.
    """
    if history is None:
        prompt = load_prompt("generate-joke.md", SUBTEXT=subtext)
    else:
        prompt = load_prompt("rewrite-with-feedback.md", FEEDBACK=feedback)

    result = _run_step(_generate_joke_agent, settings.generate_joke, prompt, history)
    return result.output, result.all_messages(), _run_cost(result)


def write_image_prompt(joke: Joke) -> tuple[ImagePrompt, StepCost]:
    """Return a text-to-image prompt for the image that accompanies the joke,
    along with the call's usage and cost.

    This step has no grading gate and no retry conversation — a single call.
    """
    prompt = load_prompt(
        "write-image-prompt.md", JOKE=joke.text, RATIONALE=joke.rationale
    )

    result = _run_step(_write_image_prompt_agent, settings.write_image_prompt, prompt)
    return result.output, _run_cost(result)


def rephrase_joke(joke: Joke) -> tuple[Rephrasings, StepCost]:
    """Return alternative phrasings of the approved joke for human review,
    along with the call's usage and cost.

    This step has no grading gate and no retry conversation — the subsequent
    human review is the evaluation.
    """
    prompt = load_prompt(
        "rephrase-joke.md",
        JOKE=joke.text,
        RATIONALE=joke.rationale,
        N=settings.rephrasing_count,
    )

    result = _run_step(_rephrase_joke_agent, settings.rephrase_joke, prompt)
    return result.output, _run_cost(result)


def generate_image(image_prompt: ImagePrompt) -> tuple[bytes, StepCost]:
    """Render the image prompt with the configured image provider; returns the
    image bytes and the call's usage and cost."""
    if settings.image_provider == "cloudflare":
        return _generate_image_cloudflare(image_prompt)
    return _generate_image_google(image_prompt)


def _generate_image_google(image_prompt: ImagePrompt) -> tuple[bytes, StepCost]:
    """Generate the image with a Gemini image model ("Nano Banana")."""
    result = _run_step(_google_image_agent, settings.google_image, image_prompt.text)
    return result.output.data, _run_cost(result)


def _generate_image_cloudflare(image_prompt: ImagePrompt) -> tuple[bytes, StepCost]:
    """Generate the image with FLUX on Cloudflare Workers AI."""
    response = httpx.post(
        "https://api.cloudflare.com/client/v4/accounts/"
        f"{settings.cloudflare_account_id}/ai/run/{settings.cloudflare_image_model}",
        headers={"Authorization": f"Bearer {settings.cloudflare_api_token}"},
        json={"prompt": image_prompt.text},
        timeout=60,
    )
    response.raise_for_status()

    payload = response.json()
    if not payload.get("success"):
        raise RuntimeError(f"Image generation failed: {payload.get('errors')}")
    # Cloudflare reports no usage; it bills per image ("neurons") on its side.
    return base64.b64decode(payload["result"]["image"]), StepCost()


def describe_image(image: bytes) -> tuple[ImageDescription, StepCost]:
    """Return a factual visual description of the generated image, along with
    the call's usage and cost.

    The description covers only what is visible in the image — the caption
    sentence of the alt text is templated on by `compose_alt_text`, never read
    back out of the pixels.
    """
    prompt = load_prompt("describe-image.md")

    result = _run_step(
        _describe_image_agent,
        settings.describe_image,
        [prompt, BinaryImage(data=image, media_type=image_media_type(image))],
    )
    return result.output, _run_cost(result)


def grade_subtext(topic: str, subtext: str) -> tuple[Grade, StepCost]:
    """Evaluate a subtext against the rules; a fail comes with feedback."""
    prompt = load_prompt("evaluate-subtext.md", TOPIC=topic, SUBTEXT=subtext)

    result = _run_step(_grade_subtext_agent, settings.grade_subtext, prompt)
    return result.output, _run_cost(result)


def grade_joke(subtext: str, joke: str) -> tuple[Grade, StepCost]:
    """Evaluate a joke against the rules; a fail comes with feedback."""
    prompt = load_prompt("evaluate-joke.md", SUBTEXT=subtext, JOKE=joke)

    result = _run_step(_grade_joke_agent, settings.grade_joke, prompt)
    return result.output, _run_cost(result)


def grade_asset(
    topic: Topic,
    subtext: Subtext,
    joke: Joke,
    captioned_image: bytes,
    image_description: ImageDescription,
) -> Grade:
    """Evaluate the finished joke asset as a whole.

    Placeholder: the evaluation criteria are not yet decided, so everything
    passes. The signature already receives the full asset so criteria can be
    added later without rewiring the pipeline.
    """
    return Grade(passed=True)


def save_asset(
    topic: Topic,
    subtext: Subtext,
    joke: Joke,
    rephrasings: Rephrasings,
    image_prompt: ImagePrompt,
    image: bytes,
    captioned_image: bytes,
    rephrased_images: list[bytes],
    image_description: ImageDescription,
    costs: dict[str, StepCost],
    evaluation: Grade,
) -> Path:
    """Write the run's artifacts to a timestamped directory; returns its path."""
    created_at = datetime.now()
    bundle_dir = settings.output_dir / created_at.strftime("%Y%m%d-%H%M%S")
    bundle_dir.mkdir(parents=True, exist_ok=True)

    (bundle_dir / "image-original.jpg").write_bytes(image)
    (bundle_dir / "image-captioned.jpg").write_bytes(captioned_image)
    # 1-based, in metadata order, so humans can map image file to rephrasing.
    # The `image-captioned-*` prefix is owned by recaption.py's versioning.
    for index, rephrased_image in enumerate(rephrased_images, 1):
        (bundle_dir / f"image-rephrased-{index}.jpg").write_bytes(rephrased_image)

    metadata = {
        "created_at": created_at.isoformat(),
        "topic": topic.model_dump(),
        "subtext": subtext.model_dump(),
        "joke": joke.model_dump(),
        "rephrasings": rephrasings.texts,
        "image_prompt": image_prompt.model_dump(),
        "image_description": image_description.model_dump(),
        "alt_text": compose_alt_text(image_description.text, joke.text),
        "evaluation": evaluation.model_dump(),
        "costs": {
            "steps": {step: cost.model_dump() for step, cost in costs.items()},
            "total": sum(costs.values(), StepCost()).model_dump(),
        },
    }
    (bundle_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return bundle_dir


def main():
    """Execute the joke generator workflow."""
    costs: defaultdict[str, StepCost] = defaultdict(StepCost)

    topic, cost = find_topic()
    costs["find_topic"] += cost
    logger.info(f"Topic: {topic.text}")
    logger.info(f"Source: {topic.source_url}")

    history = None
    feedback = None
    for attempt in range(1, settings.max_grade_attempts + 1):
        subtext, history, cost = generate_subtext(topic.text, history, feedback)
        costs["generate_subtext"] += cost
        grade, cost = grade_subtext(topic.text, subtext.text)
        costs["grade_subtext"] += cost
        if grade.passed:
            break
        feedback = grade.feedback
        logger.warning(f"Subtext failed grading (attempt {attempt}):\n{feedback}")
    else:
        raise RuntimeError(
            f"Subtext failed grading after {settings.max_grade_attempts} attempts"
        )
    logger.info(f"Subtext: {subtext.text}")

    history = None
    feedback = None
    for attempt in range(1, settings.max_grade_attempts + 1):
        joke, history, cost = generate_joke(subtext.text, history, feedback)
        costs["generate_joke"] += cost
        grade, cost = grade_joke(subtext.text, joke.text)
        costs["grade_joke"] += cost
        if grade.passed:
            break
        feedback = grade.feedback
        logger.warning(f"Joke failed grading (attempt {attempt}):\n{feedback}")
    else:
        raise RuntimeError(
            f"Joke failed grading after {settings.max_grade_attempts} attempts"
        )
    logger.info(f"Joke: {joke.text}")
    logger.info(f"Rationale: {joke.rationale}")

    rephrasings, cost = rephrase_joke(joke)
    costs["rephrase_joke"] += cost
    for index, text in enumerate(rephrasings.texts, 1):
        logger.info(f"Rephrasing {index}: {text}")

    image_prompt, cost = write_image_prompt(joke)
    costs["write_image_prompt"] += cost
    logger.info(f"Image prompt: {image_prompt.text}")

    image, cost = generate_image(image_prompt)
    costs["generate_image"] += cost
    captioned = render_caption(image, joke.text)
    rephrased_images = [render_caption(image, text) for text in rephrasings.texts]

    image_description, cost = describe_image(image)
    costs["describe_image"] += cost
    logger.info(f"Alt text: {compose_alt_text(image_description.text, joke.text)}")

    total_cost = sum(costs.values(), StepCost())
    usd = f"${total_cost.usd:.4f}" if total_cost.usd is not None else "unknown"
    logger.info(
        f"Usage: {total_cost.input_tokens} input tokens,"
        f" {total_cost.output_tokens} output tokens, estimated cost {usd}"
    )

    evaluation = grade_asset(topic, subtext, joke, captioned, image_description)
    bundle_dir = save_asset(
        topic,
        subtext,
        joke,
        rephrasings,
        image_prompt,
        image,
        captioned,
        rephrased_images,
        image_description,
        dict(costs),
        evaluation,
    )
    logger.info(f"Asset bundle saved to {bundle_dir}")


if __name__ == "__main__":
    main()
