"""Joke generator workflow (see README.md for the flowchart).

Step 1 — Find topic: agent step that uses Gemini's native Google Search
grounding to find a real, current news story and return it as a plain factual
one-sentence topic with its source URL.

Step 2 — Generate subtext: LLM step that turns a topic into a subtext — the
writer's opinion, the idea the eventual joke communicates.

Step 3 — Grade subtext: evaluation gate, run as step 2's output validator. If
the subtext breaks a rule, the grader's feedback goes back to the model as a
retry in the same conversation.

Step 4 — Generate joke: LLM step that filters the subtext through a "funny
filter" (irony, character, shock, hyperbole) into a short joke.

Step 5 — Grade joke: evaluation gate, run as step 4's output validator. If the
joke breaks a rule, the grader's feedback goes back to the model as a retry in
the same conversation.

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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TypeVar

import httpx
from loguru import logger
from pydantic import BaseModel, Field
from pydantic_ai import (
    Agent,
    AgentRunResult,
    ModelRetry,
    RunContext,
    UnexpectedModelBehavior,
)
from pydantic_ai.capabilities import WebSearch
from pydantic_ai.messages import BinaryImage, ModelMessage, ModelResponse, UserContent
from pydantic_ai.models import Model, ModelRequestParameters, infer_model
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.output import StructuredOutputMode
from pydantic_ai.profiles import ModelProfile, merge_profile
from pydantic_ai.settings import ModelSettings

from comedy_factory.agents import CLOUDFLARE_PREFIX, AgentVersion, selected_versions
from comedy_factory.captioning import render_caption
from comedy_factory.prompts import load_prompt
from comedy_factory.settings import settings
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
    model responses (a tool loop, the rewrites after failed gradings); sum
    them all."""
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


@dataclass
class GradedRun:
    """Run dependencies of a generation step with an evaluation gate: what the
    step's output validator needs to grade each attempt, and where it leaves
    the grader's costs so they are reported under the grader's own step."""

    context: str
    """What the output is graded against: the topic for a subtext, the
    subtext for a joke."""
    grader_version: AgentVersion
    grader_cost: StepCost = field(default_factory=StepCost)


# One agent per LLM step. None is bound to a model: each run is handed its
# step's agent version (see _run_step and agents.py), which supplies
# the model and model settings. Finding a topic is the one step that loops
# (search tool); the rest answer in a single structured response. The two
# generation steps with an evaluation gate carry a GradedRun: their output
# validators (below) run the grader and send a fail back as a retry.
_find_topic_agent = Agent(name="find_topic", capabilities=[WebSearch()])
_generate_subtext_agent = Agent(name="generate_subtext", output_type=Subtext, deps_type=GradedRun)
_grade_subtext_agent = Agent(name="grade_subtext", output_type=Grade)
_generate_joke_agent = Agent(name="generate_joke", output_type=Joke, deps_type=GradedRun)
_grade_joke_agent = Agent(name="grade_joke", output_type=Grade)
_rephrase_joke_agent = Agent(name="rephrase_joke", output_type=Rephrasings)
_write_image_prompt_agent = Agent(name="write_image_prompt", output_type=ImagePrompt)
_google_image_agent = Agent(name="generate_image", output_type=BinaryImage)
_describe_image_agent = Agent(name="describe_image", output_type=ImageDescription)


@_generate_subtext_agent.output_validator
def _grade_subtext_output(ctx: RunContext[GradedRun], subtext: Subtext) -> Subtext:
    """Step 3: grade the subtext; a fail goes back to the model as a retry."""
    grade, cost = grade_subtext(ctx.deps.context, subtext.text, version=ctx.deps.grader_version)
    ctx.deps.grader_cost += cost
    if not grade.passed:
        logger.warning(f"Subtext failed grading (attempt {ctx.retry + 1}):\n{grade.feedback}")
        raise ModelRetry(grade.feedback)
    return subtext


@_generate_joke_agent.output_validator
def _grade_joke_output(ctx: RunContext[GradedRun], joke: Joke) -> Joke:
    """Step 5: grade the joke; a fail goes back to the model as a retry."""
    grade, cost = grade_joke(ctx.deps.context, joke.text, version=ctx.deps.grader_version)
    ctx.deps.grader_cost += cost
    if not grade.passed:
        logger.warning(f"Joke failed grading (attempt {ctx.retry + 1}):\n{grade.feedback}")
        raise ModelRetry(grade.feedback)
    return joke


def _run_step(
    agent: Agent[None, OutputT],
    version: AgentVersion,
    prompt: str | Sequence[UserContent],
) -> AgentRunResult[OutputT]:
    """Run a step's agent on one of its versions: the version's model, with
    the version applied as a run-time agent spec for its model settings."""
    return agent.run_sync(prompt, model=_step_model(version), spec=version.spec)


def _run_graded_step(
    agent: Agent[GradedRun, OutputT],
    version: AgentVersion,
    prompt: str,
    deps: GradedRun,
    *,
    name: str,
) -> AgentRunResult[OutputT]:
    """Run a generation step whose output validator grades each attempt. The
    run gets `max_grade_attempts` attempts in all: the first plus that many
    output retries less one. pydantic-ai spends the same retry budget on a
    malformed structured response, so one of those costs a grading attempt.
    Running out raises RuntimeError naming the step's output (`name`), as the
    pipeline always has."""
    try:
        return agent.run_sync(
            prompt,
            model=_step_model(version),
            spec=version.spec,
            deps=deps,
            retries={"output": settings.max_grade_attempts - 1},
        )
    except UnexpectedModelBehavior as e:
        raise RuntimeError(
            f"{name} failed grading after {settings.max_grade_attempts} attempts"
        ) from e


def _step_model(version: AgentVersion) -> Model:
    """The model a step runs on: the version's model, set to request structured
    output the way it supports."""
    return _StructuredOutputModel(infer_model(version.model))


class _StructuredOutputModel(WrapperModel):
    """A model that asks for a step's structured output natively
    (JSON-schema constrained) where it supports that, as with Gemini and Claude
    Sonnet/Opus, otherwise with the schema in the prompt (e.g. Claude Haiku).
    Tool-call output, pydantic-ai's default, is avoided: Anthropic rejects
    forced tool use while thinking is enabled. Pinned on the model rather than
    chosen per run because the agents' output types have to stay fixed: an
    agent with output validators takes no run-time output type.
    Non-structured outputs pass through as is."""

    def __init__(self, wrapped: Model) -> None:
        super().__init__(wrapped)
        native = wrapped.profile.get("supports_json_schema_output", False)
        self._output_mode: StructuredOutputMode = "native" if native else "prompted"

    @property
    def profile(self) -> ModelProfile:  # type: ignore[override]
        return merge_profile(
            self.wrapped.profile, ModelProfile(default_structured_output_mode=self._output_mode)
        )

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        # The wrapped model resolves an undecided ("auto") output mode from its
        # own profile, so decide it here first.
        return await self.wrapped.request(
            messages,
            model_settings,
            model_request_parameters.with_default_output_mode(self._output_mode),
        )


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


def find_topic(*, version: AgentVersion) -> tuple[Topic, StepCost]:
    """Return a factual news topic suitable for joke writing, with its source
    URL when the model provides one, along with the step's usage and cost.
    Topics of recent asset bundles are passed to the model as stories to
    avoid."""
    recent = "\n".join(f"* {topic}" for topic in _recent_topics()) or "(none)"
    prompt = load_prompt("find-topic.md", RECENT_TOPICS=recent)

    result = _run_step(_find_topic_agent, version, prompt)

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
    topic: str, *, version: AgentVersion, grader_version: AgentVersion
) -> tuple[Subtext, StepCost, StepCost]:
    """Return a subtext — the writer's opinion about a news topic — that
    passed grading (step 3, run on `grader_version`), along with the
    generation step's usage and cost and the grader's.

    A failed grading goes back to the model as a retry in the same
    conversation, so it sees its previous attempts and the feedback on them;
    after `max_grade_attempts` attempts the step gives up with RuntimeError.
    """
    prompt = load_prompt("generate-subtext.md", TOPIC=topic)
    deps = GradedRun(context=topic, grader_version=grader_version)

    result = _run_graded_step(_generate_subtext_agent, version, prompt, deps, name="Subtext")
    return result.output, _run_cost(result), deps.grader_cost


def generate_joke(
    subtext: str, *, version: AgentVersion, grader_version: AgentVersion
) -> tuple[Joke, StepCost, StepCost]:
    """Return a joke communicating the subtext — along with the rationale
    behind its construction — that passed grading (step 5, run on
    `grader_version`), along with the generation step's usage and cost and
    the grader's.

    A failed grading goes back to the model as a retry in the same
    conversation, so it sees its previous attempts and the feedback on them;
    after `max_grade_attempts` attempts the step gives up with RuntimeError.
    """
    prompt = load_prompt("generate-joke.md", SUBTEXT=subtext)
    deps = GradedRun(context=subtext, grader_version=grader_version)

    result = _run_graded_step(_generate_joke_agent, version, prompt, deps, name="Joke")
    return result.output, _run_cost(result), deps.grader_cost


def write_image_prompt(joke: Joke, *, version: AgentVersion) -> tuple[ImagePrompt, StepCost]:
    """Return a text-to-image prompt for the image that accompanies the joke,
    along with the call's usage and cost.

    This step has no grading gate and no retry conversation — a single call.
    """
    prompt = load_prompt(
        "write-image-prompt.md", JOKE=joke.text, RATIONALE=joke.rationale
    )

    result = _run_step(_write_image_prompt_agent, version, prompt)
    return result.output, _run_cost(result)


def rephrase_joke(joke: Joke, *, version: AgentVersion) -> tuple[Rephrasings, StepCost]:
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

    result = _run_step(_rephrase_joke_agent, version, prompt)
    return result.output, _run_cost(result)


def generate_image(image_prompt: ImagePrompt, *, version: AgentVersion) -> tuple[bytes, StepCost]:
    """Render the image prompt with the version's image model, a Cloudflare
    Workers AI model for a cloudflare: version and a pydantic-ai one otherwise;
    returns the image bytes and the call's usage and cost."""
    if version.model.startswith(CLOUDFLARE_PREFIX):
        return _generate_image_cloudflare(
            image_prompt, version.model.removeprefix(CLOUDFLARE_PREFIX)
        )
    return _generate_image_google(image_prompt, version)


def _generate_image_google(
    image_prompt: ImagePrompt, version: AgentVersion
) -> tuple[bytes, StepCost]:
    """Generate the image with a Gemini image model ("Nano Banana")."""
    result = _run_step(_google_image_agent, version, image_prompt.text)
    return result.output.data, _run_cost(result)


def _generate_image_cloudflare(image_prompt: ImagePrompt, model: str) -> tuple[bytes, StepCost]:
    """Generate the image with a Cloudflare Workers AI model, such as FLUX."""
    response = httpx.post(
        "https://api.cloudflare.com/client/v4/accounts/"
        f"{settings.cloudflare_account_id}/ai/run/{model}",
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


def describe_image(image: bytes, *, version: AgentVersion) -> tuple[ImageDescription, StepCost]:
    """Return a factual visual description of the generated image, along with
    the call's usage and cost.

    The description covers only what is visible in the image — the caption
    sentence of the alt text is templated on by `compose_alt_text`, never read
    back out of the pixels.
    """
    prompt = load_prompt("describe-image.md")

    result = _run_step(
        _describe_image_agent,
        version,
        [prompt, BinaryImage(data=image, media_type=image_media_type(image))],
    )
    return result.output, _run_cost(result)


def grade_subtext(topic: str, subtext: str, *, version: AgentVersion) -> tuple[Grade, StepCost]:
    """Evaluate a subtext against the rules; a fail comes with feedback."""
    prompt = load_prompt("evaluate-subtext.md", TOPIC=topic, SUBTEXT=subtext)

    result = _run_step(_grade_subtext_agent, version, prompt)
    return result.output, _run_cost(result)


def grade_joke(subtext: str, joke: str, *, version: AgentVersion) -> tuple[Grade, StepCost]:
    """Evaluate a joke against the rules; a fail comes with feedback."""
    prompt = load_prompt("evaluate-joke.md", SUBTEXT=subtext, JOKE=joke)

    result = _run_step(_grade_joke_agent, version, prompt)
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
    versions: dict[str, AgentVersion],
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
        "agents": {step: version.as_metadata() for step, version in versions.items()},
        "costs": {
            "steps": {step: cost.model_dump() for step, cost in costs.items()},
            "total": sum(costs.values(), StepCost()).model_dump(),
        },
    }
    (bundle_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return bundle_dir


def main():
    """Execute the joke generator workflow."""
    # Load and check every step's agent version first, so a bad selection
    # fails before any model call is paid for.
    versions = selected_versions()
    costs: defaultdict[str, StepCost] = defaultdict(StepCost)

    topic, cost = find_topic(version=versions["find_topic"])
    costs["find_topic"] += cost
    logger.info(f"Topic: {topic.text}")
    logger.info(f"Source: {topic.source_url}")

    subtext, cost, grade_cost = generate_subtext(
        topic.text,
        version=versions["generate_subtext"],
        grader_version=versions["grade_subtext"],
    )
    costs["generate_subtext"] += cost
    costs["grade_subtext"] += grade_cost
    logger.info(f"Subtext: {subtext.text}")

    joke, cost, grade_cost = generate_joke(
        subtext.text,
        version=versions["generate_joke"],
        grader_version=versions["grade_joke"],
    )
    costs["generate_joke"] += cost
    costs["grade_joke"] += grade_cost
    logger.info(f"Joke: {joke.text}")
    logger.info(f"Rationale: {joke.rationale}")

    rephrasings, cost = rephrase_joke(joke, version=versions["rephrase_joke"])
    costs["rephrase_joke"] += cost
    for index, text in enumerate(rephrasings.texts, 1):
        logger.info(f"Rephrasing {index}: {text}")

    image_prompt, cost = write_image_prompt(joke, version=versions["write_image_prompt"])
    costs["write_image_prompt"] += cost
    logger.info(f"Image prompt: {image_prompt.text}")

    image, cost = generate_image(image_prompt, version=versions["generate_image"])
    costs["generate_image"] += cost
    captioned = render_caption(image, joke.text)
    rephrased_images = [render_caption(image, text) for text in rephrasings.texts]

    image_description, cost = describe_image(image, version=versions["describe_image"])
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
        versions,
    )
    logger.info(f"Asset bundle saved to {bundle_dir}")


if __name__ == "__main__":
    main()
