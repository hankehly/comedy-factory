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

Step 6 — Write image prompt: LLM step that writes a text-to-image prompt for a
text-free image that plays the joke straight; the joke text is rendered onto
the image later as a caption.

Step 7 — Generate image: renders the image prompt with the configured provider
— a Gemini image model ("Nano Banana") or FLUX.1-schnell on Cloudflare Workers
AI — and returns the image bytes.

Step 8 — Render caption: system step that word-wraps the joke text into a
white caption bar beneath the image and returns the combined JPEG bytes.

Step 9 — Describe image: vision LLM step that writes a factual visual
description of the generated (uncaptioned) image. The posting alt text is
this description with the caption sentence templated on, so a recaption can
rebuild the alt text without another vision call.

Step 10 — Evaluate joke holistically: evaluation of the finished asset as a
whole. Placeholder for now — the criteria are undecided, so everything passes;
the verdict is recorded in the asset bundle rather than blocking it.

Step 11 — Save asset: system step that writes the run's artifacts (original
image, captioned image, and metadata) to a timestamped output directory.
"""

import base64
import json
import re
from datetime import datetime
from pathlib import Path

import httpx
from loguru import logger

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch
from pydantic_ai.direct import model_request_sync
from pydantic_ai.messages import (
    BinaryImage,
    ModelMessage,
    ModelRequest,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.output import OutputObjectDefinition

from comedy_factory.captioning import render_caption
from comedy_factory.prompts import load_prompt
from comedy_factory.settings import settings
from comedy_factory.utils import (
    compose_alt_text,
    image_media_type,
    response_image,
    response_text,
)


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


# Finding a topic is a genuine agent step (search tool loop); the other LLM
# steps are single model calls made directly via model_request_sync.
_find_topic_agent = Agent(
    settings.find_topic_model,
    name="find_topic",
    capabilities=[WebSearch()],
)

_grade_request_parameters = ModelRequestParameters(
    output_mode="native",
    output_object=OutputObjectDefinition(
        name=Grade.__name__,
        json_schema=Grade.model_json_schema(),
    ),
)

class ImagePrompt(BaseModel):
    """A text-to-image prompt for the image that accompanies a joke."""

    text: str = Field(
        description=(
            "The image prompt and nothing else — no preamble and no"
            " explanation."
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


_subtext_request_parameters = ModelRequestParameters(
    output_mode="native",
    output_object=OutputObjectDefinition(
        name=Subtext.__name__,
        json_schema=Subtext.model_json_schema(),
    ),
)

_joke_request_parameters = ModelRequestParameters(
    output_mode="native",
    output_object=OutputObjectDefinition(
        name=Joke.__name__,
        json_schema=Joke.model_json_schema(),
    ),
)

_image_prompt_request_parameters = ModelRequestParameters(
    output_mode="native",
    output_object=OutputObjectDefinition(
        name=ImagePrompt.__name__,
        json_schema=ImagePrompt.model_json_schema(),
    ),
)

_image_description_request_parameters = ModelRequestParameters(
    output_mode="native",
    output_object=OutputObjectDefinition(
        name=ImageDescription.__name__,
        json_schema=ImageDescription.model_json_schema(),
    ),
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


def find_topic() -> Topic:
    """Return a factual news topic suitable for joke writing, with its source
    URL when the model provides one. Topics of recent asset bundles are passed
    to the model as stories to avoid."""
    recent = "\n".join(f"* {topic}" for topic in _recent_topics()) or "(none)"
    prompt = load_prompt("find-topic.md", RECENT_TOPICS=recent)

    result = _find_topic_agent.run_sync(prompt)

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
    return Topic(text=text, source_url=url_match.group() if url_match else None)


def generate_subtext(
    topic: str,
    history: list[ModelMessage] | None = None,
    feedback: str | None = None,
) -> tuple[Subtext, list[ModelMessage]]:
    """Return a subtext — the writer's opinion about a news topic — and the
    conversation history that produced it.

    To request a rewrite after a failed grading, pass back the returned
    `history` along with the grader's `feedback`; the model then sees its
    previous attempts as prior turns of the conversation.
    """
    if history is None:
        prompt = load_prompt("generate-subtext.md", TOPIC=topic)
        history = [ModelRequest.user_text_prompt(prompt)]
    if feedback:
        history.append(
            ModelRequest.user_text_prompt(
                load_prompt("rewrite-with-feedback.md", FEEDBACK=feedback)
            )
        )

    response = model_request_sync(
        settings.generate_subtext_model,
        history,
        model_request_parameters=_subtext_request_parameters,
    )
    history.append(response)

    return Subtext.model_validate_json(response_text(response)), history


def generate_joke(
    subtext: str,
    history: list[ModelMessage] | None = None,
    feedback: str | None = None,
) -> tuple[Joke, list[ModelMessage]]:
    """Return a joke communicating the subtext — along with the rationale
    behind its construction — and the conversation history that produced it.

    To request a rewrite after a failed grading, pass back the returned
    `history` along with the grader's `feedback`; the model then sees its
    previous attempts as prior turns of the conversation.
    """
    if history is None:
        prompt = load_prompt("generate-joke.md", SUBTEXT=subtext)
        history = [ModelRequest.user_text_prompt(prompt)]
    if feedback:
        history.append(
            ModelRequest.user_text_prompt(
                load_prompt("rewrite-with-feedback.md", FEEDBACK=feedback)
            )
        )

    response = model_request_sync(
        settings.generate_joke_model,
        history,
        model_request_parameters=_joke_request_parameters,
    )
    history.append(response)

    return Joke.model_validate_json(response_text(response)), history


def write_image_prompt(joke: Joke) -> ImagePrompt:
    """Return a text-to-image prompt for the image that accompanies the joke.

    This step has no grading gate and no retry conversation — a single call.
    """
    prompt = load_prompt(
        "write-image-prompt.md", JOKE=joke.text, RATIONALE=joke.rationale
    )

    response = model_request_sync(
        settings.write_image_prompt_model,
        [ModelRequest.user_text_prompt(prompt)],
        model_request_parameters=_image_prompt_request_parameters,
    )

    return ImagePrompt.model_validate_json(response_text(response))


def generate_image(image_prompt: ImagePrompt) -> bytes:
    """Render the image prompt with the configured image provider."""
    if settings.image_provider == "cloudflare":
        return _generate_image_cloudflare(image_prompt)
    return _generate_image_google(image_prompt)


def _generate_image_google(image_prompt: ImagePrompt) -> bytes:
    """Generate the image with a Gemini image model ("Nano Banana")."""
    response = model_request_sync(
        settings.google_image_model,
        [ModelRequest.user_text_prompt(image_prompt.text)],
        model_request_parameters=ModelRequestParameters(allow_image_output=True),
    )
    return response_image(response)


def _generate_image_cloudflare(image_prompt: ImagePrompt) -> bytes:
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
    return base64.b64decode(payload["result"]["image"])


def describe_image(image: bytes) -> ImageDescription:
    """Return a factual visual description of the generated image.

    The description covers only what is visible in the image — the caption
    sentence of the alt text is templated on by `compose_alt_text`, never read
    back out of the pixels.
    """
    prompt = load_prompt("describe-image.md")

    response = model_request_sync(
        settings.describe_image_model,
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            prompt,
                            BinaryImage(
                                data=image, media_type=image_media_type(image)
                            ),
                        ]
                    )
                ]
            )
        ],
        model_request_parameters=_image_description_request_parameters,
    )

    return ImageDescription.model_validate_json(response_text(response))


def grade_subtext(topic: str, subtext: str) -> Grade:
    """Evaluate a subtext against the rules; a fail comes with feedback."""
    prompt = load_prompt("evaluate-subtext.md", TOPIC=topic, SUBTEXT=subtext)

    response = model_request_sync(
        settings.grade_subtext_model,
        [ModelRequest.user_text_prompt(prompt)],
        model_request_parameters=_grade_request_parameters,
    )

    return Grade.model_validate_json(response_text(response))


def grade_joke(subtext: str, joke: str) -> Grade:
    """Evaluate a joke against the rules; a fail comes with feedback."""
    prompt = load_prompt("evaluate-joke.md", SUBTEXT=subtext, JOKE=joke)

    response = model_request_sync(
        settings.grade_joke_model,
        [ModelRequest.user_text_prompt(prompt)],
        model_request_parameters=_grade_request_parameters,
    )

    return Grade.model_validate_json(response_text(response))


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
    image_prompt: ImagePrompt,
    image: bytes,
    captioned_image: bytes,
    image_description: ImageDescription,
    evaluation: Grade,
) -> Path:
    """Write the run's artifacts to a timestamped directory; returns its path."""
    created_at = datetime.now()
    bundle_dir = settings.output_dir / created_at.strftime("%Y%m%d-%H%M%S")
    bundle_dir.mkdir(parents=True, exist_ok=True)

    (bundle_dir / "image-original.jpg").write_bytes(image)
    (bundle_dir / "image-captioned.jpg").write_bytes(captioned_image)

    metadata = {
        "created_at": created_at.isoformat(),
        "topic": topic.model_dump(),
        "subtext": subtext.model_dump(),
        "joke": joke.model_dump(),
        "image_prompt": image_prompt.model_dump(),
        "image_description": image_description.model_dump(),
        "alt_text": compose_alt_text(image_description.text, joke.text),
        "evaluation": evaluation.model_dump(),
    }
    (bundle_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return bundle_dir


def main():
    """Execute the joke generator workflow."""
    topic = find_topic()
    logger.info(f"Topic: {topic.text}")
    logger.info(f"Source: {topic.source_url}")

    history = None
    feedback = None
    for attempt in range(1, settings.max_grade_attempts + 1):
        subtext, history = generate_subtext(topic.text, history, feedback)
        grade = grade_subtext(topic.text, subtext.text)
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
        joke, history = generate_joke(subtext.text, history, feedback)
        grade = grade_joke(subtext.text, joke.text)
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

    image_prompt = write_image_prompt(joke)
    logger.info(f"Image prompt: {image_prompt.text}")

    image = generate_image(image_prompt)
    captioned = render_caption(image, joke.text)

    image_description = describe_image(image)
    logger.info(f"Alt text: {compose_alt_text(image_description.text, joke.text)}")

    evaluation = grade_asset(topic, subtext, joke, captioned, image_description)
    bundle_dir = save_asset(
        topic,
        subtext,
        joke,
        image_prompt,
        image,
        captioned,
        image_description,
        evaluation,
    )
    logger.info(f"Asset bundle saved to {bundle_dir}")


if __name__ == "__main__":
    main()
