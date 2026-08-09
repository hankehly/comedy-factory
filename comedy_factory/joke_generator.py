"""Joke generator workflow (see README.md for the flowchart).

Step 1 — Scan news for topics: agent step that uses Gemini's native Google
Search grounding to find real, current stories and returns them as plain
factual one-sentence topics.

Step 2 — Generate subtext: LLM step that turns a topic into a subtext — the
writer's opinion, the idea the eventual joke communicates.

Step 3 — Grade subtext: evaluation gate. If the subtext breaks a rule, the
workflow re-runs step 2 with the grader's feedback.

Step 4 — Generate joke: LLM step that filters the subtext through a "funny
filter" (irony, character, shock, hyperbole) into a short joke.

Step 5 — Grade joke: evaluation gate. If the joke breaks a rule, the workflow
re-runs step 4 with the grader's feedback.
"""

import re

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch
from pydantic_ai.direct import model_request_sync
from pydantic_ai.messages import ModelMessage, ModelRequest
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.output import OutputObjectDefinition

from comedy_factory.prompts import load_prompt
from comedy_factory.settings import settings
from comedy_factory.utils import response_text


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


# Scanning news is a genuine agent step (search tool loop); the subtext steps
# are single model calls made directly via model_request_sync.
_scan_news_agent = Agent(
    settings.scan_news_model,
    name="scan_news",
    capabilities=[WebSearch()],
)

_grade_request_parameters = ModelRequestParameters(
    output_mode="native",
    output_object=OutputObjectDefinition(
        name=Grade.__name__,
        json_schema=Grade.model_json_schema(),
    ),
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


def scan_news() -> str:
    """Return a factual news topic suitable for joke writing."""
    prompt = load_prompt("scan-news.md")

    result = _scan_news_agent.run_sync(prompt)

    # The prompt demands a bare one-line topic; strip any numbering/bullets or
    # extra lines that slip through anyway.
    for line in result.output.splitlines():
        topic = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if topic:
            return topic
    raise RuntimeError("News scan returned no topic")


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
        settings.model,
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
        settings.model,
        history,
        model_request_parameters=_joke_request_parameters,
    )
    history.append(response)

    return Joke.model_validate_json(response_text(response)), history


def grade_subtext(topic: str, subtext: str) -> Grade:
    """Evaluate a subtext against the rules; a fail comes with feedback."""
    prompt = load_prompt("evaluate-subtext.md", TOPIC=topic, SUBTEXT=subtext)

    response = model_request_sync(
        settings.model,
        [ModelRequest.user_text_prompt(prompt)],
        model_request_parameters=_grade_request_parameters,
    )

    return Grade.model_validate_json(response_text(response))


def grade_joke(subtext: str, joke: str) -> Grade:
    """Evaluate a joke against the rules; a fail comes with feedback."""
    prompt = load_prompt("evaluate-joke.md", SUBTEXT=subtext, JOKE=joke)

    response = model_request_sync(
        settings.model,
        [ModelRequest.user_text_prompt(prompt)],
        model_request_parameters=_grade_request_parameters,
    )

    return Grade.model_validate_json(response_text(response))


def main():
    """Execute the joke generator workflow."""
    topic = scan_news()
    print(f"Topic: {topic}")

    history = None
    feedback = None
    for attempt in range(1, settings.max_grade_attempts + 1):
        subtext, history = generate_subtext(topic, history, feedback)
        grade = grade_subtext(topic, subtext.text)
        if grade.passed:
            break
        feedback = grade.feedback
        print(f"Subtext failed grading (attempt {attempt}):\n{feedback}")
    else:
        raise RuntimeError(
            f"Subtext failed grading after {settings.max_grade_attempts} attempts"
        )
    print(f"Subtext: {subtext.text}")

    history = None
    feedback = None
    for attempt in range(1, settings.max_grade_attempts + 1):
        joke, history = generate_joke(subtext.text, history, feedback)
        grade = grade_joke(subtext.text, joke.text)
        if grade.passed:
            break
        feedback = grade.feedback
        print(f"Joke failed grading (attempt {attempt}):\n{feedback}")
    else:
        raise RuntimeError(
            f"Joke failed grading after {settings.max_grade_attempts} attempts"
        )
    print(f"Joke: {joke.text}")
    print(f"Rationale: {joke.rationale}")


if __name__ == "__main__":
    main()
