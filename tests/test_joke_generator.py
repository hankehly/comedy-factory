"""Smoke tests for the joke generator pipeline — no real API calls.

Model stubbing follows pydantic-ai's recommended seams:

* The scan-news Agent is swapped via `agent.override(model=...)`; its WebSearch
  native tool is overridden away because TestModel can't emulate built-in tools.
* The direct-call steps read `settings.model` at call time, so tests monkeypatch
  that attribute with a TestModel/FunctionModel instance.
"""

from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.profiles import ModelProfile

from comedy_factory import joke_generator
from comedy_factory.joke_generator import (
    Grade,
    _scan_news_agent,
    generate_subtext,
    grade_subtext,
    scan_news,
)
from comedy_factory.settings import settings

# Grading uses output_mode="native"; TestModel's default profile rejects it.
_PROFILE = ModelProfile(supports_json_schema_output=True)


def canned_model(text: str) -> TestModel:
    """A model that always answers with `text`."""
    return TestModel(custom_output_text=text, profile=_PROFILE)


def test_scan_news_parses_topics():
    canned = "* First fake topic.\n\n2. Second fake topic.\n"
    with _scan_news_agent.override(model=canned_model(canned), native_tools=[]):
        topics = scan_news(num_topics=2)
    assert topics == ["First fake topic.", "Second fake topic."]


def test_generate_subtext_strips_quotes(monkeypatch):
    monkeypatch.setattr(settings, "model", canned_model('"A fake subtext."'))
    subtext, history = generate_subtext("A fake topic.")
    assert subtext == "A fake subtext."
    assert len(history) == 2  # templated prompt + model response


def test_generate_subtext_retry_extends_history(monkeypatch):
    monkeypatch.setattr(settings, "model", canned_model("A better subtext."))
    _, history = generate_subtext("A fake topic.")
    subtext, history = generate_subtext(
        "A fake topic.", history=history, feedback="* Too wordy."
    )
    assert subtext == "A better subtext."
    assert len(history) == 4  # prompt, first attempt, feedback, rewrite
    assert "Too wordy" in history[2].parts[0].content


def test_grade_subtext_pass(monkeypatch):
    canned = Grade(passed=True).model_dump_json()
    monkeypatch.setattr(settings, "model", canned_model(canned))
    grade = grade_subtext("A fake topic.", "A fake subtext.")
    assert grade.passed
    assert grade.feedback == ""


def test_grade_subtext_fail(monkeypatch):
    canned = Grade(passed=False, feedback="* Not a simple sentence.").model_dump_json()
    monkeypatch.setattr(settings, "model", canned_model(canned))
    grade = grade_subtext("A fake topic.", "A fake subtext.")
    assert not grade.passed
    assert grade.feedback == "* Not a simple sentence."


def _pipeline_model(messages: list, info: AgentInfo) -> ModelResponse:
    """Play both direct-call steps: grading requests carry an output schema."""
    if info.model_request_parameters.output_object is not None:
        text = Grade(passed=True).model_dump_json()
    else:
        text = "A fake subtext."
    return ModelResponse(parts=[TextPart(content=text)])


def test_main_smoke(monkeypatch, capsys):
    monkeypatch.setattr(settings, "model", FunctionModel(_pipeline_model, profile=_PROFILE))
    with _scan_news_agent.override(model=canned_model("A fake topic."), native_tools=[]):
        joke_generator.main()

    out = capsys.readouterr().out
    assert "Topic: A fake topic." in out
    assert "Subtext: A fake subtext." in out
