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
    Joke,
    Subtext,
    _scan_news_agent,
    generate_joke,
    generate_subtext,
    grade_joke,
    grade_subtext,
    scan_news,
)
from comedy_factory.settings import settings

# Grading uses output_mode="native"; TestModel's default profile rejects it.
_PROFILE = ModelProfile(supports_json_schema_output=True)


def canned_model(text: str) -> TestModel:
    """A model that always answers with `text`."""
    return TestModel(custom_output_text=text, profile=_PROFILE)


def test_scan_news_returns_bare_topic():
    canned = "* A fake topic.\n\nSome trailing explanation."
    with _scan_news_agent.override(model=canned_model(canned), native_tools=[]):
        topic = scan_news()
    assert topic == "A fake topic."


def test_generate_subtext(monkeypatch):
    canned = Subtext(text="A fake subtext.").model_dump_json()
    monkeypatch.setattr(settings, "model", canned_model(canned))
    subtext, history = generate_subtext("A fake topic.")
    assert subtext.text == "A fake subtext."
    assert len(history) == 2  # templated prompt + model response


def test_generate_subtext_retry_extends_history(monkeypatch):
    canned = Subtext(text="A better subtext.").model_dump_json()
    monkeypatch.setattr(settings, "model", canned_model(canned))
    _, history = generate_subtext("A fake topic.")
    subtext, history = generate_subtext(
        "A fake topic.", history=history, feedback="* Too wordy."
    )
    assert subtext.text == "A better subtext."
    assert len(history) == 4  # prompt, first attempt, feedback, rewrite
    assert "Too wordy" in history[2].parts[0].content


def test_generate_joke(monkeypatch):
    canned = Joke(text="A fake joke.", rationale="Irony.").model_dump_json()
    monkeypatch.setattr(settings, "model", canned_model(canned))
    joke, history = generate_joke("A fake subtext.")
    assert joke.text == "A fake joke."
    assert joke.rationale == "Irony."
    assert len(history) == 2


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


def test_grade_joke_fail(monkeypatch):
    canned = Grade(passed=False, feedback="* The funny part is not last.").model_dump_json()
    monkeypatch.setattr(settings, "model", canned_model(canned))
    grade = grade_joke("A fake subtext.", "A fake joke.")
    assert not grade.passed
    assert grade.feedback == "* The funny part is not last."


def _pipeline_model(messages: list, info: AgentInfo) -> ModelResponse:
    """Play every direct-call step, told apart by their output schemas."""
    output_object = info.model_request_parameters.output_object
    schema_name = output_object.name if output_object is not None else None
    if schema_name == "Grade":
        text = Grade(passed=True).model_dump_json()
    elif schema_name == "Joke":
        text = Joke(text="A fake joke.", rationale="Irony.").model_dump_json()
    elif schema_name == "Subtext":
        text = Subtext(text="A fake subtext.").model_dump_json()
    else:
        raise AssertionError(f"Unexpected request schema: {schema_name}")
    return ModelResponse(parts=[TextPart(content=text)])


def test_main_smoke(monkeypatch, capsys):
    monkeypatch.setattr(settings, "model", FunctionModel(_pipeline_model, profile=_PROFILE))
    with _scan_news_agent.override(model=canned_model("A fake topic."), native_tools=[]):
        joke_generator.main()

    out = capsys.readouterr().out
    assert "Topic: A fake topic." in out
    assert "Subtext: A fake subtext." in out
    assert "Joke: A fake joke." in out
    assert "Rationale: Irony." in out


def test_main_retries_failed_joke_grading(monkeypatch, capsys):
    joke_grades = iter(
        [Grade(passed=False, feedback="* The funny part is not last."), Grade(passed=True)]
    )

    def model(messages: list, info: AgentInfo) -> ModelResponse:
        content = messages[0].parts[0].content
        output_object = info.model_request_parameters.output_object
        schema_name = output_object.name if output_object is not None else None
        if schema_name == "Grade":
            grade = next(joke_grades) if "# Evaluate Joke" in content else Grade(passed=True)
            text = grade.model_dump_json()
        elif schema_name == "Joke":
            text = Joke(text="A fake joke.", rationale="Irony.").model_dump_json()
        elif schema_name == "Subtext":
            text = Subtext(text="A fake subtext.").model_dump_json()
        else:
            raise AssertionError(f"Unexpected request schema: {schema_name}")
        return ModelResponse(parts=[TextPart(content=text)])

    monkeypatch.setattr(settings, "model", FunctionModel(model, profile=_PROFILE))
    with _scan_news_agent.override(model=canned_model("A fake topic."), native_tools=[]):
        joke_generator.main()

    out = capsys.readouterr().out
    assert "Joke failed grading (attempt 1):\n* The funny part is not last." in out
    assert "Joke: A fake joke." in out
