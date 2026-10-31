"""Smoke tests for the joke generator pipeline — no real API calls.

Model stubbing follows pydantic-ai's recommended seams:

* The scan-news Agent is swapped via `agent.override(model=...)`; its WebSearch
  native tool is overridden away because TestModel can't emulate built-in tools.
* The other steps run on a committed agent version (see `_version`), and
  their agent's model is swapped the same way: `agent.override(model=...)`
  takes precedence over the version's model, while the version's model
  settings still apply.
"""

import base64
import io
import json
import re
from collections.abc import Callable
from contextlib import ExitStack
from datetime import datetime

import pytest
from PIL import Image, ImageFont
from pydantic_ai import AgentSpec
from pydantic_ai.messages import BinaryImage, FilePart, ModelResponse, TextPart
from pydantic_ai.models import infer_model
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.profiles import ModelProfile

from comedy_factory import joke_generator, recaption
from comedy_factory.agents import STEPS, AgentVersion, AgentVersionError, load_version
from comedy_factory.captioning import _wrap_caption, render_caption
from comedy_factory.joke_generator import (
    Grade,
    ImageDescription,
    ImagePrompt,
    Joke,
    Rephrasings,
    StepCost,
    Subtext,
    Topic,
    _describe_image_agent,
    _find_topic_agent,
    _generate_joke_agent,
    _generate_subtext_agent,
    _google_image_agent,
    _grade_joke_agent,
    _grade_subtext_agent,
    _recent_topics,
    _rephrase_joke_agent,
    _write_image_prompt_agent,
    describe_image,
    find_topic,
    generate_image,
    generate_joke,
    generate_subtext,
    grade_asset,
    grade_joke,
    grade_subtext,
    rephrase_joke,
    save_asset,
    write_image_prompt,
)
from comedy_factory.models import StructuredOutputModel
from comedy_factory.settings import settings
from comedy_factory.utils import compose_alt_text, image_media_type

# Structured steps answer in native (JSON-schema) output mode, which an
# overriding stub model has to pick for itself (the step's own model, which
# would pick it, is replaced); image generation uses image output.
# TestModel/FunctionModel default profiles reject both. They also don't
# support thinking, so pydantic-ai would strip the steps' `thinking` setting.
_PROFILE = ModelProfile(
    supports_json_schema_output=True,
    default_structured_output_mode="native",
    supports_image_output=True,
    supports_thinking=True,
)


def canned_model(text: str) -> TestModel:
    """A model that always answers with `text`."""
    return TestModel(custom_output_text=text, profile=_PROFILE)


def _version(step: str, name: str | None = None) -> AgentVersion:
    """A committed version of `step`: its configured default unless `name` is
    given."""
    return load_version(step, name or getattr(settings, f"{step}_agent"))


def _tiny_jpeg(width: int = 64, height: int = 48) -> bytes:
    """A real (gray) JPEG so the caption step can decode it."""
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "gray").save(buffer, format="JPEG")
    return buffer.getvalue()


FAKE_IMAGE_BYTES = _tiny_jpeg()


@pytest.fixture
def log_output():
    """Capture loguru messages emitted during the test."""
    messages: list[str] = []
    handler_id = joke_generator.logger.add(messages.append, format="{message}")
    yield messages
    joke_generator.logger.remove(handler_id)


class _FakeCloudflareResponse:
    """Stand-in for httpx's response from the Workers AI image endpoint."""

    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


@pytest.fixture
def image_api_calls(monkeypatch):
    """Stub the Cloudflare image API at the httpx level; records POST calls."""
    calls = []
    payload = {
        "success": True,
        "result": {"image": base64.b64encode(FAKE_IMAGE_BYTES).decode()},
    }

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return _FakeCloudflareResponse(payload)

    monkeypatch.setattr(joke_generator.httpx, "post", fake_post)
    return calls


def test_find_topic_returns_bare_topic():
    canned = "* A fake topic.\n\nSome trailing explanation."
    with _find_topic_agent.override(model=canned_model(canned), native_tools=[]):
        topic, _ = find_topic(version=_version("find_topic"))
    assert topic.text == "A fake topic."
    assert topic.source_url is None


def test_find_topic_parses_source_url():
    canned = "A fake topic.\nhttps://example.com/story"
    with _find_topic_agent.override(model=canned_model(canned), native_tools=[]):
        topic, _ = find_topic(version=_version("find_topic"))
    assert topic.text == "A fake topic."
    assert topic.source_url == "https://example.com/story"


def test_find_topic_ignores_url_ordering():
    canned = "https://example.com/story\nA fake topic."
    with _find_topic_agent.override(model=canned_model(canned), native_tools=[]):
        topic, _ = find_topic(version=_version("find_topic"))
    assert topic.text == "A fake topic."
    assert topic.source_url == "https://example.com/story"


def test_find_topic_feeds_recent_topics_to_prompt(monkeypatch, tmp_path):
    output_dir = tmp_path / "output"
    monkeypatch.setattr(settings, "output_dir", output_dir)
    for name, topic in [
        ("20260101-000000", {"text": "An old story.", "source_url": None}),
        ("20260102-000000", "A legacy-format story."),
    ]:
        bundle_dir = output_dir / name
        bundle_dir.mkdir(parents=True)
        (bundle_dir / "metadata.json").write_text(json.dumps({"topic": topic}))

    prompts = []

    def model(messages: list, info: AgentInfo) -> ModelResponse:
        prompts.append(messages[0].parts[0].content)
        return ModelResponse(parts=[TextPart(content="A fresh topic.")])

    with _find_topic_agent.override(
        model=FunctionModel(model, profile=_PROFILE), native_tools=[]
    ):
        topic, _ = find_topic(version=_version("find_topic"))

    assert topic.text == "A fresh topic."
    [prompt] = prompts
    assert "* An old story." in prompt
    assert "* A legacy-format story." in prompt


def test_find_topic_rejects_output_without_topic():
    canned = "https://example.com/story\n\n"
    with _find_topic_agent.override(model=canned_model(canned), native_tools=[]):
        with pytest.raises(RuntimeError, match="no topic"):
            find_topic(version=_version("find_topic"))


def test_recent_topics_skips_unreadable_bundles_and_caps_the_count(monkeypatch, tmp_path):
    output_dir = tmp_path / "output"
    monkeypatch.setattr(settings, "output_dir", output_dir)
    monkeypatch.setattr(settings, "max_recent_topics", 2)
    bundles = {
        "20260101-000000": json.dumps({"topic": {"text": "The oldest story."}}),
        "20260102-000000": json.dumps({"topic": {"text": "An old story."}}),
        "20260103-000000": json.dumps({"topic": {"text": "A newer story."}}),
        "20260104-000000": json.dumps({"topic": {"text": ""}}),
        "20260105-000000": json.dumps({"subtext": {"text": "No topic recorded."}}),
        "20260106-000000": "{not json",
        "20260107-000000": None,  # bundle without metadata.json
    }
    for name, metadata in bundles.items():
        bundle_dir = output_dir / name
        bundle_dir.mkdir(parents=True)
        if metadata is not None:
            (bundle_dir / "metadata.json").write_text(metadata)
    (output_dir / "stray-file").write_text("not a bundle directory")

    # Newest first; bundles without a usable topic are skipped; capped at
    # max_recent_topics.
    assert _recent_topics() == ["A newer story.", "An old story."]


def test_generate_subtext():
    canned = Subtext(text="A fake subtext.").model_dump_json()
    with (
        _generate_subtext_agent.override(model=canned_model(canned)),
        _grade_subtext_agent.override(model=canned_model(Grade(passed=True).model_dump_json())),
    ):
        subtext, cost, grade_cost = generate_subtext(
            "A fake topic.",
            version=_version("generate_subtext"),
            grader_version=_version("grade_subtext"),
        )
    assert subtext.text == "A fake subtext."
    assert isinstance(cost, StepCost)
    assert isinstance(grade_cost, StepCost)


def _attempts_model(label: str, requests: list[list]) -> FunctionModel:
    """A generator-step model that records each request's messages and answers
    with a numbered attempt, so a test can see how many times it was asked and
    what each rewrite request contained."""

    def model(messages: list, info: AgentInfo) -> ModelResponse:
        requests.append(messages)
        text = (
            Joke(text=f"{label} {len(requests)}.", rationale="Irony.")
            if label == "Joke"
            else Subtext(text=f"{label} {len(requests)}.")
        ).model_dump_json()
        return ModelResponse(parts=[TextPart(content=text)])

    return FunctionModel(model, profile=_PROFILE)


def _grades_model(*grades: Grade) -> FunctionModel:
    """A grader-step model that hands out the given grades in order."""
    remaining = iter(grades)

    def model(messages: list, info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=next(remaining).model_dump_json())])

    return FunctionModel(model, profile=_PROFILE)


def test_generate_subtext_retries_with_grader_feedback(log_output):
    requests: list[list] = []
    with (
        _generate_subtext_agent.override(model=_attempts_model("Subtext", requests)),
        _grade_subtext_agent.override(
            model=_grades_model(
                Grade(passed=False, feedback="* Too wordy."), Grade(passed=True)
            )
        ),
    ):
        subtext, cost, grade_cost = generate_subtext(
            "A fake topic.",
            version=_version("generate_subtext"),
            grader_version=_version("grade_subtext"),
        )

    assert subtext.text == "Subtext 2."
    assert len(requests) == 2
    # The rewrite request continues the same conversation: the templated
    # prompt, the first attempt, and the grader's feedback as the retry.
    prompt, first_attempt, retry = requests[1]
    assert "A fake topic." in prompt.parts[0].content
    assert "Subtext 1." in first_attempt.parts[0].content
    assert "* Too wordy." in retry.parts[0].model_response()
    assert "Subtext failed grading (attempt 1):\n* Too wordy." in "".join(log_output)
    # Both attempts bill the generator; both gradings bill the grader.
    assert cost.output_tokens > 0
    assert grade_cost.output_tokens > 0


def test_generate_subtext_gives_up_after_max_grade_attempts(monkeypatch, log_output):
    monkeypatch.setattr(settings, "max_grade_attempts", 2)
    requests: list[list] = []
    with (
        _generate_subtext_agent.override(model=_attempts_model("Subtext", requests)),
        _grade_subtext_agent.override(
            model=_grades_model(*[Grade(passed=False, feedback="* Still wrong.")] * 2)
        ),
        pytest.raises(RuntimeError, match="Subtext failed grading after 2 attempts"),
    ):
        generate_subtext(
            "A fake topic.",
            version=_version("generate_subtext"),
            grader_version=_version("grade_subtext"),
        )
    assert len(requests) == 2
    out = "".join(log_output)
    assert "Subtext failed grading (attempt 1):\n* Still wrong." in out
    assert "Subtext failed grading (attempt 2):\n* Still wrong." in out


def test_generate_joke():
    canned = Joke(text="A fake joke.", rationale="Irony.").model_dump_json()
    with (
        _generate_joke_agent.override(model=canned_model(canned)),
        _grade_joke_agent.override(model=canned_model(Grade(passed=True).model_dump_json())),
    ):
        joke, cost, grade_cost = generate_joke(
            "A fake subtext.",
            version=_version("generate_joke"),
            grader_version=_version("grade_joke"),
        )
    assert joke.text == "A fake joke."
    assert joke.rationale == "Irony."
    assert isinstance(cost, StepCost)
    assert isinstance(grade_cost, StepCost)


def test_generate_joke_retries_with_grader_feedback(log_output):
    requests: list[list] = []
    with (
        _generate_joke_agent.override(model=_attempts_model("Joke", requests)),
        _grade_joke_agent.override(
            model=_grades_model(
                Grade(passed=False, feedback="* The funny part is not last."),
                Grade(passed=True),
            )
        ),
    ):
        joke, _, _ = generate_joke(
            "A fake subtext.",
            version=_version("generate_joke"),
            grader_version=_version("grade_joke"),
        )

    assert joke.text == "Joke 2."
    assert len(requests) == 2
    assert "* The funny part is not last." in requests[1][2].parts[0].model_response()
    assert "Joke failed grading (attempt 1):\n* The funny part is not last." in "".join(log_output)


def test_write_image_prompt():
    canned = ImagePrompt(text="A fake image prompt.").model_dump_json()
    joke = Joke(text="A fake joke.", rationale="Irony.")
    with _write_image_prompt_agent.override(model=canned_model(canned)):
        image_prompt, _ = write_image_prompt(joke, version=_version("write_image_prompt"))
    assert image_prompt.text == "A fake image prompt."


def test_rephrase_joke():
    canned = Rephrasings(texts=["Alt one.", "Alt two."]).model_dump_json()
    joke = Joke(text="A fake joke.", rationale="Irony.")
    with _rephrase_joke_agent.override(model=canned_model(canned)):
        rephrasings, _ = rephrase_joke(joke, version=_version("rephrase_joke"))
    assert rephrasings.texts == ["Alt one.", "Alt two."]


def test_rephrase_joke_requests_configured_count(monkeypatch):
    prompts = []

    def model(messages: list, info: AgentInfo) -> ModelResponse:
        prompts.append(messages[0].parts[0].content)
        text = Rephrasings(texts=["Alt one."]).model_dump_json()
        return ModelResponse(parts=[TextPart(content=text)])

    monkeypatch.setattr(settings, "rephrasing_count", 5)

    joke = Joke(text="A fake joke.", rationale="Irony.")
    with _rephrase_joke_agent.override(model=FunctionModel(model, profile=_PROFILE)):
        rephrase_joke(joke, version=_version("rephrase_joke"))

    [prompt] = prompts
    assert "write 5 alternative phrasings" in prompt
    assert "exactly 5" in prompt


def test_generate_image_cloudflare(monkeypatch, image_api_calls):
    monkeypatch.setattr(settings, "cloudflare_account_id", "test-account")
    monkeypatch.setattr(settings, "cloudflare_api_token", "test-token")

    image, cost = generate_image(
        ImagePrompt(text="A fake image prompt."), version=_version("generate_image", "flux-schnell")
    )

    assert image == FAKE_IMAGE_BYTES
    # Cloudflare reports no usage, so the step's cost is empty/unknown.
    assert cost == StepCost()
    (url, kwargs), = image_api_calls
    assert "test-account" in url
    assert url.endswith("/ai/run/@cf/black-forest-labs/flux-1-schnell")
    assert kwargs["headers"]["Authorization"] == "Bearer test-token"
    assert kwargs["json"] == {"prompt": "A fake image prompt."}


def test_generate_image_cloudflare_reports_api_failure(monkeypatch):
    payload = {"success": False, "errors": [{"code": 3040, "message": "Capacity exceeded"}]}
    monkeypatch.setattr(
        joke_generator.httpx, "post", lambda url, **kwargs: _FakeCloudflareResponse(payload)
    )
    version = _version("generate_image", "flux-schnell")

    with pytest.raises(RuntimeError, match="Image generation failed.*Capacity exceeded"):
        generate_image(ImagePrompt(text="A fake image prompt."), version=version)


def test_generate_image_google():
    def model(messages: list, info: AgentInfo) -> ModelResponse:
        assert info.model_request_parameters.allow_image_output
        return ModelResponse(
            parts=[
                FilePart(
                    content=BinaryImage(data=FAKE_IMAGE_BYTES, media_type="image/jpeg")
                )
            ]
        )

    with _google_image_agent.override(model=FunctionModel(model, profile=_PROFILE)):
        image, _ = generate_image(
            ImagePrompt(text="A fake image prompt."), version=_version("generate_image")
        )
    assert image == FAKE_IMAGE_BYTES


def test_describe_image_sends_image_to_model():
    contents = []

    def model(messages: list, info: AgentInfo) -> ModelResponse:
        contents.append(messages[0].parts[0].content)
        text = ImageDescription(text="A gray rectangle.").model_dump_json()
        return ModelResponse(parts=[TextPart(content=text)])

    with _describe_image_agent.override(model=FunctionModel(model, profile=_PROFILE)):
        description, _ = describe_image(FAKE_IMAGE_BYTES, version=_version("describe_image"))

    assert description.text == "A gray rectangle."
    [(prompt, image)] = contents
    assert "# Describe Image" in prompt
    assert image.data == FAKE_IMAGE_BYTES
    assert image.media_type == "image/jpeg"


def test_describe_image_detects_png():
    contents = []

    def model(messages: list, info: AgentInfo) -> ModelResponse:
        contents.append(messages[0].parts[0].content)
        text = ImageDescription(text="A gray rectangle.").model_dump_json()
        return ModelResponse(parts=[TextPart(content=text)])

    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), "gray").save(buffer, format="PNG")
    with _describe_image_agent.override(model=FunctionModel(model, profile=_PROFILE)):
        describe_image(buffer.getvalue(), version=_version("describe_image"))

    [(_, image)] = contents
    assert image.media_type == "image/png"


def test_image_media_type_handles_any_pillow_format():
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "gray").save(buffer, format="WEBP")
    assert image_media_type(buffer.getvalue()) == "image/webp"


def test_image_media_type_rejects_formats_without_a_media_type():
    # Pillow reads and writes QOI but registers no MIME type for it.
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "gray").save(buffer, format="QOI")
    with pytest.raises(ValueError, match="no known media type"):
        image_media_type(buffer.getvalue())


def test_compose_alt_text():
    assert (
        compose_alt_text("Aerial photo of a dam.", "A fake joke.")
        == 'Aerial photo of a dam. Caption reads: "A fake joke."'
    )


def test_render_caption_adds_bar_below_image():
    captioned = render_caption(FAKE_IMAGE_BYTES, "A fake joke.")
    image = Image.open(io.BytesIO(captioned))
    assert image.format == "JPEG"
    assert image.width == 64  # width unchanged
    assert image.height > 48  # caption bar appended below


def test_wrap_caption_wraps_to_width():
    font = ImageFont.load_default(size=24)
    caption = "one two three four five six seven eight"
    lines = _wrap_caption(caption, font, 100)
    assert len(lines) > 1
    assert all(font.getlength(line) <= 100 for line in lines)
    assert " ".join(lines) == caption  # no words lost or reordered


def test_wrap_caption_of_blank_caption_has_no_lines():
    font = ImageFont.load_default(size=24)
    assert _wrap_caption("", font, 100) == []
    assert _wrap_caption("   ", font, 100) == []


class _FrozenDatetime:
    """datetime stand-in whose clock never advances."""

    @staticmethod
    def now() -> datetime:
        return datetime(2026, 1, 1, 12, 0, 0, 123456)


def test_recaption_rewrites_from_original_and_keeps_versions(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "20260101-000000"
    bundle_dir.mkdir()
    (bundle_dir / "image-original.jpg").write_bytes(FAKE_IMAGE_BYTES)
    # Pin the clock so that every re-run lands in the same second.
    monkeypatch.setattr(recaption, "datetime", _FrozenDatetime)

    # The first rewrite fills the pipeline's canonical file name; later
    # rewrites get new datetime-stamped files — later stamps are newer — and a
    # same-second re-run falls back to a microsecond stamp rather than
    # overwriting a retained version.
    recaption.main([str(bundle_dir), "A rewritten caption."])
    recaption.main([str(bundle_dir), "A second, much longer rewritten caption."])
    recaption.main([str(bundle_dir), "A third rewritten caption, longest of them all."])

    names = {path.name for path in bundle_dir.glob("image-captioned*.jpg")}
    assert names == {
        "image-captioned.jpg",
        "image-captioned-20260101-120000.jpg",
        "image-captioned-20260101-120000-123456.jpg",
    }
    versions = [(bundle_dir / name).read_bytes() for name in names]

    assert (bundle_dir / "image-original.jpg").read_bytes() == FAKE_IMAGE_BYTES
    assert len(set(versions)) == 3
    for captioned in versions:
        image = Image.open(io.BytesIO(captioned))
        assert image.height > 48  # caption bar appended below the 48px original


def test_recaption_recomposes_alt_text_from_metadata(tmp_path, capsys):
    bundle_dir = tmp_path / "20260101-000000"
    bundle_dir.mkdir()
    (bundle_dir / "image-original.jpg").write_bytes(FAKE_IMAGE_BYTES)
    (bundle_dir / "metadata.json").write_text(
        json.dumps({"image_description": {"text": "A gray rectangle."}})
    )

    recaption.main([str(bundle_dir), "A new caption."])

    out = capsys.readouterr().out
    assert 'Alt text: A gray rectangle. Caption reads: "A new caption."' in out


def test_recaption_skips_alt_text_without_description(tmp_path, capsys):
    bundle_dir = tmp_path / "20260101-000000"
    bundle_dir.mkdir()
    (bundle_dir / "image-original.jpg").write_bytes(FAKE_IMAGE_BYTES)
    (bundle_dir / "metadata.json").write_text(json.dumps({"topic": "An old story."}))

    recaption.main([str(bundle_dir), "A new caption."])

    assert "Alt text:" not in capsys.readouterr().out


def test_recaption_requires_original_image(tmp_path):
    with pytest.raises(SystemExit):
        recaption.main([str(tmp_path), "A caption."])


def test_grade_subtext_pass():
    canned = Grade(passed=True).model_dump_json()
    with _grade_subtext_agent.override(model=canned_model(canned)):
        grade, _ = grade_subtext(
            "A fake topic.", "A fake subtext.", version=_version("grade_subtext")
        )
    assert grade.passed
    assert grade.feedback == ""


def test_grade_subtext_fail():
    canned = Grade(passed=False, feedback="* Not a simple sentence.").model_dump_json()
    with _grade_subtext_agent.override(model=canned_model(canned)):
        grade, _ = grade_subtext(
            "A fake topic.", "A fake subtext.", version=_version("grade_subtext")
        )
    assert not grade.passed
    assert grade.feedback == "* Not a simple sentence."


def test_grade_joke_fail():
    canned = Grade(passed=False, feedback="* The funny part is not last.").model_dump_json()
    with _grade_joke_agent.override(model=canned_model(canned)):
        grade, _ = grade_joke("A fake subtext.", "A fake joke.", version=_version("grade_joke"))
    assert not grade.passed
    assert grade.feedback == "* The funny part is not last."


def test_steps_send_their_configured_model_settings():
    seen = []

    def model(messages: list, info: AgentInfo) -> ModelResponse:
        seen.append(info)
        return ModelResponse(parts=[TextPart(content=Grade(passed=True).model_dump_json())])

    version = AgentVersion(
        "grade_joke",
        "test",
        AgentSpec(
            model="google-cloud:gemini-3.5-flash",
            model_settings={"thinking": "low", "temperature": 0.0},
        ),
    )
    with _grade_joke_agent.override(model=FunctionModel(model, profile=_PROFILE)):
        grade_joke("A fake subtext.", "A fake joke.", version=version)

    [info] = seen
    # pydantic-ai resolves the unified `thinking` setting into the request
    # parameters; the remaining settings reach the model as given.
    assert info.model_request_parameters.thinking == "low"
    assert info.model_settings == {"temperature": 0.0}


@pytest.mark.parametrize(
    ("supports_json_schema_output", "output_mode"), [(True, "native"), (False, "prompted")]
)
def test_structured_steps_request_the_output_mode_their_model_supports(
    supports_json_schema_output, output_mode
):
    seen = []

    def model(messages: list, info: AgentInfo) -> ModelResponse:
        seen.append(info)
        return ModelResponse(parts=[TextPart(content=Grade(passed=True).model_dump_json())])

    # A model with native (JSON-schema) output is asked for it; one without,
    # such as Claude Haiku, is asked for prompted output instead. Tool-call
    # output is never used: Anthropic rejects forced tool use while thinking is
    # enabled.
    stub = FunctionModel(
        model,
        profile=ModelProfile(
            supports_json_schema_output=supports_json_schema_output, supports_thinking=True
        ),
    )
    with _grade_joke_agent.override(model=StructuredOutputModel(stub)):
        grade, _ = grade_joke("A fake subtext.", "A fake joke.", version=_version("grade_joke"))

    assert grade.passed
    [info] = seen
    assert info.model_request_parameters.output_mode == output_mode


@pytest.mark.parametrize(
    ("model_name", "output_mode"),
    [
        ("anthropic:claude-sonnet-5-5", "native"),
        ("anthropic:claude-opus-5-5", "native"),
        ("anthropic:claude-haiku-5-5", "prompted"),
    ],
)
def test_structured_output_mode_follows_claude_model_profiles(monkeypatch, model_name, output_mode):
    # Building the model needs a key; no request is made.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
    model = StructuredOutputModel(infer_model(model_name))
    assert model.profile["default_structured_output_mode"] == output_mode


def test_step_cost_addition_sums_tokens_and_known_costs():
    priced = StepCost(input_tokens=10, output_tokens=5, usd=0.01)
    unpriced = StepCost(input_tokens=1, output_tokens=2)
    total = priced + unpriced
    assert total.input_tokens == 11
    assert total.output_tokens == 7
    assert total.usd == 0.01
    assert (unpriced + unpriced).usd is None


def test_grade_asset_is_a_passthrough():
    grade = grade_asset(
        Topic(text="Any topic."),
        Subtext(text="Any subtext."),
        Joke(text="Any joke.", rationale="Any rationale."),
        b"any-image-bytes",
        ImageDescription(text="Any description."),
    )
    assert grade.passed


def test_save_asset(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "output_dir", tmp_path / "output")

    bundle_dir = save_asset(
        topic=Topic(text="A fake topic.", source_url="https://example.com/story"),
        subtext=Subtext(text="A fake subtext."),
        joke=Joke(text="A fake joke.", rationale="Irony."),
        rephrasings=Rephrasings(texts=["Alt one.", "Alt two."]),
        image_prompt=ImagePrompt(text="A fake image prompt."),
        image=b"original-bytes",
        captioned_image=FAKE_IMAGE_BYTES,
        rephrased_images=[b"alt-1-bytes", b"alt-2-bytes"],
        image_description=ImageDescription(text="A gray rectangle."),
        costs={
            "generate_joke": StepCost(input_tokens=100, output_tokens=20, usd=0.001),
            "generate_image": StepCost(),
        },
        evaluation=Grade(passed=True),
        versions={"generate_joke": _version("generate_joke")},
    )

    assert bundle_dir.parent == tmp_path / "output"
    assert (bundle_dir / "image-original.jpg").read_bytes() == b"original-bytes"
    assert (bundle_dir / "image-captioned.jpg").read_bytes() == FAKE_IMAGE_BYTES
    assert (bundle_dir / "image-rephrased-1.jpg").read_bytes() == b"alt-1-bytes"
    assert (bundle_dir / "image-rephrased-2.jpg").read_bytes() == b"alt-2-bytes"
    metadata = json.loads((bundle_dir / "metadata.json").read_text())
    assert metadata["topic"]["text"] == "A fake topic."
    assert metadata["topic"]["source_url"] == "https://example.com/story"
    assert metadata["subtext"]["text"] == "A fake subtext."
    assert metadata["joke"]["text"] == "A fake joke."
    assert metadata["rephrasings"] == ["Alt one.", "Alt two."]
    assert metadata["image_prompt"]["text"] == "A fake image prompt."
    assert metadata["image_description"]["text"] == "A gray rectangle."
    assert metadata["alt_text"] == 'A gray rectangle. Caption reads: "A fake joke."'
    assert metadata["evaluation"]["passed"] is True
    assert metadata["costs"]["steps"]["generate_joke"]["input_tokens"] == 100
    assert metadata["costs"]["steps"]["generate_image"]["usd"] is None
    assert metadata["costs"]["total"] == {
        "input_tokens": 100,
        "output_tokens": 20,
        "usd": 0.001,
    }
    assert metadata["agents"] == {
        "generate_joke": {
            "version": "gemini-flash",
            "model": "google-cloud:gemini-3.5-flash",
            "model_settings": {"thinking": "high"},
        }
    }
    assert "created_at" in metadata


def _override_step_models(model) -> ExitStack:
    """Swap the same stub model into every step's agent except find_topic's,
    whose tests also override away its native tools."""
    stack = ExitStack()
    for agent in (
        _generate_subtext_agent,
        _grade_subtext_agent,
        _generate_joke_agent,
        _grade_joke_agent,
        _rephrase_joke_agent,
        _write_image_prompt_agent,
        _google_image_agent,
        _describe_image_agent,
    ):
        stack.enter_context(agent.override(model=model))
    return stack


_FAKE_IMAGE_RESPONSE = ModelResponse(
    parts=[FilePart(content=BinaryImage(data=FAKE_IMAGE_BYTES, media_type="image/jpeg"))]
)


def _pipeline_model(
    grade_for: Callable[[str], Grade] | None = None,
) -> Callable[[list, AgentInfo], ModelResponse]:
    """A FunctionModel function that plays every agent step, told apart
    by their output schemas. Every evaluation gate passes unless `grade_for`
    is given to pick the Grade from the evaluation prompt."""

    def model(messages: list, info: AgentInfo) -> ModelResponse:
        if info.model_request_parameters.allow_image_output:
            return _FAKE_IMAGE_RESPONSE
        output_object = info.model_request_parameters.output_object
        schema_name = output_object.json_schema["title"] if output_object is not None else None
        if schema_name == "Grade":
            prompt = messages[0].parts[0].content
            grade = grade_for(prompt) if grade_for is not None else Grade(passed=True)
            text = grade.model_dump_json()
        elif schema_name == "Joke":
            text = Joke(text="A fake joke.", rationale="Irony.").model_dump_json()
        elif schema_name == "Subtext":
            text = Subtext(text="A fake subtext.").model_dump_json()
        elif schema_name == "ImagePrompt":
            text = ImagePrompt(text="A fake image prompt.").model_dump_json()
        elif schema_name == "ImageDescription":
            text = ImageDescription(text="A gray rectangle.").model_dump_json()
        elif schema_name == "Rephrasings":
            text = Rephrasings(texts=["Alt one.", "Alt two."]).model_dump_json()
        else:
            raise AssertionError(f"Unexpected request schema: {schema_name}")
        return ModelResponse(parts=[TextPart(content=text)])

    return model


def test_main_smoke(monkeypatch, tmp_path, log_output):
    monkeypatch.setattr(settings, "output_dir", tmp_path / "output")
    with (
        _override_step_models(FunctionModel(_pipeline_model(), profile=_PROFILE)),
        _find_topic_agent.override(model=canned_model("A fake topic."), native_tools=[]),
    ):
        joke_generator.main()

    out = "".join(log_output)
    assert "Topic: A fake topic." in out
    assert "Subtext: A fake subtext." in out
    assert "Joke: A fake joke." in out
    assert "Rationale: Irony." in out
    assert "Rephrasing 1: Alt one." in out
    assert "Rephrasing 2: Alt two." in out
    assert "Image prompt: A fake image prompt." in out
    assert 'Alt text: A gray rectangle. Caption reads: "A fake joke."' in out
    assert "estimated cost" in out
    assert "Asset bundle saved to" in out

    [bundle_dir] = list((tmp_path / "output").iterdir())
    assert (bundle_dir / "image-original.jpg").read_bytes() == FAKE_IMAGE_BYTES
    # The captioned images are the generated image plus a rendered caption bar.
    for name in ("image-captioned", "image-rephrased-1", "image-rephrased-2"):
        saved = Image.open(bundle_dir / f"{name}.jpg")
        assert saved.format == "JPEG"
        assert saved.width == 64
        assert saved.height > 48
    metadata = json.loads((bundle_dir / "metadata.json").read_text())
    assert metadata["joke"]["text"] == "A fake joke."
    assert metadata["rephrasings"] == ["Alt one.", "Alt two."]
    assert metadata["alt_text"] == 'A gray rectangle. Caption reads: "A fake joke."'
    assert set(metadata["costs"]["steps"]) == {
        "find_topic",
        "generate_subtext",
        "grade_subtext",
        "generate_joke",
        "grade_joke",
        "rephrase_joke",
        "write_image_prompt",
        "generate_image",
        "describe_image",
    }
    assert set(metadata["costs"]["total"]) == {"input_tokens", "output_tokens", "usd"}
    assert set(metadata["agents"]) == set(STEPS)
    assert metadata["agents"]["generate_image"]["version"] == "gemini-flash-image"


def test_main_retries_failed_joke_grading(log_output):
    joke_grades = iter(
        [Grade(passed=False, feedback="* The funny part is not last."), Grade(passed=True)]
    )

    def grade_for(prompt: str) -> Grade:
        return next(joke_grades) if "# Evaluate Joke" in prompt else Grade(passed=True)

    with (
        _override_step_models(FunctionModel(_pipeline_model(grade_for), profile=_PROFILE)),
        _find_topic_agent.override(model=canned_model("A fake topic."), native_tools=[]),
    ):
        joke_generator.main()

    out = "".join(log_output)
    assert "Joke failed grading (attempt 1):\n* The funny part is not last." in out
    assert "Joke: A fake joke." in out


def test_main_retries_failed_subtext_grading(log_output):
    subtext_grades = iter(
        [Grade(passed=False, feedback="* Not a simple sentence."), Grade(passed=True)]
    )

    def grade_for(prompt: str) -> Grade:
        return next(subtext_grades) if "# Evaluate Subtext" in prompt else Grade(passed=True)

    with (
        _override_step_models(FunctionModel(_pipeline_model(grade_for), profile=_PROFILE)),
        _find_topic_agent.override(model=canned_model("A fake topic."), native_tools=[]),
    ):
        joke_generator.main()

    out = "".join(log_output)
    assert "Subtext failed grading (attempt 1):\n* Not a simple sentence." in out
    assert "Subtext: A fake subtext." in out


@pytest.mark.parametrize(
    ("heading", "step"),
    [("# Evaluate Subtext", "Subtext"), ("# Evaluate Joke", "Joke")],
)
def test_main_gives_up_after_max_grade_attempts(monkeypatch, log_output, heading, step):
    monkeypatch.setattr(settings, "max_grade_attempts", 2)

    def grade_for(prompt: str) -> Grade:
        if heading in prompt:
            return Grade(passed=False, feedback="* Still wrong.")
        return Grade(passed=True)

    with (
        _override_step_models(FunctionModel(_pipeline_model(grade_for), profile=_PROFILE)),
        _find_topic_agent.override(model=canned_model("A fake topic."), native_tools=[]),
        pytest.raises(RuntimeError, match=f"{step} failed grading after 2 attempts"),
    ):
        joke_generator.main()

    out = "".join(log_output)
    assert f"{step} failed grading (attempt 1):\n* Still wrong." in out
    assert f"{step} failed grading (attempt 2):\n* Still wrong." in out


def test_main_rejects_an_unknown_version_before_any_model_call(monkeypatch):
    calls = []

    def model(messages: list, info: AgentInfo) -> ModelResponse:
        calls.append(info)
        return ModelResponse(parts=[TextPart(content="A fake topic.")])

    monkeypatch.setattr(settings, "describe_image_agent", "claud-opus")
    message = "Unknown describe_image agent version 'claud-opus'; available: gemini-flash"
    with _find_topic_agent.override(model=FunctionModel(model, profile=_PROFILE), native_tools=[]):
        with pytest.raises(AgentVersionError, match=re.escape(message)):
            joke_generator.main()
    assert calls == []
