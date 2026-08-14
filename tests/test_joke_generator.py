"""Smoke tests for the joke generator pipeline — no real API calls.

Model stubbing follows pydantic-ai's recommended seams:

* The scan-news Agent is swapped via `agent.override(model=...)`; its WebSearch
  native tool is overridden away because TestModel can't emulate built-in tools.
* The direct-call steps read their per-step model setting (e.g.
  `settings.generate_subtext_model`) at call time, so tests monkeypatch that
  attribute with a TestModel/FunctionModel instance.
"""

import base64
import io
import json

import pytest
from PIL import Image, ImageFont
from pydantic_ai.messages import BinaryImage, FilePart, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.profiles import ModelProfile

from comedy_factory import joke_generator, recaption
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
    _find_topic_agent,
    describe_image,
    grade_asset,
    generate_image,
    generate_joke,
    generate_subtext,
    grade_joke,
    grade_subtext,
    rephrase_joke,
    save_asset,
    find_topic,
    write_image_prompt,
)
from comedy_factory.settings import settings
from comedy_factory.utils import compose_alt_text, image_media_type

# Grading uses output_mode="native" and image generation uses image output;
# TestModel/FunctionModel default profiles reject both.
_PROFILE = ModelProfile(supports_json_schema_output=True, supports_image_output=True)


def canned_model(text: str) -> TestModel:
    """A model that always answers with `text`."""
    return TestModel(custom_output_text=text, profile=_PROFILE)


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


@pytest.fixture
def image_api_calls(monkeypatch):
    """Stub the Cloudflare image API at the httpx level; records POST calls."""
    calls = []
    payload = {
        "success": True,
        "result": {"image": base64.b64encode(FAKE_IMAGE_BYTES).decode()},
    }

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse()

    monkeypatch.setattr(joke_generator.httpx, "post", fake_post)
    return calls


def test_find_topic_returns_bare_topic():
    canned = "* A fake topic.\n\nSome trailing explanation."
    with _find_topic_agent.override(model=canned_model(canned), native_tools=[]):
        topic, _ = find_topic()
    assert topic.text == "A fake topic."
    assert topic.source_url is None


def test_find_topic_parses_source_url():
    canned = "A fake topic.\nhttps://example.com/story"
    with _find_topic_agent.override(model=canned_model(canned), native_tools=[]):
        topic, _ = find_topic()
    assert topic.text == "A fake topic."
    assert topic.source_url == "https://example.com/story"


def test_find_topic_ignores_url_ordering():
    canned = "https://example.com/story\nA fake topic."
    with _find_topic_agent.override(model=canned_model(canned), native_tools=[]):
        topic, _ = find_topic()
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
        topic, _ = find_topic()

    assert topic.text == "A fresh topic."
    [prompt] = prompts
    assert "* An old story." in prompt
    assert "* A legacy-format story." in prompt


def test_generate_subtext(monkeypatch):
    canned = Subtext(text="A fake subtext.").model_dump_json()
    monkeypatch.setattr(settings, "generate_subtext_model", canned_model(canned))
    subtext, history, cost = generate_subtext("A fake topic.")
    assert subtext.text == "A fake subtext."
    assert len(history) == 2  # templated prompt + model response
    assert isinstance(cost, StepCost)


def test_generate_subtext_retry_extends_history(monkeypatch):
    canned = Subtext(text="A better subtext.").model_dump_json()
    monkeypatch.setattr(settings, "generate_subtext_model", canned_model(canned))
    _, history, _ = generate_subtext("A fake topic.")
    subtext, history, _ = generate_subtext(
        "A fake topic.", history=history, feedback="* Too wordy."
    )
    assert subtext.text == "A better subtext."
    assert len(history) == 4  # prompt, first attempt, feedback, rewrite
    assert "Too wordy" in history[2].parts[0].content


def test_generate_joke(monkeypatch):
    canned = Joke(text="A fake joke.", rationale="Irony.").model_dump_json()
    monkeypatch.setattr(settings, "generate_joke_model", canned_model(canned))
    joke, history, _ = generate_joke("A fake subtext.")
    assert joke.text == "A fake joke."
    assert joke.rationale == "Irony."
    assert len(history) == 2


def test_write_image_prompt(monkeypatch):
    canned = ImagePrompt(text="A fake image prompt.").model_dump_json()
    monkeypatch.setattr(settings, "write_image_prompt_model", canned_model(canned))
    joke = Joke(text="A fake joke.", rationale="Irony.")
    image_prompt, _ = write_image_prompt(joke)
    assert image_prompt.text == "A fake image prompt."


def test_rephrase_joke(monkeypatch):
    canned = Rephrasings(texts=["Alt one.", "Alt two."]).model_dump_json()
    monkeypatch.setattr(settings, "rephrase_joke_model", canned_model(canned))
    rephrasings, _ = rephrase_joke(Joke(text="A fake joke.", rationale="Irony."))
    assert rephrasings.texts == ["Alt one.", "Alt two."]


def test_rephrase_joke_requests_configured_count(monkeypatch):
    prompts = []

    def model(messages: list, info: AgentInfo) -> ModelResponse:
        prompts.append(messages[0].parts[0].content)
        text = Rephrasings(texts=["Alt one."]).model_dump_json()
        return ModelResponse(parts=[TextPart(content=text)])

    monkeypatch.setattr(settings, "rephrasing_count", 5)
    monkeypatch.setattr(
        settings, "rephrase_joke_model", FunctionModel(model, profile=_PROFILE)
    )

    rephrase_joke(Joke(text="A fake joke.", rationale="Irony."))

    [prompt] = prompts
    assert "write 5 alternative phrasings" in prompt
    assert "exactly 5" in prompt


def test_generate_image_cloudflare(monkeypatch, image_api_calls):
    monkeypatch.setattr(settings, "image_provider", "cloudflare")
    monkeypatch.setattr(settings, "cloudflare_account_id", "test-account")
    monkeypatch.setattr(settings, "cloudflare_api_token", "test-token")

    image, cost = generate_image(ImagePrompt(text="A fake image prompt."))

    assert image == FAKE_IMAGE_BYTES
    # Cloudflare reports no usage, so the step's cost is empty/unknown.
    assert cost == StepCost()
    (url, kwargs), = image_api_calls
    assert "test-account" in url
    assert url.endswith(settings.cloudflare_image_model)
    assert kwargs["headers"]["Authorization"] == "Bearer test-token"
    assert kwargs["json"] == {"prompt": "A fake image prompt."}


def test_generate_image_google(monkeypatch):
    def model(messages: list, info: AgentInfo) -> ModelResponse:
        assert info.model_request_parameters.allow_image_output
        return ModelResponse(
            parts=[
                FilePart(
                    content=BinaryImage(data=FAKE_IMAGE_BYTES, media_type="image/jpeg")
                )
            ]
        )

    monkeypatch.setattr(settings, "image_provider", "google")
    monkeypatch.setattr(settings, "google_image_model", FunctionModel(model, profile=_PROFILE))

    image, _ = generate_image(ImagePrompt(text="A fake image prompt."))
    assert image == FAKE_IMAGE_BYTES


def test_describe_image_sends_image_to_model(monkeypatch):
    contents = []

    def model(messages: list, info: AgentInfo) -> ModelResponse:
        contents.append(messages[0].parts[0].content)
        text = ImageDescription(text="A gray rectangle.").model_dump_json()
        return ModelResponse(parts=[TextPart(content=text)])

    monkeypatch.setattr(
        settings, "describe_image_model", FunctionModel(model, profile=_PROFILE)
    )

    description, _ = describe_image(FAKE_IMAGE_BYTES)

    assert description.text == "A gray rectangle."
    [(prompt, image)] = contents
    assert "# Describe Image" in prompt
    assert image.data == FAKE_IMAGE_BYTES
    assert image.media_type == "image/jpeg"


def test_describe_image_detects_png(monkeypatch):
    contents = []

    def model(messages: list, info: AgentInfo) -> ModelResponse:
        contents.append(messages[0].parts[0].content)
        text = ImageDescription(text="A gray rectangle.").model_dump_json()
        return ModelResponse(parts=[TextPart(content=text)])

    monkeypatch.setattr(
        settings, "describe_image_model", FunctionModel(model, profile=_PROFILE)
    )

    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), "gray").save(buffer, format="PNG")
    describe_image(buffer.getvalue())

    [(_, image)] = contents
    assert image.media_type == "image/png"


def test_image_media_type_handles_any_pillow_format():
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "gray").save(buffer, format="WEBP")
    assert image_media_type(buffer.getvalue()) == "image/webp"


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


def test_recaption_rewrites_from_original_and_keeps_versions(tmp_path, capsys):
    bundle_dir = tmp_path / "20260101-000000"
    bundle_dir.mkdir()
    (bundle_dir / "image-original.jpg").write_bytes(FAKE_IMAGE_BYTES)

    # The first rewrite fills the pipeline's canonical file name; later
    # rewrites get new datetime-stamped files — later stamps are newer.
    recaption.main([str(bundle_dir), "A rewritten caption."])
    recaption.main([str(bundle_dir), "A second, much longer rewritten caption."])
    recaption.main([str(bundle_dir), "A third rewritten caption, longest of them all."])

    stamped = sorted(bundle_dir.glob("image-captioned-*.jpg"))
    assert len(stamped) == 2
    versions = [(bundle_dir / "image-captioned.jpg").read_bytes()] + [
        path.read_bytes() for path in stamped
    ]

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


def test_grade_subtext_pass(monkeypatch):
    canned = Grade(passed=True).model_dump_json()
    monkeypatch.setattr(settings, "grade_subtext_model", canned_model(canned))
    grade, _ = grade_subtext("A fake topic.", "A fake subtext.")
    assert grade.passed
    assert grade.feedback == ""


def test_grade_subtext_fail(monkeypatch):
    canned = Grade(passed=False, feedback="* Not a simple sentence.").model_dump_json()
    monkeypatch.setattr(settings, "grade_subtext_model", canned_model(canned))
    grade, _ = grade_subtext("A fake topic.", "A fake subtext.")
    assert not grade.passed
    assert grade.feedback == "* Not a simple sentence."


def test_grade_joke_fail(monkeypatch):
    canned = Grade(passed=False, feedback="* The funny part is not last.").model_dump_json()
    monkeypatch.setattr(settings, "grade_joke_model", canned_model(canned))
    grade, _ = grade_joke("A fake subtext.", "A fake joke.")
    assert not grade.passed
    assert grade.feedback == "* The funny part is not last."


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
    assert "created_at" in metadata


def _set_all_step_models(monkeypatch, model):
    """Point every direct-call step's model setting at the same stub."""
    for name in (
        "generate_subtext_model",
        "grade_subtext_model",
        "generate_joke_model",
        "grade_joke_model",
        "rephrase_joke_model",
        "write_image_prompt_model",
        "describe_image_model",
        "google_image_model",
    ):
        monkeypatch.setattr(settings, name, model)


_FAKE_IMAGE_RESPONSE = ModelResponse(
    parts=[FilePart(content=BinaryImage(data=FAKE_IMAGE_BYTES, media_type="image/jpeg"))]
)


def _pipeline_model(messages: list, info: AgentInfo) -> ModelResponse:
    """Play every direct-call step, told apart by their output schemas."""
    if info.model_request_parameters.allow_image_output:
        return _FAKE_IMAGE_RESPONSE
    output_object = info.model_request_parameters.output_object
    schema_name = output_object.name if output_object is not None else None
    if schema_name == "Grade":
        text = Grade(passed=True).model_dump_json()
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


def test_main_smoke(monkeypatch, tmp_path, log_output):
    _set_all_step_models(monkeypatch, FunctionModel(_pipeline_model, profile=_PROFILE))
    monkeypatch.setattr(settings, "output_dir", tmp_path / "output")
    with _find_topic_agent.override(model=canned_model("A fake topic."), native_tools=[]):
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


def test_main_retries_failed_joke_grading(monkeypatch, tmp_path, log_output):
    joke_grades = iter(
        [Grade(passed=False, feedback="* The funny part is not last."), Grade(passed=True)]
    )

    def model(messages: list, info: AgentInfo) -> ModelResponse:
        if info.model_request_parameters.allow_image_output:
            return _FAKE_IMAGE_RESPONSE
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
        elif schema_name == "ImagePrompt":
            text = ImagePrompt(text="A fake image prompt.").model_dump_json()
        elif schema_name == "ImageDescription":
            text = ImageDescription(text="A gray rectangle.").model_dump_json()
        elif schema_name == "Rephrasings":
            text = Rephrasings(texts=["Alt one.", "Alt two."]).model_dump_json()
        else:
            raise AssertionError(f"Unexpected request schema: {schema_name}")
        return ModelResponse(parts=[TextPart(content=text)])

    _set_all_step_models(monkeypatch, FunctionModel(model, profile=_PROFILE))
    monkeypatch.setattr(settings, "output_dir", tmp_path / "output")
    with _find_topic_agent.override(model=canned_model("A fake topic."), native_tools=[]):
        joke_generator.main()

    out = "".join(log_output)
    assert "Joke failed grading (attempt 1):\n* The funny part is not last." in out
    assert "Joke: A fake joke." in out
