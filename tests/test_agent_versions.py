"""Tests for loading and checking the per-step agent version files."""

import re

import pytest

from comedy_factory.agent_versions import (
    STEPS,
    AgentVersionError,
    available_versions,
    check_version_file,
    load_version,
)
from comedy_factory.settings import settings

GEMINI = "google-cloud:gemini-3.5-flash"
FLUX = "cloudflare:@cf/black-forest-labs/flux-1-schnell"

# The repo's agents/ directory; tests repoint the setting only via the
# `agents_dir` fixture, so this is read before any of them runs.
COMMITTED = settings.agents_dir


@pytest.fixture
def agents_dir(monkeypatch, tmp_path):
    """Point the agents directory at an empty temp dir."""
    monkeypatch.setattr(settings, "agents_dir", tmp_path / "agents")
    return tmp_path / "agents"


def write_version(agents_dir, step, name, text):
    path = agents_dir / step / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_load_version_reads_model_and_settings(agents_dir):
    write_version(
        agents_dir,
        "generate_joke",
        "gemini-flash",
        f"model: {GEMINI}\nmodel_settings:\n  thinking: high\n",
    )
    version = load_version("generate_joke", "gemini-flash")
    assert (version.step, version.name, version.model) == ("generate_joke", "gemini-flash", GEMINI)
    assert version.as_metadata() == {
        "version": "gemini-flash",
        "model": GEMINI,
        "model_settings": {"thinking": "high"},
    }


def test_version_without_settings_has_empty_settings(agents_dir):
    write_version(agents_dir, "find_topic", "gemini-flash", f"model: {GEMINI}\n")
    assert load_version("find_topic", "gemini-flash").model_settings == {}


def test_load_version_rejects_unknown_names_listing_available(agents_dir):
    write_version(agents_dir, "generate_joke", "gemini-flash", f"model: {GEMINI}\n")
    message = "Unknown generate_joke agent version 'claud-opus'; available: gemini-flash"
    with pytest.raises(AgentVersionError, match=re.escape(message)):
        load_version("generate_joke", "claud-opus")


@pytest.mark.parametrize("name", ["../grade_joke/gemini-flash", "gemini-flash.yaml", ""])
def test_load_version_rejects_path_like_names(agents_dir, name):
    write_version(agents_dir, "grade_joke", "gemini-flash", f"model: {GEMINI}\n")
    write_version(agents_dir, "generate_joke", "gemini-flash", f"model: {GEMINI}\n")
    with pytest.raises(AgentVersionError, match="Unknown generate_joke agent version"):
        load_version("generate_joke", name)


def test_missing_step_folder_has_no_versions(agents_dir):
    assert available_versions("describe_image") == []
    with pytest.raises(AgentVersionError, match="available: none"):
        load_version("describe_image", "gemini-flash")


def test_available_versions_lists_only_yaml_files(agents_dir):
    write_version(agents_dir, "generate_joke", "b-version", f"model: {GEMINI}\n")
    write_version(agents_dir, "generate_joke", "a-version", f"model: {GEMINI}\n")
    (agents_dir / "generate_joke" / ".DS_Store").write_bytes(b"\0")
    (agents_dir / "generate_joke" / "README.md").write_text("notes")
    assert available_versions("generate_joke") == ["a-version", "b-version"]


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("model: [", "not valid YAML"),
        ("", "must be a mapping"),
        ("- a\n- b\n", "must be a mapping"),
        ("model_settings:\n  thinking: high\n", "'model' is required"),
        ("model:\n", "'model' is required"),
        (f"model: {GEMINI}\ninstructions: Be funny.\n", "unexpected keys: instructions"),
        (f"model: {GEMINI}\nmodel_settings:\n  thinking: hgh\n", "'hgh'"),
        (
            f"model: {GEMINI}\nmodel_settings:\n  thinkng: high\n",
            "unknown model_settings keys: thinkng",
        ),
        (
            f"model: {GEMINI}\nmodel_settings:\n  anthropic_betas: []\n",
            "unknown model_settings keys: anthropic_betas",
        ),
        ("model: nope:some-model\n", "can't be built"),
    ],
)
def test_check_rejects_broken_version_files(agents_dir, text, problem):
    path = write_version(agents_dir, "generate_joke", "broken", text)
    with pytest.raises(AgentVersionError, match=re.escape(problem)) as excinfo:
        check_version_file("generate_joke", path)
    assert "generate_joke agent version 'broken'" in str(excinfo.value)


def test_check_allows_provider_specific_settings(agents_dir):
    gemini = write_version(
        agents_dir,
        "generate_joke",
        "gemini",
        f"model: {GEMINI}\nmodel_settings:\n  google_thinking_config:\n    include_thoughts: true\n",
    )
    claude = write_version(
        agents_dir,
        "generate_joke",
        "claude",
        "model: anthropic:claude-opus-5-5\nmodel_settings:\n  anthropic_betas: []\n",
    )
    assert check_version_file("generate_joke", gemini).model == GEMINI
    assert check_version_file("generate_joke", claude).model == "anthropic:claude-opus-5-5"


def test_check_allows_only_common_settings_for_other_providers(agents_dir):
    common = write_version(
        agents_dir, "generate_joke", "common", "model: test\nmodel_settings:\n  thinking: high\n"
    )
    google = write_version(
        agents_dir,
        "generate_joke",
        "google",
        "model: test\nmodel_settings:\n  google_thinking_config:\n    include_thoughts: true\n",
    )
    assert check_version_file("generate_joke", common).model == "test"
    with pytest.raises(AgentVersionError, match="unknown model_settings keys: google_thinking_config"):
        check_version_file("generate_joke", google)


def test_check_reports_a_missing_anthropic_key(agents_dir, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    path = write_version(
        agents_dir, "generate_joke", "claude", "model: anthropic:claude-opus-5-5\n"
    )
    with pytest.raises(AgentVersionError, match="ANTHROPIC_API_KEY"):
        check_version_file("generate_joke", path)


def test_cloudflare_models_are_allowed_for_generate_image(agents_dir):
    path = write_version(agents_dir, "generate_image", "flux-schnell", f"model: {FLUX}\n")
    assert check_version_file("generate_image", path).model == FLUX


def test_cloudflare_models_are_only_for_generate_image(agents_dir):
    path = write_version(agents_dir, "describe_image", "flux", f"model: {FLUX}\n")
    with pytest.raises(AgentVersionError, match="cloudflare: models are only for generate_image"):
        check_version_file("describe_image", path)


def test_cloudflare_models_take_no_settings(agents_dir):
    path = write_version(
        agents_dir, "generate_image", "flux", f"model: {FLUX}\nmodel_settings:\n  temperature: 0.5\n"
    )
    with pytest.raises(AgentVersionError, match="cloudflare: models take no model_settings"):
        check_version_file("generate_image", path)


INITIAL_VERSIONS = [
    ("find_topic", "gemini-flash", GEMINI, {}),
    ("generate_subtext", "gemini-flash", GEMINI, {"thinking": "medium"}),
    ("grade_subtext", "gemini-flash", GEMINI, {"thinking": "medium"}),
    ("generate_joke", "gemini-flash", GEMINI, {"thinking": "high"}),
    ("grade_joke", "gemini-flash", GEMINI, {"thinking": "medium"}),
    ("rephrase_joke", "gemini-flash", GEMINI, {"thinking": "low"}),
    ("write_image_prompt", "gemini-flash", GEMINI, {"thinking": "low"}),
    ("generate_image", "gemini-flash-image", "google-cloud:gemini-3.1-flash-image", {}),
    ("generate_image", "flux-schnell", FLUX, {}),
    ("describe_image", "gemini-flash", GEMINI, {"thinking": False}),
]


@pytest.mark.parametrize(("step", "name", "model", "model_settings"), INITIAL_VERSIONS)
def test_initial_versions_carry_over_todays_settings(step, name, model, model_settings):
    version = load_version(step, name)
    assert (version.model, version.model_settings) == (model, model_settings)


@pytest.mark.parametrize(
    "path",
    sorted(COMMITTED.glob("*/*.yaml")),
    ids=lambda path: f"{path.parent.name}/{path.stem}",
)
def test_committed_version_files_pass_the_checks(path):
    check_version_file(path.parent.name, path)


def test_agents_folder_has_one_folder_per_step():
    assert {path.name for path in COMMITTED.iterdir() if path.is_dir()} == set(STEPS)

