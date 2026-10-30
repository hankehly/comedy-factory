"""Tests for the settings that pick each step's agent version."""

from comedy_factory.agent_versions import STEPS, available_versions
from comedy_factory.settings import Settings, settings


def test_every_step_has_a_version_setting_with_a_committed_default():
    steps = {
        name.removesuffix("_agent") for name in Settings.model_fields if name.endswith("_agent")
    }
    assert steps == set(STEPS)
    for step in STEPS:
        assert getattr(settings, f"{step}_agent") in available_versions(step)


def test_default_versions():
    defaults = {step: getattr(Settings(), f"{step}_agent") for step in STEPS}
    assert defaults == {
        **dict.fromkeys(STEPS, "gemini-flash"),
        "generate_image": "gemini-flash-image",
    }


def test_env_var_selects_a_steps_version(monkeypatch):
    monkeypatch.setenv("COMEDY_FACTORY_GENERATE_JOKE_AGENT", "claude-opus")
    assert Settings().generate_joke_agent == "claude-opus"


def test_no_setting_holds_a_model():
    assert [name for name in Settings.model_fields if "model" in name] == []
