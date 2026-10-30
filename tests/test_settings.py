"""Tests for the settings that pick each step's agent version, and the check
that rejects unknown COMEDY_FACTORY_* environment variables."""

import pytest

from comedy_factory.agent_versions import STEPS, available_versions
from comedy_factory.settings import Settings, check_env, settings


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


def test_unknown_env_vars_fail_naming_them():
    with pytest.raises(ValueError, match="COMEDY_FACTORY_GENERATE_IMAGE_AGNT"):
        check_env({"COMEDY_FACTORY_GENERATE_IMAGE_AGNT": "flux-schnell"})


@pytest.mark.parametrize(
    ("name", "replacement"),
    [
        ("COMEDY_FACTORY_GENERATE_JOKE__MODEL", "COMEDY_FACTORY_GENERATE_JOKE_AGENT"),
        (
            "COMEDY_FACTORY_GENERATE_JOKE__MODEL_SETTINGS__THINKING",
            "COMEDY_FACTORY_GENERATE_JOKE_AGENT",
        ),
        ("COMEDY_FACTORY_GENERATE_JOKE_MODEL", "COMEDY_FACTORY_GENERATE_JOKE_AGENT"),
        ("COMEDY_FACTORY_IMAGE_PROVIDER", "COMEDY_FACTORY_GENERATE_IMAGE_AGENT"),
        ("COMEDY_FACTORY_CLOUDFLARE_IMAGE_MODEL", "COMEDY_FACTORY_GENERATE_IMAGE_AGENT"),
        ("COMEDY_FACTORY_GOOGLE_IMAGE__MODEL", "COMEDY_FACTORY_GENERATE_IMAGE_AGENT"),
    ],
)
def test_removed_env_vars_point_at_the_version_override(name, replacement):
    with pytest.raises(ValueError, match=f"{name}.*{replacement}"):
        check_env({name: "x"})


def test_settings_and_unprefixed_env_vars_pass():
    check_env(
        {
            "COMEDY_FACTORY_GENERATE_JOKE_AGENT": "claude-opus",
            "COMEDY_FACTORY_MAX_GRADE_ATTEMPTS": "2",
            "GOOGLE_CLOUD_LOCATION": "global",
            "CLOUDFLARE_API_TOKEN": "token",
            "PATH": "/bin",
        }
    )


def test_aliased_settings_have_no_prefixed_name():
    with pytest.raises(ValueError, match="COMEDY_FACTORY_GOOGLE_CLOUD_LOCATION"):
        check_env({"COMEDY_FACTORY_GOOGLE_CLOUD_LOCATION": "global"})


def test_env_var_names_are_case_insensitive(monkeypatch):
    monkeypatch.setenv("comedy_factory_generate_joke_agent", "claude-opus")
    assert Settings().generate_joke_agent == "claude-opus"
    with pytest.raises(ValueError, match="(?i)comedy_factory_generate_joke_agnt"):
        check_env({"comedy_factory_generate_joke_agnt": "x"})


def test_loading_settings_checks_the_environment(monkeypatch):
    monkeypatch.setenv("COMEDY_FACTORY_IMAGE_PROVIDER", "cloudflare")
    with pytest.raises(ValueError, match="COMEDY_FACTORY_GENERATE_IMAGE_AGENT"):
        Settings()


def test_env_errors_never_echo_secret_values(monkeypatch):
    token = "cf-secret-0123456789abcdefghijklmnopqrstuvwxyz"
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", token)
    monkeypatch.setenv("COMEDY_FACTORY_GENERATE_IMAGE_AGNT", "flux-schnell")
    with pytest.raises(ValueError, match="COMEDY_FACTORY_GENERATE_IMAGE_AGNT") as excinfo:
        Settings()
    message = str(excinfo.value)
    assert token[:12] not in message
    assert token[-12:] not in message
