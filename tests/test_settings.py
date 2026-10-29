"""Tests for the per-step model configuration in settings."""

import pytest
from pydantic import ValidationError

from comedy_factory.settings import Settings, StepConfig


def test_step_env_override_keeps_the_step_defaults(monkeypatch):
    monkeypatch.setenv("COMEDY_FACTORY_GOOGLE_IMAGE__MODEL_SETTINGS__TEMPERATURE", "0.5")
    monkeypatch.setenv("COMEDY_FACTORY_GENERATE_JOKE__MODEL", "anthropic:claude-sonnet-5-5")

    settings = Settings()

    # Only the overridden key changes; the step's other defaults survive.
    assert settings.google_image.model == "google-cloud:gemini-3.1-flash-image"
    assert settings.google_image.model_settings == {"temperature": 0.5}
    assert settings.generate_joke.model == "anthropic:claude-sonnet-5-5"
    assert settings.generate_joke.model_settings == {"thinking": "high"}


def test_step_model_settings_keep_provider_specific_keys():
    step = StepConfig.model_validate(
        {"model_settings": {"google_thinking_config": {"thinking_budget": 1024}}}
    )
    assert step.model_settings == {"google_thinking_config": {"thinking_budget": 1024}}


def test_step_model_settings_reject_invalid_common_keys():
    with pytest.raises(ValidationError):
        StepConfig.model_validate({"model_settings": {"thinking": "hgh"}})
