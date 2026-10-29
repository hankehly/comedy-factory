"""Shared settings for the comedy factory workflows."""

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, with_config
from pydantic_ai.settings import ModelSettings
from pydantic_settings import BaseSettings, SettingsConfigDict


# Extra keys are allowed so provider-specific settings (`google_*`,
# `anthropic_*`, ...) pass through instead of being silently dropped; the
# common keys are still validated. httpx.Timeout (the `timeout` key's type)
# has no pydantic schema, hence arbitrary types.
@with_config(ConfigDict(extra="allow", arbitrary_types_allowed=True))
class StepModelSettings(ModelSettings, total=False):
    """pydantic-ai model settings (thinking, temperature, max_tokens, ...)."""


class StepConfig(BaseModel):
    """The model and model settings one pipeline step's LLM calls use."""

    model: str = "google-cloud:gemini-3.5-flash"
    model_settings: StepModelSettings = Field(default_factory=StepModelSettings)


class Settings(BaseSettings):
    # Step configs are overridable per key from the environment, e.g.
    # COMEDY_FACTORY_GENERATE_JOKE__MODEL_SETTINGS__THINKING=low; keys not
    # given keep the step's defaults below.
    model_config = SettingsConfigDict(
        env_prefix="COMEDY_FACTORY_",
        env_nested_delimiter="__",
        nested_model_default_partial_update=True,
    )

    # Thinking effort per step. Gemini 3 models think by default, and
    # `thinking=False` turns them down to minimal rather than fully off.
    find_topic: StepConfig = StepConfig()
    generate_subtext: StepConfig = StepConfig(model_settings={"thinking": "medium"})
    grade_subtext: StepConfig = StepConfig(model_settings={"thinking": "medium"})
    generate_joke: StepConfig = StepConfig(model_settings={"thinking": "high"})
    grade_joke: StepConfig = StepConfig(model_settings={"thinking": "medium"})
    rephrase_joke: StepConfig = StepConfig(model_settings={"thinking": "low"})
    write_image_prompt: StepConfig = StepConfig(model_settings={"thinking": "low"})
    describe_image: StepConfig = StepConfig(model_settings={"thinking": False})
    prompts_dir: Path = Path(__file__).parent.parent / "prompts"
    output_dir: Path = Path(__file__).parent.parent / "output"
    max_grade_attempts: int = 3
    max_recent_topics: int = 20
    """How many recent bundle topics to feed the topic scan as stories to avoid."""
    rephrasing_count: int = 3
    """How many alternative phrasings of the approved joke to generate."""

    # Vertex AI region for the google-cloud: models. Newer Gemini models are
    # served from the "global" endpoint, not the provider's us-central1
    # default. The env var is unprefixed by Google convention.
    google_cloud_location: str = Field("global", validation_alias="GOOGLE_CLOUD_LOCATION")

    # Image generation. `image_provider` picks the backend; each provider keeps
    # its own model setting so switching back and forth is a one-variable flip.
    image_provider: Literal["google", "cloudflare"] = "google"
    google_image: StepConfig = StepConfig(model="google-cloud:gemini-3.1-flash-image")
    cloudflare_image_model: str = "@cf/black-forest-labs/flux-1-schnell"

    # Cloudflare Workers AI credentials. The env vars are unprefixed by
    # Cloudflare convention.
    cloudflare_account_id: str = Field("", validation_alias="CLOUDFLARE_ACCOUNT_ID")
    cloudflare_api_token: str = Field("", validation_alias="CLOUDFLARE_API_TOKEN")


settings = Settings()

# pydantic-ai's google-cloud provider reads GOOGLE_CLOUD_LOCATION from the
# environment when a model is constructed, so make the setting authoritative
# whenever the variable isn't already set.
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", settings.google_cloud_location)
