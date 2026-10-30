"""Shared settings for the comedy factory workflows."""

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="COMEDY_FACTORY_")

    # The agent version each pipeline step runs: a committed file at
    # agents/<step>/<version>.yaml holding the step's model and model settings
    # (see agents/README.md). Override one to test or try a candidate version,
    # e.g. COMEDY_FACTORY_GENERATE_JOKE_AGENT=claude-opus.
    find_topic_agent: str = "gemini-flash"
    generate_subtext_agent: str = "gemini-flash"
    grade_subtext_agent: str = "gemini-flash"
    generate_joke_agent: str = "gemini-flash"
    grade_joke_agent: str = "gemini-flash"
    rephrase_joke_agent: str = "gemini-flash"
    write_image_prompt_agent: str = "gemini-flash"
    generate_image_agent: str = "gemini-flash-image"
    describe_image_agent: str = "gemini-flash"
    prompts_dir: Path = Path(__file__).parent.parent / "prompts"
    agents_dir: Path = Path(__file__).parent.parent / "agents"
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

    # Cloudflare Workers AI credentials, used when the generate_image version
    # is a cloudflare: model (e.g. flux-schnell); that version, not a setting,
    # picks the image backend. The env vars are unprefixed by Cloudflare
    # convention.
    cloudflare_account_id: str = Field("", validation_alias="CLOUDFLARE_ACCOUNT_ID")
    cloudflare_api_token: str = Field("", validation_alias="CLOUDFLARE_API_TOKEN")


settings = Settings()

# pydantic-ai's google-cloud provider reads GOOGLE_CLOUD_LOCATION from the
# environment when a model is constructed, so make the setting authoritative
# whenever the variable isn't already set.
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", settings.google_cloud_location)
