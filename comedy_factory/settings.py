"""Shared settings for the comedy factory workflows."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="COMEDY_FACTORY_")

    scan_news_model: str = "google-cloud:gemini-3.5-flash"
    generate_subtext_model: str = "google-cloud:gemini-3.5-flash"
    grade_subtext_model: str = "google-cloud:gemini-3.5-flash"
    generate_joke_model: str = "google-cloud:gemini-3.5-flash"
    grade_joke_model: str = "google-cloud:gemini-3.5-flash"
    write_image_prompt_model: str = "google-cloud:gemini-3.5-flash"
    prompts_dir: Path = Path(__file__).parent.parent / "prompts"
    output_dir: Path = Path(__file__).parent.parent / "output"
    max_grade_attempts: int = 3

    # Image generation. `image_provider` picks the backend; each provider keeps
    # its own model setting so switching back and forth is a one-variable flip.
    image_provider: Literal["google", "cloudflare"] = "google"
    google_image_model: str = "google-cloud:gemini-3.1-flash-image"
    cloudflare_image_model: str = "@cf/black-forest-labs/flux-1-schnell"

    # Cloudflare Workers AI credentials. The env vars are unprefixed by
    # Cloudflare convention.
    cloudflare_account_id: str = Field("", validation_alias="CLOUDFLARE_ACCOUNT_ID")
    cloudflare_api_token: str = Field("", validation_alias="CLOUDFLARE_API_TOKEN")


settings = Settings()
