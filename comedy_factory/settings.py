"""Shared settings for the comedy factory workflows."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="COMEDY_FACTORY_")

    model: str = "google:gemini-2.5-flash"
    scan_news_model: str = "google:gemini-2.5-flash"
    prompts_dir: Path = Path(__file__).parent.parent / "prompts"
    max_grade_attempts: int = 3

    # Cloudflare Workers AI, used for image generation. The credential env vars
    # are unprefixed by Cloudflare convention.
    image_model: str = "@cf/black-forest-labs/flux-1-schnell"
    cloudflare_account_id: str = Field("", validation_alias="CLOUDFLARE_ACCOUNT_ID")
    cloudflare_api_token: str = Field("", validation_alias="CLOUDFLARE_API_TOKEN")


settings = Settings()
