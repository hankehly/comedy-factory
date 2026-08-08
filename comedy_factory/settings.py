"""Shared settings for the comedy factory workflows."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="COMEDY_FACTORY_")

    model: str = "google:gemini-3.6-flash"
    scan_news_model: str = "google:gemini-3.1-flash-live-preview"
    prompts_dir: Path = Path(__file__).parent.parent / "prompts"
    max_grade_attempts: int = 3


settings = Settings()
