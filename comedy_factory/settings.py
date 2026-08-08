"""Shared settings for the comedy factory workflows."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="COMEDY_FACTORY_")

    model: str = "google:gemini-3.5-flash-lite"
    scan_news_model: str = "google:gemini-3.5-flash-lite"
    prompts_dir: Path = Path(__file__).parent.parent / "prompts"
    max_grade_attempts: int = 3


settings = Settings()
