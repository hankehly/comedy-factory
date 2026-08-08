"""Shared settings for the comedy factory workflows."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="COMEDY_FACTORY_")

    model: str = "google:gemini-3.6-flash"
    prompts_dir: Path = Path(__file__).parent.parent / "prompts"


settings = Settings()
