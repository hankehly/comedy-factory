"""Shared settings for the comedy factory workflows."""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_PREFIX = "COMEDY_FACTORY_"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix=_ENV_PREFIX)

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

    def __init__(self, **values: Any) -> None:
        # Checked before validation, not in a validator or model_post_init:
        # pydantic would wrap the error with a repr of the settings input,
        # which ends with the Cloudflare API token.
        check_env(os.environ)
        super().__init__(**values)


def check_env(environ: Mapping[str, str]) -> None:
    """Fail on COMEDY_FACTORY_* environment variables that match no setting.

    pydantic-settings ignores them, so a misspelled override, or a variable
    from before agent versions, would otherwise quietly run on the defaults.
    Names compare case-insensitively, as pydantic-settings matches them.
    """
    valid = {
        f"{_ENV_PREFIX}{name}".upper()
        for name, field in Settings.model_fields.items()
        if field.validation_alias is None
    }
    unknown = sorted(
        name
        for name in environ
        if name.upper().startswith(_ENV_PREFIX) and name.upper() not in valid
    )
    if unknown:
        raise ValueError(
            f"Unknown {_ENV_PREFIX}* environment variables (no such setting):\n"
            + "\n".join(f"  {name}{_replacement_hint(name)}" for name in unknown)
        )


def _replacement_hint(name: str) -> str:
    """For a variable that set a step's model before agent versions, the
    version override that replaces it."""
    rest = name.upper().removeprefix(_ENV_PREFIX)
    if rest in {"IMAGE_PROVIDER", "CLOUDFLARE_IMAGE_MODEL", "GOOGLE_IMAGE_MODEL"} or (
        rest.startswith("GOOGLE_IMAGE__")
    ):
        return f" (removed; pick an image version with {_ENV_PREFIX}GENERATE_IMAGE_AGENT)"
    for field_name in Settings.model_fields:
        step = field_name.removesuffix("_agent").upper()
        if field_name.endswith("_agent") and (
            rest == f"{step}_MODEL" or rest.startswith(f"{step}__")
        ):
            return f" (removed; pick a version with {_ENV_PREFIX}{step}_AGENT)"
    return ""


settings = Settings()

# pydantic-ai's google-cloud provider reads GOOGLE_CLOUD_LOCATION from the
# environment when a model is constructed, so make the setting authoritative
# whenever the variable isn't already set.
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", settings.google_cloud_location)
