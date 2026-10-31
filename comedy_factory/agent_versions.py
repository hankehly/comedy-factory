"""Per-step agent versions: the committed configurations each pipeline step
can run. A version is one pydantic-ai agent spec file holding a model and its
model settings, at agents/<step>/<version>.yaml (see agents/README.md)."""

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import ConfigDict, TypeAdapter, ValidationError, with_config
from pydantic_ai import AgentSpec
from pydantic_ai.models import Model, infer_model
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.settings import ModelSettings

from comedy_factory.settings import settings

STEPS: tuple[str, ...] = (
    "find_topic",
    "generate_subtext",
    "grade_subtext",
    "generate_joke",
    "grade_joke",
    "rephrase_joke",
    "write_image_prompt",
    "generate_image",
    "describe_image",
)
"""The pipeline's model-backed steps, in pipeline order."""

CLOUDFLARE_PREFIX = "cloudflare:"
"""Prefix of Cloudflare Workers AI image models, which pydantic-ai can't
build; only generate_image versions may use one."""

# A version's spec is applied per run, and run-time spec values add to the
# agent's own, so instructions or capabilities in a version file would quietly
# change the step's prompt or tools. Versions hold only these keys.
_VERSION_KEYS = {"model", "model_settings"}

# pydantic-ai's settings TypedDicts, made to reject keys they don't define:
# AgentSpec leaves model_settings unchecked and pydantic-ai ignores unknown
# keys, so a misspelled setting would otherwise silently not apply. (The
# `timeout` setting's type, httpx.Timeout, has no pydantic schema.)
_STRICT_SETTINGS = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


@with_config(_STRICT_SETTINGS)
class _ModelSettings(ModelSettings, total=False):
    """The settings every provider accepts."""


@with_config(_STRICT_SETTINGS)
class _GoogleModelSettings(GoogleModelSettings, total=False):
    """The common settings plus the `google_*` ones."""


@with_config(_STRICT_SETTINGS)
class _AnthropicModelSettings(AnthropicModelSettings, total=False):
    """The common settings plus the `anthropic_*` ones."""


# The settings class for each model class; other models get the common ones.
_PROVIDER_SETTINGS: dict[type[Model], Any] = {
    GoogleModel: _GoogleModelSettings,
    AnthropicModel: _AnthropicModelSettings,
}


class AgentVersionError(ValueError):
    """An unknown or broken agent version."""


@dataclass(frozen=True)
class AgentVersion:
    """One committed configuration of a pipeline step's agent."""

    step: str
    name: str
    spec: AgentSpec

    @property
    def model(self) -> str:
        # Every checked version file sets its model.
        return cast(str, self.spec.model)

    @property
    def model_settings(self) -> dict[str, Any]:
        return self.spec.model_settings or {}

    def as_metadata(self) -> dict[str, Any]:
        """The version as an asset bundle's metadata records it."""
        return {"version": self.name, "model": self.model, "model_settings": self.model_settings}


def available_versions(step: str) -> list[str]:
    """Names of the step's committed versions, sorted."""
    return sorted(path.stem for path in (settings.agents_dir / step).glob("*.yaml"))


def load_version(step: str, name: str) -> AgentVersion:
    """Load and check one of the step's committed versions, by name."""
    available = available_versions(step)
    if name not in available:
        raise AgentVersionError(
            f"Unknown {step} agent version {name!r}; available: {', '.join(available) or 'none'}"
        )
    path = settings.agents_dir / step / f"{name}.yaml"
    return AgentVersion(step, name, check_version_file(step, path))


def selected_versions() -> dict[str, AgentVersion]:
    """The version each step is set to run (`<step>_agent` settings), loaded
    and checked, so a bad selection fails before any model is called."""
    return {step: load_version(step, getattr(settings, f"{step}_agent")) for step in STEPS}


def check_version_file(step: str, path: Path) -> AgentSpec:
    """Check a version file of `step` and return its spec. The first problem
    found raises AgentVersionError naming the step, version and file."""

    def error(problem: str) -> AgentVersionError:
        return AgentVersionError(f"{step} agent version {path.stem!r} ({path}): {problem}")

    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise error(f"not valid YAML: {e}") from e
    if not isinstance(data, dict):
        raise error("must be a mapping with model and optional model_settings")
    if unexpected := data.keys() - _VERSION_KEYS:
        raise error(f"unexpected keys: {', '.join(sorted(map(str, unexpected)))}")
    if not data.get("model"):
        raise error("'model' is required")
    try:
        spec = AgentSpec.model_validate(data)
    except ValidationError as e:
        raise error(_describe(e)) from e

    model_name = cast(str, spec.model)
    if model_name.startswith(CLOUDFLARE_PREFIX):
        if step != "generate_image":
            raise error("cloudflare: models are only for generate_image")
        if spec.model_settings:
            raise error("cloudflare: models take no model_settings")
        return spec
    try:
        model = infer_model(model_name)
    except Exception as e:
        raise error(f"model {model_name!r} can't be built: {e}") from e
    try:
        _check_settings(model, spec.model_settings or {})
    except ValidationError as e:
        raise error(_describe(e, "model_settings")) from e
    return spec


def _check_settings(model: Model, model_settings: dict[str, Any]) -> None:
    """Validate model_settings against pydantic-ai's settings class for the
    model's provider; raises ValidationError.

    Validated as JSON in strict mode: YAML values are JSON values, and strict
    JSON validation rejects coercions (a quoted number) while still accepting
    enum values written as strings. The file's own values are what runs."""
    settings_class = next(
        (cls for model_class, cls in _PROVIDER_SETTINGS.items() if isinstance(model, model_class)),
        _ModelSettings,
    )
    _settings_adapter(settings_class).validate_json(
        json.dumps(model_settings, default=str), strict=True
    )


@cache
def _settings_adapter(settings_class: Any) -> TypeAdapter[Any]:
    return TypeAdapter(settings_class)


def _describe(error: ValidationError, *prefix: str) -> str:
    """pydantic's errors as `field: message` pairs, leaving out the input
    values pydantic would otherwise echo."""
    return "; ".join(
        f"{'.'.join(map(str, (*prefix, *detail['loc'])))}: {detail['msg']}"
        for detail in error.errors()
    )
