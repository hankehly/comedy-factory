"""Per-step agent versions: the committed configurations each pipeline step
can run. A version is one pydantic-ai agent spec file holding a model and its
model settings, at agents/<step>/<version>.yaml (see agents/README.md)."""

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, cast

import jsonschema
import yaml
from jsonschema.protocols import Validator
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

SCHEMA_FILENAME = "agent-spec.schema.json"
"""The committed copy of schema() in the agents directory, which version
files point the editor at; regenerate it with `just agent-schema`."""

# A version's spec is applied per run, and run-time spec values add to the
# agent's own, so instructions or capabilities in a version file would quietly
# change the step's prompt or tools. Versions hold only these keys.
_VERSION_KEYS = {"model", "model_settings"}

# pydantic-ai's settings TypedDict for each model class; each includes the
# common ModelSettings keys, and other model classes get only those.
_PROVIDER_SETTINGS: dict[type[Model], Any] = {
    GoogleModel: GoogleModelSettings,
    AnthropicModel: AnthropicModelSettings,
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


@cache
def schema() -> dict[str, Any]:
    """pydantic-ai's JSON schema for agent spec files."""
    return AgentSpec.model_json_schema_with_capabilities()


@cache
def _schema_validator() -> Validator:
    # Built once: jsonschema.validate() re-checks the large schema every call.
    return jsonschema.validators.validator_for(schema())(schema())


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
        _schema_validator().validate(data)
    except jsonschema.ValidationError as e:
        raise error(e.message) from e

    model_name: str = data["model"]
    model_settings: dict[str, Any] = data.get("model_settings") or {}
    if model_name.startswith(CLOUDFLARE_PREFIX):
        if step != "generate_image":
            raise error("cloudflare: models are only for generate_image")
        if model_settings:
            raise error("cloudflare: models take no model_settings")
    else:
        try:
            model = infer_model(model_name)
        except Exception as e:
            raise error(f"model {model_name!r} can't be built: {e}") from e
        if unknown := model_settings.keys() - _settings_keys(model):
            raise error(f"unknown model_settings keys: {', '.join(sorted(unknown))}")
    return AgentSpec.model_validate(data)


def _settings_keys(model: Model) -> frozenset[str]:
    """The model_settings keys pydantic-ai defines for the model's provider."""
    settings_class = next(
        (cls for model_class, cls in _PROVIDER_SETTINGS.items() if isinstance(model, model_class)),
        ModelSettings,
    )
    return settings_class.__optional_keys__ | settings_class.__required_keys__


if __name__ == "__main__":
    # `just agent-schema`: refresh the committed schema, e.g. after upgrading
    # pydantic-ai changes it (a test fails until this is re-run).
    (settings.agents_dir / SCHEMA_FILENAME).write_text(json.dumps(schema(), indent=2) + "\n")
