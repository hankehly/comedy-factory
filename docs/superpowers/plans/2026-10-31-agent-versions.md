# Agent Versions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-step model settings with committed, per-step agent version files that settings select by name, so that every model and model setting that runs is reviewed code.

**Architecture:**
- Version files live at `agents/<step>/<version>.yaml`. Each one is a pydantic-ai agent spec limited to `model` and `model_settings`.
- A new `comedy_factory/agent_versions.py` loads and checks them, using pydantic-ai's spec schema, pydantic-ai's provider settings classes and `infer_model`.
- Settings name one version per step. `main()` loads every selected version before the first model call and hands each step its version.
- A step runs its agent with `model=infer_model(version.model)` and `spec=version.spec`.

**Tech Stack:** Python 3.12, pydantic-ai 2.27 (`AgentSpec`, run-time `spec=`), pydantic-settings 2.15, PyYAML, jsonschema, pytest.

**Spec:** `docs/superpowers/specs/2026-10-30-agent-versions-design.md`

## Global Constraints

- Python 3.12; `pydantic-ai>=2.27.0`; `pydantic-settings>=2.15.0`.
- Gates unchanged and all must pass: `just test` (100% branch coverage), `just lint`, `just mypy`, `just checkov`. CI mirrors them.
- Tests never call real models (`models.ALLOW_MODEL_REQUESTS = False` in `tests/conftest.py`). CI has no model credentials.
- Version files contain only `model` and `model_settings`.
- Each step's version is chosen by the setting `<step>_agent`, overridable as `COMEDY_FACTORY_<STEP>_AGENT`.
- Defaults are `gemini-flash` for every step except `generate_image_agent = "gemini-flash-image"`.
- An override takes a version name only, never a path.
- Unknown-version error text: `Unknown {step} agent version {name!r}; available: {names, comma-separated, or "none"}`. For example: `Unknown generate_joke agent version 'claud-opus'; available: gemini-flash`.
- The Cloudflare credentials (`CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`) stay unprefixed env vars.
- No model string or model setting can be set outside a version file.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Stray files in `agents/`**, such as macOS `.DS_Store` or a `README.md`: these must never count as versions or step folders. Tested in Task 1 (`test_available_versions_lists_only_yaml_files`) and Task 2 (`test_agents_folder_has_one_folder_per_step` counts directories only).
2. **Path-like or suffixed version names**, such as `../grade_joke/gemini-flash` or `gemini-flash.yaml`: these are rejected as unknown versions and never resolved. Tested in Task 1 (`test_load_version_rejects_path_like_names`).
3. **Malformed version files** (a YAML syntax error, an empty file, a top-level list): these raise `AgentVersionError` naming the step, version and file, not a raw traceback. Tested in Task 1 (`test_check_rejects_broken_version_files`).
4. **Lowercase env var names:** pydantic-settings matches names case-insensitively, so the env check must accept `comedy_factory_generate_joke_agent` and reject unknown lowercase names. Tested in Task 4 (`test_env_var_names_are_case_insensitive`).
5. **A missing step folder:** loading reports `available: none` instead of crashing with `FileNotFoundError`. Tested in Task 1 (`test_missing_step_folder_has_no_versions`).

## File Structure

- Create `comedy_factory/agent_versions.py`: the step list, plus loading and checking of version files.
- Create `agents/`: 10 version files, `agent-spec.schema.json` and `README.md`.
- Create `tests/test_agent_versions.py`.
- Modify `comedy_factory/settings.py`:
  - add `agents_dir`, the `<step>_agent` fields and the env check
  - remove `StepConfig`, `StepModelSettings`, the step-config fields, `google_image`, `image_provider` and `cloudflare_image_model`
- Modify `comedy_factory/joke_generator.py`: steps take versions, image generation dispatches on the version's model, and `metadata.json` records the versions.
- Modify `tests/test_joke_generator.py`, `tests/test_settings.py` and `tests/conftest.py`.
- Modify `pyproject.toml`, `uv.lock`, `justfile` and `.env.template`.

---

### Task 1: Version loader and checks

**Files:**
- Create: `comedy_factory/agent_versions.py`
- Modify: `comedy_factory/settings.py` (add `agents_dir` only)
- Modify: `tests/conftest.py`, `pyproject.toml`, `uv.lock`
- Test: `tests/test_agent_versions.py` (create)

**Interfaces:**
- Consumes: `settings.agents_dir: Path`, added in this task.
- Produces, in `comedy_factory/agent_versions.py`:
  - `STEPS: tuple[str, ...] = ("find_topic", "generate_subtext", "grade_subtext", "generate_joke", "grade_joke", "rephrase_joke", "write_image_prompt", "generate_image", "describe_image")`
  - `CLOUDFLARE_PREFIX = "cloudflare:"`
  - `class AgentVersionError(ValueError)`
  - `@dataclass(frozen=True) class AgentVersion`, with fields `step: str`, `name: str` and `spec: pydantic_ai.AgentSpec`, plus:
    - property `model -> str`, which is `spec.model` (never `None` for a checked file)
    - property `model_settings -> dict[str, Any]`, which is `spec.model_settings` or `{}`
    - method `as_metadata() -> dict[str, Any]`, which returns `{"version": name, "model": model, "model_settings": model_settings}`
  - `schema() -> dict[str, Any]`: `AgentSpec.model_json_schema_with_capabilities()`, cached
  - `available_versions(step: str) -> list[str]`
  - `check_version_file(step: str, path: Path) -> AgentSpec`
  - `load_version(step: str, name: str) -> AgentVersion`

- [ ] **Step 1: Add the dependencies**

Run: `uv add jsonschema "pydantic-ai-slim[spec]>=2.27.0" && uv add --dev types-jsonschema types-pyyaml`
Expected: `pyproject.toml` lists `jsonschema` and `pydantic-ai-slim[spec]` under `dependencies`, and the two stub packages under the `dev` group. `uv.lock` is updated, and `uv run python -c "import yaml, jsonschema"` exits 0.

- [ ] **Step 2: Add `agents_dir` and a dummy Anthropic key**

In `comedy_factory/settings.py`, next to `prompts_dir`, add `agents_dir: Path = Path(__file__).parent.parent / "agents"`.

In `tests/conftest.py`, beside the `GOOGLE_API_KEY` default, add `os.environ.setdefault("ANTHROPIC_API_KEY", "test-api-key")`. Give it a comment saying that version checks build models with `infer_model`, which demands provider keys.

- [ ] **Step 3: Write the failing tests** in `tests/test_agent_versions.py`

```python
import re

import pytest

from comedy_factory.agent_versions import (
    AgentVersionError, available_versions, check_version_file, load_version,
)
from comedy_factory.settings import settings

GEMINI = "google-cloud:gemini-3.5-flash"
FLUX = "cloudflare:@cf/black-forest-labs/flux-1-schnell"


@pytest.fixture
def agents_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "agents_dir", tmp_path / "agents")
    return tmp_path / "agents"


def write_version(agents_dir, step, name, text):
    path = agents_dir / step / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_load_version_reads_model_and_settings(agents_dir):
    write_version(agents_dir, "generate_joke", "gemini-flash",
                  f"model: {GEMINI}\nmodel_settings:\n  thinking: high\n")
    version = load_version("generate_joke", "gemini-flash")
    assert (version.step, version.name, version.model) == ("generate_joke", "gemini-flash", GEMINI)
    assert version.as_metadata() == {
        "version": "gemini-flash", "model": GEMINI, "model_settings": {"thinking": "high"},
    }


def test_version_without_settings_has_empty_settings(agents_dir):
    write_version(agents_dir, "find_topic", "gemini-flash", f"model: {GEMINI}\n")
    assert load_version("find_topic", "gemini-flash").model_settings == {}


def test_load_version_rejects_unknown_names_listing_available(agents_dir):
    write_version(agents_dir, "generate_joke", "gemini-flash", f"model: {GEMINI}\n")
    message = "Unknown generate_joke agent version 'claud-opus'; available: gemini-flash"
    with pytest.raises(AgentVersionError, match=re.escape(message)):
        load_version("generate_joke", "claud-opus")


@pytest.mark.parametrize("name", ["../grade_joke/gemini-flash", "gemini-flash.yaml", ""])
def test_load_version_rejects_path_like_names(agents_dir, name):
    write_version(agents_dir, "grade_joke", "gemini-flash", f"model: {GEMINI}\n")
    write_version(agents_dir, "generate_joke", "gemini-flash", f"model: {GEMINI}\n")
    with pytest.raises(AgentVersionError, match="Unknown generate_joke agent version"):
        load_version("generate_joke", name)


def test_missing_step_folder_has_no_versions(agents_dir):
    assert available_versions("describe_image") == []
    with pytest.raises(AgentVersionError, match="available: none"):
        load_version("describe_image", "gemini-flash")


def test_available_versions_lists_only_yaml_files(agents_dir):
    write_version(agents_dir, "generate_joke", "b-version", f"model: {GEMINI}\n")
    write_version(agents_dir, "generate_joke", "a-version", f"model: {GEMINI}\n")
    (agents_dir / "generate_joke" / ".DS_Store").write_bytes(b"\0")
    (agents_dir / "generate_joke" / "README.md").write_text("notes")
    assert available_versions("generate_joke") == ["a-version", "b-version"]


@pytest.mark.parametrize(("text", "problem"), [
    ("model: [", "not valid YAML"),
    ("", "must be a mapping"),
    ("- a\n- b\n", "must be a mapping"),
    ("model_settings:\n  thinking: high\n", "'model' is required"),
    (f"model: {GEMINI}\ninstructions: Be funny.\n", "unexpected keys: instructions"),
    (f"model: {GEMINI}\nmodel_settings:\n  thinking: hgh\n", "'hgh'"),
    (f"model: {GEMINI}\nmodel_settings:\n  thinkng: high\n", "unknown model_settings keys: thinkng"),
    (f"model: {GEMINI}\nmodel_settings:\n  anthropic_betas: []\n",
     "unknown model_settings keys: anthropic_betas"),
    ("model: nope:some-model\n", "can't be built"),
])
def test_check_rejects_broken_version_files(agents_dir, text, problem):
    path = write_version(agents_dir, "generate_joke", "broken", text)
    with pytest.raises(AgentVersionError, match=re.escape(problem)) as excinfo:
        check_version_file("generate_joke", path)
    assert "generate_joke agent version 'broken'" in str(excinfo.value)


def test_check_allows_provider_specific_settings(agents_dir):
    gemini = write_version(agents_dir, "generate_joke", "gemini",
        f"model: {GEMINI}\nmodel_settings:\n  google_thinking_config:\n    include_thoughts: true\n")
    claude = write_version(agents_dir, "generate_joke", "claude",
        "model: anthropic:claude-opus-5-5\nmodel_settings:\n  anthropic_betas: []\n")
    assert check_version_file("generate_joke", gemini).model == GEMINI
    assert check_version_file("generate_joke", claude).model == "anthropic:claude-opus-5-5"


def test_check_reports_a_missing_anthropic_key(agents_dir, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    path = write_version(agents_dir, "generate_joke", "claude", "model: anthropic:claude-opus-5-5\n")
    with pytest.raises(AgentVersionError, match="ANTHROPIC_API_KEY"):
        check_version_file("generate_joke", path)


def test_cloudflare_models_are_allowed_for_generate_image(agents_dir):
    path = write_version(agents_dir, "generate_image", "flux-schnell", f"model: {FLUX}\n")
    assert check_version_file("generate_image", path).model == FLUX


def test_cloudflare_models_are_only_for_generate_image(agents_dir):
    path = write_version(agents_dir, "describe_image", "flux", f"model: {FLUX}\n")
    with pytest.raises(AgentVersionError, match="cloudflare: models are only for generate_image"):
        check_version_file("describe_image", path)


def test_cloudflare_models_take_no_settings(agents_dir):
    path = write_version(agents_dir, "generate_image", "flux",
                         f"model: {FLUX}\nmodel_settings:\n  temperature: 0.5\n")
    with pytest.raises(AgentVersionError, match="cloudflare: models take no model_settings"):
        check_version_file("generate_image", path)
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `uv run pytest tests/test_agent_versions.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'comedy_factory.agent_versions'`.

- [ ] **Step 5: Implement `comedy_factory/agent_versions.py`**

Implement the names in the Interfaces block. The module reads `settings.agents_dir` at call time, never at import, so tests can repoint it.

- `available_versions(step)`: the sorted stems of `*.yaml` files in `settings.agents_dir / step`, or `[]` if the folder is missing.
- `load_version(step, name)`:
  1. If `name` isn't in `available_versions(step)`, raise the unknown-version error (Global Constraints). This also rejects path-like names.
  2. Otherwise, return `AgentVersion(step, name, check_version_file(step, <that file>))`.
- `check_version_file(step, path)`: every error is an `AgentVersionError` whose message starts with `f"{step} agent version {path.stem!r} ({path}): "` and ends with the problem. Run the checks in this order:
  1. `yaml.safe_load`. On `yaml.YAMLError`, the problem is `not valid YAML: {error}`.
  2. If the result isn't a dict, the problem is `must be a mapping with model and optional model_settings`.
  3. If there are keys other than `model` and `model_settings`, the problem is `unexpected keys: {sorted, ", "-joined}`.
  4. If there's no `model`, the problem is `'model' is required`.
  5. Validate against `schema()`. On `jsonschema.ValidationError`, the problem is `error.message`. Build the validator once and cache it (for example `validator_for(schema())(schema())`), because `jsonschema.validate` re-checks the large schema on every call.
  6. If `model` starts with `CLOUDFLARE_PREFIX`:
     - when `step != "generate_image"`, the problem is `cloudflare: models are only for generate_image`
     - when `model_settings` is non-empty, the problem is `cloudflare: models take no model_settings`
     - otherwise skip to step 9
  7. `infer_model(model)`. On any exception, the problem is `model {model!r} can't be built: {error}`.
  8. The allowed `model_settings` keys are the `__optional_keys__ | __required_keys__` of pydantic-ai's settings TypedDict for the built model's class:
     - `GoogleModel` uses `GoogleModelSettings`
     - `AnthropicModel` uses `AnthropicModelSettings`
     - any other class falls back to `ModelSettings`

     Any other key gives the problem `unknown model_settings keys: {sorted, ", "-joined}`.
  9. Return `AgentSpec.model_validate(data)`.

- [ ] **Step 6: Run the tests to verify they pass, then the gates**

Run: `uv run pytest tests/test_agent_versions.py -q && just test && just lint && just mypy`
Expected: all pass, with coverage at 100% (the new module is fully covered by this task's tests).

- [ ] **Step 7: Commit**

```bash
git add comedy_factory/agent_versions.py comedy_factory/settings.py tests/conftest.py tests/test_agent_versions.py pyproject.toml uv.lock
git commit -m "Add loader and checks for per-step agent version files" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Commit the initial versions and the spec schema

**Files:**
- Create: `agents/<step>/<version>.yaml` (10 files, see the table), `agents/agent-spec.schema.json`
- Modify: `comedy_factory/agent_versions.py` (add `SCHEMA_FILENAME` and a `__main__` block), `justfile`
- Test: `tests/test_agent_versions.py`

**Interfaces:**
- Consumes: from Task 1, `STEPS`, `schema()`, `check_version_file()` and `load_version()`.
- Produces: `SCHEMA_FILENAME = "agent-spec.schema.json"` in `agent_versions.py`; the committed files; the `just agent-schema` recipe.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_agent_versions.py`; import `json`, `STEPS`, `SCHEMA_FILENAME` and `schema`)

```python
COMMITTED = settings.agents_dir  # the repo's agents/; tests repoint it only via the fixture

INITIAL_VERSIONS = [
    ("find_topic", "gemini-flash", GEMINI, {}),
    ("generate_subtext", "gemini-flash", GEMINI, {"thinking": "medium"}),
    ("grade_subtext", "gemini-flash", GEMINI, {"thinking": "medium"}),
    ("generate_joke", "gemini-flash", GEMINI, {"thinking": "high"}),
    ("grade_joke", "gemini-flash", GEMINI, {"thinking": "medium"}),
    ("rephrase_joke", "gemini-flash", GEMINI, {"thinking": "low"}),
    ("write_image_prompt", "gemini-flash", GEMINI, {"thinking": "low"}),
    ("generate_image", "gemini-flash-image", "google-cloud:gemini-3.1-flash-image", {}),
    ("generate_image", "flux-schnell", FLUX, {}),
    ("describe_image", "gemini-flash", GEMINI, {"thinking": False}),
]


@pytest.mark.parametrize(("step", "name", "model", "model_settings"), INITIAL_VERSIONS)
def test_initial_versions_carry_over_todays_settings(step, name, model, model_settings):
    version = load_version(step, name)
    assert (version.model, version.model_settings) == (model, model_settings)


@pytest.mark.parametrize("path", sorted(COMMITTED.glob("*/*.yaml")),
                         ids=lambda path: f"{path.parent.name}/{path.stem}")
def test_committed_version_files_pass_the_checks(path):
    check_version_file(path.parent.name, path)
    first_line = path.read_text().splitlines()[0]
    assert first_line == "# yaml-language-server: $schema=../agent-spec.schema.json"


def test_agents_folder_has_one_folder_per_step():
    assert {path.name for path in COMMITTED.iterdir() if path.is_dir()} == set(STEPS)


def test_committed_schema_is_current():
    committed = json.loads((COMMITTED / SCHEMA_FILENAME).read_text())
    assert committed == schema(), "agents/agent-spec.schema.json is stale; run `just agent-schema`"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_agent_versions.py -q`
Expected: FAIL. `SCHEMA_FILENAME` can't be imported, and once it exists the version files and the schema file are missing.

- [ ] **Step 3: Create the 10 version files**

Each file starts with `# yaml-language-server: $schema=../agent-spec.schema.json`, followed by `model:` and, when the table has settings, a `model_settings:` block. The model and settings come from `INITIAL_VERSIONS` above. Leave `model_settings` out entirely when it's `{}`. In YAML, `describe_image` is `thinking: false`.

- [ ] **Step 4: Add the schema generator**

In `agent_versions.py`, define `SCHEMA_FILENAME`. Add an `if __name__ == "__main__":` block that writes `json.dumps(schema(), indent=2) + "\n"` to `settings.agents_dir / SCHEMA_FILENAME`. Coverage already excludes `__main__` guards.

Add this `justfile` recipe:

```
# Regenerate agents/agent-spec.schema.json from the installed pydantic-ai (after upgrading it)
agent-schema:
    uv run python -m comedy_factory.agent_versions
```

Run: `just agent-schema`
Expected: `agents/agent-spec.schema.json` exists.

- [ ] **Step 5: Run the tests to verify they pass, then the gates**

Run: `uv run pytest tests/test_agent_versions.py -q && just test && just lint && just mypy`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add agents comedy_factory/agent_versions.py justfile tests/test_agent_versions.py
git commit -m "Commit initial agent versions and the agent spec schema" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Run the pipeline on agent versions

**Files:**
- Modify: `comedy_factory/settings.py`, `comedy_factory/agent_versions.py` (add `selected_versions`), `comedy_factory/joke_generator.py`, `.env.template`
- Create: `agents/README.md`
- Test: `tests/test_settings.py` (replace the contents), `tests/test_joke_generator.py`

**Interfaces:**
- Consumes: from Task 1, `AgentVersion`, `AgentVersionError`, `STEPS`, `CLOUDFLARE_PREFIX`, `load_version()` and `available_versions()`; from Task 2, the committed versions.
- Produces:
  - Settings fields `find_topic_agent`, `generate_subtext_agent`, `grade_subtext_agent`, `generate_joke_agent`, `grade_joke_agent`, `rephrase_joke_agent`, `write_image_prompt_agent`, `generate_image_agent` and `describe_image_agent`, all `str`, with defaults as in Global Constraints and declared in this pipeline order.
  - `selected_versions() -> dict[str, AgentVersion]`: `load_version(step, getattr(settings, f"{step}_agent"))` for each step in `STEPS`.
  - Every step function takes a keyword-only `version: AgentVersion`:
    - `find_topic(*, version)`
    - `generate_subtext(topic, history=None, feedback=None, *, version)`
    - `grade_subtext(topic, subtext, *, version)`
    - `generate_joke(subtext, history=None, feedback=None, *, version)`
    - `grade_joke(subtext, joke, *, version)`
    - `rephrase_joke(joke, *, version)`
    - `write_image_prompt(joke, *, version)`
    - `generate_image(image_prompt, *, version)`
    - `describe_image(image, *, version)`
  - `save_asset(..., evaluation, versions: dict[str, AgentVersion]) -> Path`, with `versions` as the new last parameter.

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_settings.py` with these tests (`StepConfig` is going away):

```python
from comedy_factory.agent_versions import STEPS, available_versions
from comedy_factory.settings import Settings, settings


def test_every_step_has_a_version_setting_with_a_committed_default():
    steps = {name.removesuffix("_agent") for name in Settings.model_fields if name.endswith("_agent")}
    assert steps == set(STEPS)
    for step in STEPS:
        assert getattr(settings, f"{step}_agent") in available_versions(step)


def test_default_versions():
    defaults = {step: getattr(Settings(), f"{step}_agent") for step in STEPS}
    assert defaults == {**dict.fromkeys(STEPS, "gemini-flash"), "generate_image": "gemini-flash-image"}


def test_env_var_selects_a_steps_version(monkeypatch):
    monkeypatch.setenv("COMEDY_FACTORY_GENERATE_JOKE_AGENT", "claude-opus")
    assert Settings().generate_joke_agent == "claude-opus"


def test_no_setting_holds_a_model():
    assert [name for name in Settings.model_fields if "model" in name] == []
```

In `tests/test_joke_generator.py`:
- Update the module docstring's second bullet: steps get a committed version (see `_version`), and stub models go in through `agent.override(model=...)`, which takes precedence over the version's model while the version's settings still apply.
- Add helpers:
  - `_version(step: str, name: str | None = None) -> AgentVersion`: `load_version(step, name or getattr(settings, f"{step}_agent"))`
  - `_override_step_models(model) -> contextlib.ExitStack`: enters `override(model=model)` on every module-level agent except `_find_topic_agent`. It replaces `_set_all_step_models`.
- Convert every test that does `monkeypatch.setattr(settings.<step>, "model", stub)`: wrap the call in `with joke_generator._<step>_agent.override(model=stub):` and pass `version=_version("<step>")`. Image generation uses `_google_image_agent`. The `find_topic` tests pass `version=_version("find_topic")`.
- Change the Cloudflare tests: drop the `image_provider` monkeypatch and pass `version=_version("generate_image", "flux-schnell")`. Assert `url.endswith("/ai/run/@cf/black-forest-labs/flux-1-schnell")`.
- Rewrite `test_steps_send_their_configured_model_settings`. Build the version as `AgentVersion("grade_joke", "test", AgentSpec(model="google-cloud:gemini-3.5-flash", model_settings={"thinking": "low", "temperature": 0.0}))`, then assert `info.model_request_parameters.thinking == "low"` and `info.model_settings == {"temperature": 0.0}`.
- Rewrite `test_structured_steps_fall_back_to_prompted_output` with `AgentVersion("grade_joke", "claude-haiku", AgentSpec(model="anthropic:claude-haiku-5-5"))` and the stub `FunctionModel(model, profile=_PROFILE)`. Assert `output_mode == "prompted"`.
- In `test_save_asset`, pass `versions={"generate_joke": _version("generate_joke")}` and assert:
  `metadata["agents"] == {"generate_joke": {"version": "gemini-flash", "model": "google-cloud:gemini-3.5-flash", "model_settings": {"thinking": "high"}}}`
- In `test_main_smoke`, assert `set(metadata["agents"]) == set(STEPS)` and `metadata["agents"]["generate_image"]["version"] == "gemini-flash-image"`.
- Add this test:

```python
def test_main_rejects_an_unknown_version_before_any_model_call(monkeypatch):
    calls = []

    def model(messages: list, info: AgentInfo) -> ModelResponse:
        calls.append(info)
        return ModelResponse(parts=[TextPart(content="A fake topic.")])

    monkeypatch.setattr(settings, "describe_image_agent", "claud-opus")
    message = "Unknown describe_image agent version 'claud-opus'; available: gemini-flash"
    with _find_topic_agent.override(model=FunctionModel(model, profile=_PROFILE), native_tools=[]):
        with pytest.raises(AgentVersionError, match=re.escape(message)):
            joke_generator.main()
    assert calls == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_settings.py tests/test_joke_generator.py -q`
Expected: FAIL. Settings has no `<step>_agent` fields, and the step functions reject `version=`.

- [ ] **Step 3: Implement the settings changes**

In `comedy_factory/settings.py`:
- Remove `StepModelSettings`, `StepConfig`, the eight step-config fields, `google_image`, `image_provider`, `cloudflare_image_model` and any imports they leave unused.
- Set `model_config` back to `SettingsConfigDict(env_prefix="COMEDY_FACTORY_")`.
- Add the nine `<step>_agent` fields, in pipeline order, under a comment saying that each picks a committed version under `agents/<step>/` and can be overridden with `COMEDY_FACTORY_<STEP>_AGENT`.
- Rewrite the image comment: the image backend is now picked by the `generate_image` version, and the Cloudflare credentials stay env vars.

- [ ] **Step 4: Implement `selected_versions()`**

Add it to `agent_versions.py` as specified in the Interfaces block.

- [ ] **Step 5: Implement the pipeline changes** in `comedy_factory/joke_generator.py`

- Change `_run_step(agent, version: AgentVersion, prompt, message_history=None)`. It builds `model = infer_model(version.model)` and calls `run_sync(prompt, model=model, spec=version.spec, message_history=message_history, output_type=_output_spec(agent.output_type, model))`.
- Give each step function the keyword-only `version` parameter from the Interfaces block, and have it call `_run_step` with that version.
- `generate_image`:
  - If `version.model` starts with `CLOUDFLARE_PREFIX`, call `_generate_image_cloudflare(image_prompt, version.model.removeprefix(CLOUDFLARE_PREFIX))`.
  - Otherwise call `_generate_image_google(image_prompt, version)`.
  - `_generate_image_cloudflare` takes `model: str` and uses it in the URL in place of `settings.cloudflare_image_model`.
- `save_asset` writes `"agents": {step: version.as_metadata() for step, version in versions.items()}` into `metadata`.
- `main()`: its first statement is `versions = selected_versions()`. Pass `version=versions["<step>"]` to every step, and pass `versions` to `save_asset`.
- Update the comment above the agent definitions: each run gets its step's agent version, and no agent is bound to a model.

- [ ] **Step 6: Update the env template and add the agents README**

Replace the Anthropic comment in `.env.template` with this text:

```
# Anthropic API — only needed when a step runs an anthropic: agent version,
# e.g. COMEDY_FACTORY_GENERATE_JOKE_AGENT=claude-opus (the version file
# agents/generate_joke/claude-opus.yaml must exist).
```

Replace the Cloudflare comment with:

```
# Cloudflare Workers AI — used by the flux-schnell image version
# (COMEDY_FACTORY_GENERATE_IMAGE_AGENT=flux-schnell).
```

`agents/README.md` should cover:
- **Purpose:** committed per-step agent versions.
- **Layout:** files are `agents/<step>/<version>.yaml`, holding only `model` and `model_settings`.
- **Trying a candidate:**
  1. Add a version file.
  2. Run with `COMEDY_FACTORY_<STEP>_AGENT=<version>`.
  3. Commit it if it works.
  4. Optionally make it the default in `settings.py`.
- **Checks:** the startup and test checks, with the editor schema.
- **Known gap:** misspelled model names surface at the first request.
- **Schema upkeep:** run `just agent-schema` after a pydantic-ai upgrade.

- [ ] **Step 7: Run the tests to verify they pass, then the gates**

Run: `just test && just lint && just mypy`
Expected: all pass, with coverage at 100%.

- [ ] **Step 8: Commit**

```bash
git add comedy_factory tests .env.template agents/README.md
git commit -m "Run each pipeline step on its selected agent version" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Reject unknown COMEDY_FACTORY_* environment variables

**Files:**
- Modify: `comedy_factory/settings.py`, `tests/conftest.py`
- Test: `tests/test_settings.py`

**Interfaces:**
- Consumes: the `Settings` fields from Task 3.
- Produces: `check_env(environ: Mapping[str, str]) -> None` in `settings.py`. It raises `ValueError`, and `Settings.model_post_init` calls `check_env(os.environ)`.

`check_env` builds its message as follows:
- A name is valid if its upper-case form is `COMEDY_FACTORY_` plus the upper-case name of a `Settings` field that has no `validation_alias`. Unprefixed names are ignored.
- The message lists every invalid name.
- A removed name points at its replacement:
  - `<STEP>_MODEL` or `<STEP>__*`, where `<STEP>` is a step from the `*_agent` fields, points at `COMEDY_FACTORY_<STEP>_AGENT`
  - `IMAGE_PROVIDER`, `CLOUDFLARE_IMAGE_MODEL`, `GOOGLE_IMAGE_MODEL` and `GOOGLE_IMAGE__*` point at `COMEDY_FACTORY_GENERATE_IMAGE_AGENT`

pydantic wraps errors raised in `model_post_init` in a `ValidationError`. That's a `ValueError` subclass that keeps the message, so it's acceptable.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_settings.py`; import `pytest` and `check_env`)

```python
def test_unknown_env_vars_fail_naming_them():
    with pytest.raises(ValueError, match="COMEDY_FACTORY_GENERATE_IMAGE_AGNT"):
        check_env({"COMEDY_FACTORY_GENERATE_IMAGE_AGNT": "flux-schnell"})


@pytest.mark.parametrize(("name", "replacement"), [
    ("COMEDY_FACTORY_GENERATE_JOKE__MODEL", "COMEDY_FACTORY_GENERATE_JOKE_AGENT"),
    ("COMEDY_FACTORY_GENERATE_JOKE__MODEL_SETTINGS__THINKING", "COMEDY_FACTORY_GENERATE_JOKE_AGENT"),
    ("COMEDY_FACTORY_GENERATE_JOKE_MODEL", "COMEDY_FACTORY_GENERATE_JOKE_AGENT"),
    ("COMEDY_FACTORY_IMAGE_PROVIDER", "COMEDY_FACTORY_GENERATE_IMAGE_AGENT"),
    ("COMEDY_FACTORY_CLOUDFLARE_IMAGE_MODEL", "COMEDY_FACTORY_GENERATE_IMAGE_AGENT"),
    ("COMEDY_FACTORY_GOOGLE_IMAGE__MODEL", "COMEDY_FACTORY_GENERATE_IMAGE_AGENT"),
])
def test_removed_env_vars_point_at_the_version_override(name, replacement):
    with pytest.raises(ValueError, match=f"{name}.*{replacement}"):
        check_env({name: "x"})


def test_settings_and_unprefixed_env_vars_pass():
    check_env({
        "COMEDY_FACTORY_GENERATE_JOKE_AGENT": "claude-opus",
        "COMEDY_FACTORY_MAX_GRADE_ATTEMPTS": "2",
        "GOOGLE_CLOUD_LOCATION": "global",
        "CLOUDFLARE_API_TOKEN": "token",
        "PATH": "/bin",
    })


def test_aliased_settings_have_no_prefixed_name():
    with pytest.raises(ValueError, match="COMEDY_FACTORY_GOOGLE_CLOUD_LOCATION"):
        check_env({"COMEDY_FACTORY_GOOGLE_CLOUD_LOCATION": "global"})


def test_env_var_names_are_case_insensitive(monkeypatch):
    monkeypatch.setenv("comedy_factory_generate_joke_agent", "claude-opus")
    assert Settings().generate_joke_agent == "claude-opus"
    with pytest.raises(ValueError, match="(?i)comedy_factory_generate_joke_agnt"):
        check_env({"comedy_factory_generate_joke_agnt": "x"})


def test_loading_settings_checks_the_environment(monkeypatch):
    monkeypatch.setenv("COMEDY_FACTORY_IMAGE_PROVIDER", "cloudflare")
    with pytest.raises(ValueError, match="COMEDY_FACTORY_GENERATE_IMAGE_AGENT"):
        Settings()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_settings.py -q`
Expected: FAIL with `ImportError: cannot import name 'check_env'`.

- [ ] **Step 3: Implement `check_env` and the `model_post_init` hook** in `settings.py`, following the rules above

- [ ] **Step 4: Keep the developer's shell out of the tests**

In `tests/conftest.py`, before the first `comedy_factory` import, delete every `os.environ` entry whose upper-case name starts with `COMEDY_FACTORY_`. Add a comment: settings reject unknown names, and the rest would override the committed defaults the tests run against.

- [ ] **Step 5: Run the tests to verify they pass, then the gates**

Run: `just test && just lint && just mypy`
Expected: all pass, with coverage at 100%.

- [ ] **Step 6: Commit**

```bash
git add comedy_factory/settings.py tests/conftest.py tests/test_settings.py
git commit -m "Reject unknown COMEDY_FACTORY_* environment variables at startup" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Verify, open the PR, update issue #4

**Files:** none changed. This task covers the gates, the PR and issue #4.

- [ ] **Step 1: Run every gate**

Run: `just test && just lint && just mypy && just checkov`
Expected: all pass, with coverage at 100%.

- [ ] **Step 2: Confirm nothing still sets models outside version files**

Run: `git grep -nE "StepConfig|image_provider|google_image\b|cloudflare_image_model|__MODEL" -- comedy_factory tests .env.template`
Expected: matches only in `comedy_factory/settings.py` (the removed-variable hints in `check_env`) and in `tests/test_settings.py`.

- [ ] **Step 3: Push and open the PR**

Push `feat/agent-versions` and open a PR against `main`. The body should include:
- a summary
- how to switch a step's version
- the spec path
- references to #4 and #5
- a test plan with the gates checked and two **unchecked** owner items: a live run with the default versions, and a live run with `COMEDY_FACTORY_GENERATE_IMAGE_AGENT=flux-schnell`

The spec makes these live runs the evidence that the initial versions work. End the body with the `🤖 Generated with [Claude Code](https://claude.com/claude-code)` line.

- [ ] **Step 4: Update issue #4**

Run `gh issue edit 4 --body-file <file>`. Replace the profile framing with per-step agent versions:
- every version of a step must pass that step's eval
- the eval command selects versions by name, for example `just eval generate_joke claude-opus`, so candidates can be compared before one is committed
- link the spec

Keep the open questions that still apply.
