# Agent versions: committed, per-step model configurations

## Why

The joke pipeline (`comedy_factory/joke_generator.py`) has nine model-backed
steps: `find_topic`, `generate_subtext`, `grade_subtext`, `generate_joke`,
`grade_joke`, `rephrase_joke`, `write_image_prompt`, `generate_image` and
`describe_image`. Today each step's model and model settings (thinking effort,
temperature, ...) are defaults in `comedy_factory/settings.py` (`StepConfig`),
and any of them can be changed freely through environment variables such as
`COMEDY_FACTORY_GENERATE_JOKE__MODEL=...` or
`COMEDY_FACTORY_GENERATE_JOKE__MODEL_SETTINGS__THINKING=low`.

The owner wants model choice and model settings treated as code. Every
configuration that runs is a committed file, reviewed in a PR, and (once evals
exist, issue #4) shown to pass that step's eval. The unit is one step's agent,
not the whole pipeline. A step can have several configurations that work, for
example one on Gemini and one on Claude Opus, and a single run can mix
providers across steps.

## Goals

- Each step has a set of committed **agent versions**. A version is a
  pydantic-ai agent spec file holding a model and its model settings.
- Settings name the version each step runs, with committed defaults. An
  environment variable can override one step's version, for testing and for
  trying a candidate version before committing it.
- Model strings and model settings exist only in version files.
- A broken version file or version name is caught in CI and at
  pipeline start before any paid model call.
- Every run records which version of each step it used.

## Non-goals

- Evals (issue #4). This change gives evals something to target (a step and a
  version name) but doesn't add them.
- Grade-and-retry loops as output validators (issue #5).
- Prompt variants. Prompts stay in `prompts/`, and version files don't carry
  instructions.
- Picking versions automatically, at random, or by rotation.
- Changing settings that aren't about models: `max_grade_attempts`,
  `rephrasing_count`, `max_recent_topics`, the directories and the credentials
  keep working as they do today.

## Anti-goals

Any of these would make the change a failure even if it works:

- Any way to set a model string or model setting without a committed file,
  whether through an env var, a CLI flag or a settings field that holds a
  model name.
- A hand-rolled config format or validator where pydantic-ai already provides
  one: the agent spec file format and its JSON schema.
- A mistyped version name, broken version file or misspelled setting that
  silently falls back to something else, or that only surfaces after earlier
  steps have already spent money. Misspelled model names are the one accepted
  exception (see Known gaps).

## Constraints

- Python 3.12, pydantic-ai 2.27, pydantic-settings 2.15.
- The existing gates stay: ruff, mypy, checkov and 100% test coverage
  (`just test`), mirrored in `.github/workflows/ci.yml`.
- Tests never call real models: `tests/conftest.py` sets
  `models.ALLOW_MODEL_REQUESTS = False`. CI has no model credentials.
- pydantic-ai's `AgentSpec` silently ignores unknown keys when it loads a
  file, and it doesn't validate `model_settings`. The JSON schema from
  `AgentSpec.model_json_schema_with_capabilities()` rejects unknown top-level
  keys, and bad values for the settings it knows, like `thinking: hgh`. To let
  provider-specific settings (`google_*`, `anthropic_*`) through, it accepts
  any key inside `model_settings`. So a misspelled setting such as
  `thinkng: high` passes the schema and is silently ignored at run time.
  Validation uses the schema plus a key check against pydantic-ai's settings
  classes (see Version-file checks).
- `infer_model` only checks a model string's provider prefix:
  `google-cloud:gemini-3.5-flsh` builds fine and fails at its first request.
  pydantic-ai's `KnownModelName` list can't serve as an allowlist, because it
  lacks models this repo uses, such as `anthropic:claude-opus-5-5`.
- pydantic-settings silently ignores environment variables that match no
  setting.
- Spec values passed at run time (`run_sync(..., spec=...)`) are added to the
  agent's own configuration. A version file carrying `instructions` or
  `capabilities` would change the step's prompt or tools, so version files are
  limited to `model` and `model_settings`.
- Cloudflare's FLUX model isn't a pydantic-ai model, so `infer_model` can't
  build it.

## Design

### Version files

```
agents/
  find_topic/
    gemini-flash.yaml
  generate_subtext/
    gemini-flash.yaml
  grade_subtext/
    gemini-flash.yaml
  generate_joke/
    gemini-flash.yaml
  grade_joke/
    gemini-flash.yaml
  rephrase_joke/
    gemini-flash.yaml
  write_image_prompt/
    gemini-flash.yaml
  generate_image/
    gemini-flash-image.yaml
    flux-schnell.yaml
  describe_image/
    gemini-flash.yaml
```

Each file is a standard pydantic-ai agent spec, the format
`AgentSpec.from_file` loads, limited to `model` and `model_settings`:

```yaml
model: google-cloud:gemini-3.5-flash
model_settings:
  thinking: high
```

The version name is the file name without `.yaml`. Versions are named after
their model, with a suffix when a variant is added (for example
`gemini-flash-thinking-low`).

The initial versions carry today's `settings.py` defaults over unchanged:

| Step | Version | Model | Model settings |
|---|---|---|---|
| find_topic | gemini-flash | `google-cloud:gemini-3.5-flash` | none (model default) |
| generate_subtext | gemini-flash | `google-cloud:gemini-3.5-flash` | `thinking: medium` |
| grade_subtext | gemini-flash | `google-cloud:gemini-3.5-flash` | `thinking: medium` |
| generate_joke | gemini-flash | `google-cloud:gemini-3.5-flash` | `thinking: high` |
| grade_joke | gemini-flash | `google-cloud:gemini-3.5-flash` | `thinking: medium` |
| rephrase_joke | gemini-flash | `google-cloud:gemini-3.5-flash` | `thinking: low` |
| write_image_prompt | gemini-flash | `google-cloud:gemini-3.5-flash` | `thinking: low` |
| generate_image | gemini-flash-image | `google-cloud:gemini-3.1-flash-image` | none |
| generate_image | flux-schnell | `cloudflare:@cf/black-forest-labs/flux-1-schnell` | none |
| describe_image | gemini-flash | `google-cloud:gemini-3.5-flash` | `thinking: false` |

No Claude versions are committed in this change. Claude versions are added
once they've been tried and work.

Tools stay in code. For example, `find_topic`'s `WebSearch` capability stays
on its agent in `joke_generator.py`.

### Choosing versions

`Settings` gets one field per step naming its version, for example
`generate_joke_agent: str = "gemini-flash"`. pydantic-settings makes each one
overridable as `COMEDY_FACTORY_<STEP>_AGENT`, for example
`COMEDY_FACTORY_GENERATE_JOKE_AGENT=claude-opus`. Every default is
`gemini-flash`, except `generate_image_agent`, which defaults to
`gemini-flash-image`.

An override names a file in that step's folder. Paths aren't accepted, so
every version that runs lives under `agents/`.

A new `agents_dir` setting points at `agents/`, mirroring `prompts_dir`. Tests
can point it at a temporary folder.

The following are removed from `settings.py`:
- `StepConfig` and `StepModelSettings`
- the nine step-config fields, including `google_image`
- `image_provider` and `cloudflare_image_model`
- the nested env-var options (`env_nested_delimiter` and
  `nested_model_default_partial_update`)

The Cloudflare credentials (`CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`)
stay env vars, because they're secrets rather than configuration.

### Running a step

Each step loads its selected version and runs its agent with it. The model
comes from the version's `model`, which is still resolved with `infer_model`,
because `_output_spec` needs the model to choose native or prompted output.
The version is also passed as `spec=`, so its `model_settings` apply.

`generate_image` dispatches on its version's model:
- A `cloudflare:` prefix calls Workers AI with the model name after the
  prefix. This is today's `_generate_image_cloudflare`.
- Any other model runs the image agent. This is today's
  `_generate_image_google`.

### Failing early

Two checks run before the first model call.

**Environment variables.** Loading settings fails on any `COMEDY_FACTORY_*`
variable that doesn't match a setting, and the error names the variable. This
catches misspelled overrides, such as `COMEDY_FACTORY_GENERATE_IMAGE_AGNT`. It
also catches the variables this change removes (`COMEDY_FACTORY_GENERATE_JOKE__MODEL`,
`COMEDY_FACTORY_GENERATE_JOKE__MODEL_SETTINGS__THINKING`,
`COMEDY_FACTORY_IMAGE_PROVIDER`, ...), which pydantic-settings would otherwise
ignore silently. For a removed variable, the error points at the version
override that replaces it.

**Selected versions.** `main()` loads every selected version and runs the
full version-file checks below on each one. That way a broken candidate fails
before any paid call, not just a mistyped name. An unknown version name raises
an error that names the step, the requested version and the versions that
exist, for example:
`Unknown generate_joke agent version 'claud-opus'; available: gemini-flash`.

Candidate versions run before CI ever sees them, so this startup check is
their gate. The tests are the gate for committed versions.

### Version-file checks

One function checks a version file. `main()` runs it on each selected version,
and the tests run it on every file under `agents/`. A version file must:
- validate against pydantic-ai's agent spec schema
  (`AgentSpec.model_json_schema_with_capabilities()`), which catches unknown top-level
  keys and bad values for known settings
- contain only `model` and `model_settings`, with `model` set
- use only `model_settings` keys that pydantic-ai defines for the model's
  provider. Those are the common `ModelSettings` keys plus the provider's
  settings class: `GoogleModelSettings` for `google-cloud:` models and
  `AnthropicModelSettings` for `anthropic:` models. This catches misspelled
  keys like `thinkng`.
- name a model that can be built, meaning `infer_model` succeeds. The tests
  set dummy API keys. At startup, this also catches a missing
  `ANTHROPIC_API_KEY` for a Claude version.
- if the model is a `cloudflare:` model, live in `generate_image/` and have no
  `model_settings`. Workers AI calls don't use them, so any settings would be
  silently ignored.

The tests also check that:
- every step's default version exists
- every folder under `agents/` must be a known step, which catches stray or
  misspelled step folders

### Editor checks

None. This design first committed the generated schema
(`agents/agent-spec.schema.json`) so that VS Code's YAML extension could
autocomplete and flag bad values while a version file is edited. The owner
dropped it on 2026-10-31: nothing at run time or in the tests reads the file,
and keeping a 90 KB generated copy current after every pydantic-ai upgrade
wasn't worth the convenience. The startup check and the tests catch everything
the editor did, and more.

### Known gaps

**Misspelled model names.** Nothing offline can tell `gemini-3.5-flsh` from a
real model (see Constraints). A mistyped model therefore fails at its step's
first request, after the earlier steps have run. This is accepted for now:
committed versions are proven by a live run (and later by evals), and a
candidate's typo costs one partial run. If it becomes a problem, the fix is a
startup preflight that asks each provider whether the model exists. That would
bring network calls and credentials into startup and need stubbing in tests.

### Traceability

`metadata.json` gains an `agents` object recording, for each step, the version
name, model and model settings that ran:

```json
"agents": {
  "generate_joke": {
    "version": "gemini-flash",
    "model": "google-cloud:gemini-3.5-flash",
    "model_settings": {"thinking": "high"}
  }
}
```

### Dependencies

- pydantic-ai's `spec` extra, for example `pydantic-ai-slim[spec]` with the
  same version floor as `pydantic-ai`. It declares PyYAML, which
  `AgentSpec.from_file` needs and which only arrives transitively today.
- `jsonschema`, as a runtime dependency because the startup check uses it. It
  only arrives transitively today, through `mcp`.
- `uv.lock` updated to match.

### Docs

- `.env.template`: replace the `COMEDY_FACTORY_GENERATE_JOKE__MODEL=...`
  example with the version override, and say that the version file must
  exist. Point the Cloudflare comment at the `flux-schnell` version.
- A short note on trying a candidate, in the README or `agents/README.md`
  (builder's call):
  1. Add a version file.
  2. Run with the env override.
  3. Commit it if it works.
  4. Optionally change the default in `settings.py`.

## Who decided what

Decided by the owner:
- Configurations are per step, not whole-pipeline profiles. Providers can mix
  across steps.
- Settings pick the version for each step, with committed defaults, and an env
  var overrides that choice for testing.
- Only configurations that work get committed. Evals will become the bar for
  "works".
- It must be easy to try (and later evaluate) candidates before deciding on
  one.
- No committed schema file and no editor checks (decided after the build,
  2026-10-31; see Editor checks).

Proposed by Claude and accepted by the owner:
- The `agents/` folder, version name = file name, named after the model with
  a suffix for variants.
- pydantic-ai's spec file format, validated with its generated JSON schema.
- Image generation as an ordinary step. The `cloudflare:` model prefix
  replaces `image_provider` and `cloudflare_image_model`.
- Recording versions in `metadata.json`.
- An initial set of today's Gemini configurations plus FLUX, with no Claude
  versions yet.
- pydantic-ai's `spec` extra and `jsonschema` as declared dependencies.

Calls made while writing this spec, which the owner should check:
- Version files are limited to `model` and `model_settings`.
- Overrides take a version name only, never a path.
- A new `agents_dir` setting.
- `main()` runs the full version-file checks on every selected version before
  the first model call, not just the name check. As a result `jsonschema` is
  a runtime dependency.
- `model_settings` keys are checked against pydantic-ai's provider settings
  classes.
- `cloudflare:` versions take no model settings.
- Unknown `COMEDY_FACTORY_*` environment variables fail at startup.
- Misspelled model names are accepted as a known gap rather than adding a
  network preflight.

## Left to the builder

- Module layout. For example, a small `comedy_factory/agent_versions.py`
  loader, in the spirit of `prompts.py`.
- Exact error wording, docstrings and comments.
- How tests stub models. pydantic-ai's `agent.override(model=...)` takes
  precedence over the model passed to `run_sync` and still applies the spec's
  settings (checked against 2.27), so the existing seam keeps working. Point
  `agents_dir` at a temporary folder where a test needs special versions.
- Whether helpers take a loaded version or a step name.
- How the environment-variable check reads the environment (for example, a
  settings validator that scans `os.environ`), and how tests keep the
  developer's shell from leaking into it.

## Done when

- The pipeline runs with versions loaded from `agents/`, and `metadata.json`
  records them.
- `COMEDY_FACTORY_<STEP>_AGENT` switches a step to another committed version.
  A mistyped version name, a broken version file (including a misspelled
  setting key) or an unknown `COMEDY_FACTORY_*` variable fails before any
  model call.
- No model string or model setting can be set outside a version file.
- The tests cover the checks above. `just test` (100% coverage), `just lint`
  and `just mypy` pass, and CI is green.
- Before merge, the owner does one live run with the default versions and one
  with `COMEDY_FACTORY_GENERATE_IMAGE_AGENT=flux-schnell`. Until evals exist,
  those runs are the evidence that the initial versions work. They need
  credentials and cost money, so they can't run in CI.
- Issue #4 is updated: evals are per step, every version of a step must pass
  that step's eval, and the eval command selects versions by name.
