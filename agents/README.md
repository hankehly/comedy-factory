# Agent versions

Each model-backed step of the joke pipeline runs one **agent version**: a
committed file holding the model the step uses and that model's settings. A
step can have several versions that work, for example one on Gemini and one on
Claude, and a single run can mix providers across steps.

## Layout

```
agents/
  agent-spec.schema.json    # pydantic-ai's spec schema, generated (see below)
  <step>/<version>.yaml     # one file per version
```

A version file is a standard pydantic-ai agent spec, limited to two keys:

```yaml
# yaml-language-server: $schema=../agent-spec.schema.json
model: google-cloud:gemini-3.5-flash
model_settings:
  thinking: high
```

The file name without `.yaml` is the version name. Each step's setting in
`comedy_factory/settings.py` names the version it runs, for example
`generate_joke_agent = "gemini-flash"`. Image generation also accepts Cloudflare
Workers AI models, written as `cloudflare:<model>`, which take no model
settings.

## Trying a candidate version

1. Add a version file, for example `agents/generate_joke/claude-opus.yaml`.
2. Run the pipeline with the step's override:
   `COMEDY_FACTORY_GENERATE_JOKE_AGENT=claude-opus uv run python main.py`
3. Commit the file if it works.
4. Optionally, make it the default by changing `generate_joke_agent` in
   `comedy_factory/settings.py`.

The override picks a version by name, never by path, so models and model
settings only ever come from files in this folder.

## Checks

- **At startup**, any `COMEDY_FACTORY_*` environment variable that isn't a
  setting stops the run, so a misspelled override can't be silently ignored.
  The pipeline then loads every selected version before its first model call.
  An unknown version name fails with the list of versions that exist. Each
  file must also pass these checks:
  - valid YAML, holding only `model` and `model_settings`
  - values that match the schema, so `thinking: hgh` fails
  - setting keys that pydantic-ai defines for the model's provider, so
    `thinkng` fails
  - a model pydantic-ai can build, so an `anthropic:` model fails without
    `ANTHROPIC_API_KEY`
- **In the tests (and so in CI)**, every committed file gets the same checks.
  The tests also check that each step's default version exists, that every
  folder here is a known step, and that the schema file is current.
- **In the editor**, the first-line comment lets Red Hat's YAML extension for
  VS Code autocomplete keys and flag bad values from the schema. It can't flag
  a misspelled setting key, because the schema allows provider-specific ones;
  the startup check and the tests do.

**Known gap:** nothing offline can tell a misspelled model name such as
`gemini-3.5-flsh` from a real one, so it fails at that step's first request.

## Keeping the schema current

`agent-spec.schema.json` is generated from the installed pydantic-ai. After
upgrading pydantic-ai, run `just agent-schema`. A test fails until the
committed copy matches.
