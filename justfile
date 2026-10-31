# Create missing prompt files from their committed *.template.md boilerplate
init-prompts:
    #!/usr/bin/env sh
    for template in prompts/*.template.md; do
        target="${template%.template.md}.md"
        test -f "$target" || cp "$template" "$target"
    done

# Run the unit tests with a coverage report (config in pyproject.toml [tool.coverage.*]; gated at 100%)
test *args:
    uv run pytest --cov --cov-report=term-missing {{args}}

# Lint Python with ruff (rules in pyproject.toml [tool.ruff]; e.g. just lint --fix)
lint *args:
    uv run ruff check . {{args}}

# Type-check Python with mypy (checked files + config in pyproject.toml [tool.mypy])
mypy *args:
    uv run mypy {{args}}

# Version pinned here and in .github/workflows/ci.yml — bump both together.
# Scan workflows and committed files with checkov (config in .checkov.yaml)
checkov *args:
    uvx checkov@3.3.11 {{args}}

# Draws on the pristine image-original.jpg and writes a datetime-stamped
# image-captioned-<YYYYmmdd-HHMMSS>.jpg (later stamps are newer; existing
# versions are kept untouched), so it can be re-run on the same bundle any
# number of times. Example:
#   just recaption output/20260809-153859 "The funnier caption"
# Re-render the caption on a saved asset bundle
recaption bundle caption:
    uv run python -m comedy_factory.recaption {{quote(bundle)}} {{quote(caption)}}
