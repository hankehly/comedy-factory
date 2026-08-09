# Create missing prompt files from their committed *.template.md boilerplate
init-prompts:
    #!/usr/bin/env sh
    for template in prompts/*.template.md; do
        target="${template%.template.md}.md"
        test -f "$target" || cp "$template" "$target"
    done

test *args:
    uv run pytest --cov=comedy_factory --cov-report=term-missing {{args}}

# Draws on the pristine image-original.jpg and writes a datetime-stamped
# image-captioned-<YYYYmmdd-HHMMSS>.jpg (later stamps are newer; existing
# versions are kept untouched), so it can be re-run on the same bundle any
# number of times. Example:
#   just recaption output/20260809-153859 "The funnier caption"
# Re-render the caption on a saved asset bundle
recaption bundle caption:
    uv run python -m comedy_factory.recaption {{quote(bundle)}} {{quote(caption)}}
