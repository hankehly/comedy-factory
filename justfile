# Create missing prompt files from their committed *.template.md boilerplate
init-prompts:
    #!/usr/bin/env sh
    for template in prompts/*.template.md; do
        target="${template%.template.md}.md"
        test -f "$target" || cp "$template" "$target"
    done

test *args:
    uv run pytest --cov=comedy_factory --cov-report=term-missing {{args}}
