test *args:
    uv run pytest --cov=comedy_factory --cov-report=term-missing {{args}}
