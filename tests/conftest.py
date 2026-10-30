import os

import pytest

# The Google provider is constructed at import time and demands an API key,
# even though tests never make real requests. Must be set before any
# comedy_factory import.
os.environ.setdefault("GOOGLE_API_KEY", "test-api-key")
# Agent version checks build each version's model with infer_model, which
# demands the provider's API key; Claude versions need this one.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-api-key")
# Settings reject unknown COMEDY_FACTORY_* variables, and the rest would
# override the committed defaults the tests run against, so drop any the
# developer's shell carries before comedy_factory is imported.
for name in [name for name in os.environ if name.upper().startswith("COMEDY_FACTORY_")]:
    del os.environ[name]

from pydantic_ai import models  # noqa: E402

from comedy_factory.settings import settings  # noqa: E402

# Safety net: any test that accidentally reaches a real provider model raises
# instead of making a network call. TestModel/FunctionModel are unaffected.
models.ALLOW_MODEL_REQUESTS = False


@pytest.fixture(autouse=True)
def isolated_output_dir(monkeypatch, tmp_path):
    """Point the asset-bundle directory at a fresh temp dir for every test, so
    no test reads the developer's real output/ bundles (the topic scan feeds
    recent bundles into its prompt) or writes into it. Tests that need the
    directory in a particular state set it up themselves."""
    monkeypatch.setattr(settings, "output_dir", tmp_path / "output")
