import os

# The Google provider is constructed at import time and demands an API key,
# even though tests never make real requests. Must be set before any
# comedy_factory import.
os.environ.setdefault("GOOGLE_API_KEY", "test-api-key")

from pydantic_ai import models  # noqa: E402

# Safety net: any test that accidentally reaches a real provider model raises
# instead of making a network call. TestModel/FunctionModel are unaffected.
models.ALLOW_MODEL_REQUESTS = False
