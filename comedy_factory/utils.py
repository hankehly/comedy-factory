"""General helpers shared across the comedy factory workflows."""

from pydantic_ai.messages import ModelResponse, TextPart


def response_text(response: ModelResponse) -> str:
    """Concatenate the text parts of a model response."""
    return "".join(
        part.content for part in response.parts if isinstance(part, TextPart)
    )
