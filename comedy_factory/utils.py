"""General helpers shared across the comedy factory workflows."""

from pydantic_ai.messages import FilePart, ModelResponse, TextPart


def response_text(response: ModelResponse) -> str:
    """Concatenate the text parts of a model response."""
    return "".join(
        part.content for part in response.parts if isinstance(part, TextPart)
    )


def response_image(response: ModelResponse) -> bytes:
    """Return the binary content of the first file part of a model response."""
    for part in response.parts:
        if isinstance(part, FilePart):
            return part.content.data
    raise ValueError("Model response contains no image")
