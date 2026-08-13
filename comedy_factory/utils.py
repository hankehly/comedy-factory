"""General helpers shared across the comedy factory workflows."""

import io

from PIL import Image
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


def image_media_type(image: bytes) -> str:
    """Sniff the media type of image bytes; only the header is parsed, the
    image is not decoded."""
    media_type = Image.open(io.BytesIO(image)).get_format_mimetype()
    if media_type is None:
        raise ValueError("Image bytes are in a format with no known media type")
    return media_type


def compose_alt_text(image_description: str, caption: str) -> str:
    """Compose posting alt text: the visual description of the image followed
    by the caption sentence."""
    return f'{image_description} Caption reads: "{caption}"'
