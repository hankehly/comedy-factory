"""Rendering a joke caption onto an image (workflow step 9)."""

import io
import math

from PIL import Image, ImageDraw, ImageFont


def _wrap_caption(
    caption: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, max_width: int
) -> list[str]:
    """Greedily wrap the caption into lines at most `max_width` pixels wide.

    A single word wider than `max_width` gets its own (overflowing) line.
    """
    lines: list[str] = []
    line = ""
    for word in caption.split():
        candidate = f"{line} {word}".strip()
        if line and font.getlength(candidate) > max_width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


def render_caption(image: bytes, caption: str) -> bytes:
    """Return the image extended with a white caption bar showing the joke."""
    base = Image.open(io.BytesIO(image)).convert("RGB")

    # Scale typography with image width so captions look the same at any
    # resolution. load_default(size=...) uses Pillow's embedded vector font,
    # so no font file needs to ship with the project.
    font_size = max(16, base.width // 20)
    padding = font_size
    spacing = font_size // 3
    font = ImageFont.load_default(size=font_size)

    lines = _wrap_caption(caption, font, base.width - 2 * padding)
    text = "\n".join(lines)

    draw = ImageDraw.Draw(base)
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing)
    # The bbox is float-typed; round the bar height up so no text is clipped.
    text_height = math.ceil(bbox[3] - bbox[1])

    canvas = Image.new("RGB", (base.width, base.height + text_height + 2 * padding), "white")
    canvas.paste(base, (0, 0))
    ImageDraw.Draw(canvas).multiline_text(
        (
            (base.width - (bbox[2] - bbox[0])) // 2 - bbox[0],
            base.height + padding - bbox[1],
        ),
        text,
        font=font,
        fill="black",
        spacing=spacing,
        align="center",
    )

    buffer = io.BytesIO()
    canvas.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()
