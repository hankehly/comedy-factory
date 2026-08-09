"""Rewrite the caption on a saved asset bundle's image.

Renders the new caption onto the bundle's pristine `image-original.jpg` and
writes it to a datetime-stamped `image-captioned-<YYYYmmdd-HHMMSS>.jpg` —
later stamps are newer, and `image-captioned.jpg` is the pipeline's first
version. Nothing is modified or overwritten, so this can be re-run on the
same bundle any number of times and every caption version is retained.
"""

import argparse
from datetime import datetime
from pathlib import Path

from comedy_factory.captioning import render_caption


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "bundle_dir",
        type=Path,
        help="Asset bundle directory (e.g. output/20260809-153859)",
    )
    parser.add_argument("caption", help="The new caption text")
    args = parser.parse_args(argv)

    original_path = args.bundle_dir / "image-original.jpg"
    if not original_path.is_file():
        parser.error(f"No original image found at {original_path}")

    captioned_path = args.bundle_dir / "image-captioned.jpg"
    if captioned_path.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        captioned_path = args.bundle_dir / f"image-captioned-{stamp}.jpg"
        if captioned_path.exists():
            # Same-second re-run: fall back to microsecond resolution rather
            # than overwrite a retained version.
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            captioned_path = args.bundle_dir / f"image-captioned-{stamp}.jpg"

    captioned_path.write_bytes(render_caption(original_path.read_bytes(), args.caption))
    print(f"Wrote {captioned_path}")


if __name__ == "__main__":
    main()
