"""Small shared helpers (paths, encoding)."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image


def project_root() -> Path:
    """Repository root (parent of the `app` package directory)."""
    return Path(__file__).resolve().parent.parent


def image_to_base64_png(image: Image.Image) -> str:
    """Encode a PIL image as a base64 PNG string (no data URL prefix)."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.standard_b64encode(buffer.getvalue()).decode("ascii")
