"""Screen capture utilities (mss + Pillow). Annotation helpers for visual grounding."""

from __future__ import annotations

import base64
import io
from datetime import datetime, timezone
from pathlib import Path

import mss
from PIL import Image, ImageDraw, ImageFont

from app.config import SETTINGS
from app.utils import project_root


def create_annotated_screenshot(
    image: Image.Image,
    cursor_position: tuple[int, int] | None = None,
    grid_spacing: int = 100,
    draw_labels: bool = True,
) -> Image.Image:
    """
    Return a new image with grid lines, axis labels, size text, and optional cursor marker.

    Does not mutate the input image.
    """
    out = image.copy()
    draw = ImageDraw.Draw(out, "RGBA")
    w, h = out.size
    grid_spacing = max(8, int(grid_spacing))

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    line_color = (255, 90, 90, 55)
    for x in range(0, w, grid_spacing):
        draw.line([(x, 0), (x, h)], fill=line_color, width=1)
    for y in range(0, h, grid_spacing):
        draw.line([(0, y), (w, y)], fill=line_color, width=1)

    if draw_labels:
        label_fill = (255, 240, 120, 255)
        for x in range(0, w, grid_spacing):
            draw.text((min(x + 2, w - 24), 2), str(x), fill=label_fill, font=font)
        for y in range(0, h, grid_spacing):
            draw.text((2, min(y + 2, h - 10)), str(y), fill=label_fill, font=font)

    draw.rectangle([0, 0, w - 1, h - 1], outline=(180, 180, 180, 220), width=1)
    size_tag = f"{w} x {h} px"
    draw.text((4, 4), size_tag, fill=(0, 255, 200, 255), font=font)

    if cursor_position is not None:
        cx, cy = int(cursor_position[0]), int(cursor_position[1])
        r = max(8, min(w, h) // 80)
        ring = (0, 220, 255, 255)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=ring, width=3)
        arm = r * 2
        draw.line([(cx - arm, cy), (cx + arm, cy)], fill=ring, width=2)
        draw.line([(cx, cy - arm), (cx, cy + arm)], fill=ring, width=2)

    return out


def save_debug_image(image: Image.Image, name_prefix: str = "debug") -> Path:
    """
    Save a PNG under `screenshots/` at `{name_prefix}.png` (project root relative).

    Slashes in `name_prefix` are sanitized for safe filenames.
    """
    root = project_root()
    out_dir = root / "screenshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = name_prefix.replace("/", "_").strip() or "debug"
    path = out_dir / f"{safe}.png"
    image.save(path, format="PNG")
    return path


class ScreenCapture:
    """Capture monitor images using `mss`."""

    def __init__(self, monitor_index: int | None = None) -> None:
        idx = SETTINGS.screen.monitor_index if monitor_index is None else monitor_index
        self._monitor_index = idx

    def capture(self) -> Image.Image:
        """Grab the configured monitor and return a PIL RGB image."""
        with mss.mss() as sct:
            try:
                monitor = sct.monitors[self._monitor_index]
            except IndexError as exc:
                raise ValueError(
                    f"Invalid monitor_index={self._monitor_index}; "
                    f"mss exposes indices 0..{len(sct.monitors) - 1} (0 = all monitors combined)."
                ) from exc
            raw = sct.grab(monitor)
        return Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")

    def save_screenshot(self, image: Image.Image, directory: str | Path = "screenshots") -> Path:
        """Save a timestamped PNG under `directory` (relative paths resolve from project root)."""
        root = project_root()
        out_dir = Path(directory)
        if not out_dir.is_absolute():
            out_dir = root / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = out_dir / f"screenshot_{ts}.png"
        image.save(path, format="PNG")
        return path

    @staticmethod
    def resize_for_llm(image: Image.Image, max_width: int | None = None) -> Image.Image:
        """Resize so width is at most `max_width`, preserving aspect ratio."""
        cap = SETTINGS.screen.screenshot_width if max_width is None else max_width
        w, h = image.size
        if w <= cap:
            return image.copy()
        new_w = cap
        new_h = max(1, int(round(h * (new_w / w))))
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    @staticmethod
    def image_to_base64(image: Image.Image, image_format: str = "PNG") -> str:
        """
        Encode a PIL image as a base64 string (raw base64 only, no data URL prefix).

        `image_format` is a Pillow format name (e.g. ``PNG``, ``JPEG``).
        """
        try:
            buffer = io.BytesIO()
            image.save(buffer, format=image_format)
            return base64.standard_b64encode(buffer.getvalue()).decode("ascii")
        except Exception as exc:
            raise ValueError(f"Failed to encode image to base64 ({image_format}): {exc}") from exc
