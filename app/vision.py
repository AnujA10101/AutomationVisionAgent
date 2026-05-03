"""Screen capture utilities and coordinate-map debug rendering."""

from __future__ import annotations

import base64
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import mss
from PIL import Image, ImageDraw, ImageFont

from app.config import SETTINGS
from app.utils import project_root

if TYPE_CHECKING:
    from app.targeting import LocatedTarget


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


def _label_for_col(idx: int) -> str:
    letters = ""
    n = idx
    while True:
        n, rem = divmod(n, 26)
        letters = chr(ord("A") + rem) + letters
        if n == 0:
            break
        n -= 1
    return letters


def create_coordinate_map_screenshot(
    image: Image.Image,
    grid_cols: int = 12,
    grid_rows: int = 8,
    fine_grid_spacing: int = 50,
    cursor_position: tuple[int, int] | None = None,
) -> Image.Image:
    """Return a copy annotated with coarse cell labels (A1..), fine grid, and axes."""
    out = image.copy()
    draw = ImageDraw.Draw(out, "RGBA")
    w, h = out.size
    cols = max(1, int(grid_cols))
    rows = max(1, int(grid_rows))
    fine = max(10, int(fine_grid_spacing))
    font = ImageFont.load_default()

    # fine grid
    fine_color = (150, 150, 150, 45)
    for x in range(0, w, fine):
        draw.line([(x, 0), (x, h)], fill=fine_color, width=1)
    for y in range(0, h, fine):
        draw.line([(0, y), (w, y)], fill=fine_color, width=1)

    # coarse grid
    coarse_color = (255, 90, 90, 120)
    for i in range(cols + 1):
        x = int(round(i * w / cols))
        x = min(max(0, x), w - 1)
        draw.line([(x, 0), (x, h)], fill=coarse_color, width=2)
    for j in range(rows + 1):
        y = int(round(j * h / rows))
        y = min(max(0, y), h - 1)
        draw.line([(0, y), (w, y)], fill=coarse_color, width=2)

    # axes pixel labels
    axis_fill = (255, 240, 120, 220)
    for x in range(0, w, max(50, fine)):
        draw.text((min(x + 2, w - 30), 2), str(x), fill=axis_fill, font=font)
    for y in range(0, h, max(50, fine)):
        draw.text((2, min(y + 2, h - 12)), str(y), fill=axis_fill, font=font)

    # cell labels
    for j in range(rows):
        y1 = int(round(j * h / rows))
        y2 = int(round((j + 1) * h / rows))
        for i in range(cols):
            x1 = int(round(i * w / cols))
            x2 = int(round((i + 1) * w / cols))
            cell = f"{_label_for_col(i)}{j + 1}"
            tx = min(max(4, x1 + 4), max(4, x2 - 40))
            ty = min(max(20, y1 + 3), max(20, y2 - 14))
            draw.rectangle([tx - 2, ty - 1, tx + 28, ty + 11], fill=(0, 0, 0, 110))
            draw.text((tx, ty), cell, fill=(255, 255, 255, 240), font=font)

    draw.rectangle([0, 0, w - 1, h - 1], outline=(180, 180, 180, 230), width=1)
    draw.rectangle([4, 4, 170, 19], fill=(0, 0, 0, 160))
    draw.text((6, 6), f"Screenshot: {w} x {h}", fill=(0, 255, 200, 255), font=font)

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


def draw_located_target(
    image: Image.Image,
    target: "LocatedTarget",
    output_path: Path,
) -> Path:
    """Draw target box + click point metadata to `output_path` and return it."""
    out = image.copy()
    draw = ImageDraw.Draw(out, "RGBA")
    font = ImageFont.load_default()

    if target.found and None not in (target.x1, target.y1, target.x2, target.y2):
        assert target.x1 is not None and target.y1 is not None
        assert target.x2 is not None and target.y2 is not None
        draw.rectangle([target.x1, target.y1, target.x2, target.y2], outline=(255, 60, 60, 255), width=3)
    if target.found and target.click_x is not None and target.click_y is not None:
        r = 5
        draw.ellipse(
            [target.click_x - r, target.click_y - r, target.click_x + r, target.click_y + r],
            fill=(255, 60, 60, 235),
        )

    tag = f"{target.target_label or 'none'} | {target.grid_cell or '-'} | conf={target.confidence:.2f}"
    draw.rectangle([6, 24, min(out.width - 6, 460), 42], fill=(0, 0, 0, 170))
    draw.text((8, 27), tag, fill=(255, 255, 255, 255), font=font)
    if target.reason:
        rs = target.reason[:90]
        draw.rectangle([6, 44, min(out.width - 6, 640), 62], fill=(0, 0, 0, 145))
        draw.text((8, 47), rs, fill=(230, 230, 230, 255), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path, format="PNG")
    return output_path


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
