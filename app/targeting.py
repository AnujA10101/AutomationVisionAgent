"""Visual targeting: bounding boxes, template matching, validation."""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from pydantic import BaseModel, Field, field_validator, model_validator

from app.utils import project_root

logger = logging.getLogger(__name__)


class TargetBox(BaseModel):
    """Axis-aligned UI target in image pixel coordinates."""

    label: str
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = Field(ge=0.0, le=1.0)
    source: str
    reason: str | None = None

    @model_validator(mode="after")
    def validate_geometry(self) -> TargetBox:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(f"Invalid box: need x2>x1 and y2>y1, got ({self.x1},{self.y1})-({self.x2},{self.y2})")
        return self

    def within_bounds(self, width: int, height: int) -> bool:
        """Return True if inclusive corners lie inside pixel indices [0, width-1] x [0, height-1]."""
        if width <= 0 or height <= 0:
            return False
        return (
            0 <= self.x1 <= self.x2 < width
            and 0 <= self.y1 <= self.y2 < height
        )


def center_of_box(box: TargetBox) -> tuple[int, int]:
    """Integer center of the bounding box."""
    cx = (box.x1 + box.x2) // 2
    cy = (box.y1 + box.y2) // 2
    return cx, cy


def scale_target_box_to_capture(
    box: TargetBox,
    *,
    original_size: tuple[int, int],
    resized_size: tuple[int, int],
) -> TargetBox:
    """Map a box from resized screenshot space to full capture space."""
    from app.coordinates import scale_xy

    ow, oh = original_size
    rw, rh = resized_size
    ax1, ay1 = scale_xy(box.x1, box.y1, (ow, oh), (rw, rh))
    ax2, ay2 = scale_xy(box.x2, box.y2, (ow, oh), (rw, rh))
    x1, x2 = sorted((ax1, ax2))
    y1, y2 = sorted((ay1, ay2))
    return box.model_copy(update={"x1": x1, "y1": y1, "x2": x2, "y2": y2})


def draw_target_box(image: Image.Image, box: TargetBox) -> Image.Image:
    """Return a copy of `image` with the target rectangle drawn (no mutation)."""
    out = image.copy()
    draw = ImageDraw.Draw(out, "RGBA")
    outline = (255, 60, 60, 255)
    width = max(2, min(out.size) // 400 or 2)
    draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline=outline, width=width)
    cx, cy = center_of_box(box)
    r = 6
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 60, 60, 220))
    label = f"{box.label} ({box.confidence:.2f})"
    try:
        font = ImageFont.load_default()
        tw, th = draw.textbbox((0, 0), label, font=font)[2:4]
    except Exception:
        tw, th = (len(label) * 6, 12)
        font = None
    pad = 2
    lx = min(box.x1, out.width - tw - pad * 2)
    ly = max(0, box.y1 - th - pad * 2)
    draw.rectangle([lx, ly, lx + tw + pad * 2, ly + th + pad * 2], fill=(0, 0, 0, 180))
    draw.text((lx + pad, ly + pad), label, fill=(255, 255, 255, 255), font=font)
    return out


def list_template_paths() -> list[Path]:
    """PNG templates under `assets/templates/` (may be empty)."""
    d = project_root() / "assets" / "templates"
    if not d.is_dir():
        return []
    return sorted(p for p in d.glob("*.png") if p.is_file())


def locate_template(
    screenshot: Image.Image,
    template_path: Path,
    threshold: float = 0.78,
) -> TargetBox | None:
    """
    Optional OpenCV template matching. Returns None if OpenCV missing, file missing, or no match.
    Coordinates are in the same pixel space as `screenshot`.
    """
    if not template_path.is_file():
        logger.debug("Template file missing: %s", template_path)
        return None
    try:
        import cv2
        import numpy as np
    except ImportError:
        logger.debug("OpenCV not installed; skipping template matching.")
        return None

    try:
        tpl_img = Image.open(template_path).convert("RGB")
    except OSError as exc:
        logger.warning("Could not open template %s: %s", template_path, exc)
        return None

    scr = np.array(screenshot.convert("RGB"))
    tpl = np.array(tpl_img)
    th, tw = tpl.shape[:2]
    sh, sw = scr.shape[:2]
    if tw > sw or th > sh or tw < 2 or th < 2:
        logger.debug("Template larger than screenshot or too small: %s", template_path)
        return None

    scr_gray = cv2.cvtColor(scr, cv2.COLOR_RGB2GRAY)
    tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_RGB2GRAY)
    result = cv2.matchTemplate(scr_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if max_val < threshold:
        logger.debug(
            "Template %s best match %.3f below threshold %.3f",
            template_path.name,
            max_val,
            threshold,
        )
        return None

    x1, y1 = max_loc
    x2, y2 = x1 + tw - 1, y1 + th - 1
    return TargetBox(
        label=template_path.stem,
        x1=int(x1),
        y1=int(y1),
        x2=int(x2),
        y2=int(y2),
        confidence=float(max_val),
        source="template",
        reason=f"matchTemplate({template_path.name})",
    )
