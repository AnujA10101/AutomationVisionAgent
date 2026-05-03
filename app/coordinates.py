"""Map LLM coordinates from resized screenshots back to full capture pixel space."""

from __future__ import annotations

import logging

from app.schema import Action, AgentPlan

logger = logging.getLogger(__name__)


def scale_xy(
    x: int,
    y: int,
    original_size: tuple[int, int],
    resized_size: tuple[int, int],
) -> tuple[int, int]:
    """Scale a point from resized image coordinates to original image coordinates."""
    ow, oh = original_size
    rw, rh = resized_size
    if rw <= 0 or rh <= 0:
        raise ValueError(f"Invalid resized dimensions: {resized_size}")
    rx = int(round(x * ow / rw))
    ry = int(round(y * oh / rh))
    return rx, ry


def clamp_xy(x: int, y: int, width: int, height: int) -> tuple[int, int]:
    """Clamp coordinates to lie within [0, width-1] and [0, height-1]."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid bounds: {width}x{height}")
    cx = max(0, min(width - 1, x))
    cy = max(0, min(height - 1, y))
    if (cx, cy) != (x, y):
        logger.warning("Clamped out-of-bounds coordinates (%s, %s) -> (%s, %s)", x, y, cx, cy)
    return cx, cy


def plan_coordinates_valid_in_resized_space(
    plan: AgentPlan,
    resized_width: int,
    resized_height: int,
) -> tuple[bool, str | None]:
    """
    Return whether all move/click coordinates lie within the resized screenshot bounds.

    Coordinates are validated before scaling; invalid plans must not be executed.
    """
    if resized_width <= 0 or resized_height <= 0:
        return False, f"invalid resized dimensions {resized_width}x{resized_height}"

    for action in plan.actions:
        if action.type in ("move", "click"):
            if action.x is None or action.y is None:
                return False, f"{action.type} action missing x/y"
            if not (0 <= action.x < resized_width and 0 <= action.y < resized_height):
                return (
                    False,
                    f"({action.x}, {action.y}) outside screenshot bounds {resized_width}x{resized_height}",
                )
        if action.type in ("drag", "resize"):
            if action.x is None or action.y is None or action.end_x is None or action.end_y is None:
                return False, f"{action.type} action missing start or end coordinates"
            for label, px, py in (
                ("start", action.x, action.y),
                ("end", action.end_x, action.end_y),
            ):
                if not (0 <= px < resized_width and 0 <= py < resized_height):
                    return (
                        False,
                        f"({px}, {py}) {label} outside screenshot bounds {resized_width}x{resized_height}",
                    )
    return True, None


def scale_action_coordinates(
    action: Action,
    original_size: tuple[int, int],
    resized_size: tuple[int, int],
) -> Action:
    """Scale x/y (and end_x/end_y for drag/resize) from resized space to capture space."""
    if action.type in ("move", "click"):
        if action.x is None or action.y is None:
            return action
        sx, sy = scale_xy(action.x, action.y, original_size, resized_size)
        return action.model_copy(update={"x": sx, "y": sy})
    if action.type in ("drag", "resize"):
        if (
            action.x is None
            or action.y is None
            or action.end_x is None
            or action.end_y is None
        ):
            return action
        sx1, sy1 = scale_xy(action.x, action.y, original_size, resized_size)
        sx2, sy2 = scale_xy(action.end_x, action.end_y, original_size, resized_size)
        return action.model_copy(update={"x": sx1, "y": sy1, "end_x": sx2, "end_y": sy2})
    return action


def scale_plan_coordinates(
    plan: AgentPlan,
    original_size: tuple[int, int],
    resized_size: tuple[int, int],
) -> AgentPlan:
    """Return a new plan with all move/click coordinates mapped to original screenshot space."""
    scaled = [scale_action_coordinates(a, original_size, resized_size) for a in plan.actions]
    return plan.model_copy(update={"actions": scaled})
