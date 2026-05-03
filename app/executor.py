"""Execute structured actions with pyautogui."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import pyautogui

from app.config import SETTINGS
from app.controller import AgentController
from app.schema import Action, ActionType, AgentPlan

if TYPE_CHECKING:
    from app.overlay import OverlayWindow

logger = logging.getLogger(__name__)


def _is_risky(action_type: ActionType) -> bool:
    return action_type in ("click", "type", "drag", "resize")


class ActionExecutor:
    """Runs `Action` / `AgentPlan` using pyautogui with pacing and logging."""

    _MOVE_STEPS = 18
    _MOVE_DURATION_SEC = 0.55
    _DEFAULT_DRAG_DURATION_SEC = 0.5

    def __init__(
        self,
        overlay: OverlayWindow | None = None,
        controller: AgentController | None = None,
    ) -> None:
        self._overlay = overlay
        self._controller = controller
        pyautogui.FAILSAFE = bool(SETTINGS.safety.pyautogui_failsafe)

    def _delay_between_actions(self) -> None:
        self._interruptible_sleep(float(SETTINGS.agent.action_delay_seconds))

    def _interruptible_sleep(self, seconds: float) -> None:
        if seconds <= 0:
            return
        if self._controller is None:
            time.sleep(seconds)
            return
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if self._controller.should_stop():
                return
            time.sleep(min(0.05, end - time.monotonic()))

    def _maybe_skip_risky(self, action: Action) -> bool:
        if not SETTINGS.safety.require_confirmation_for_risky_actions:
            return False
        if _is_risky(action.type):
            logger.warning(
                "Skipping risky action %r (require_confirmation_for_risky_actions=true; confirmation UI not implemented).",
                action.type,
            )
            return True
        return False

    def _check_stop_before_action(self) -> bool:
        if self._controller is not None and self._controller.should_stop():
            logger.info("Stop requested before action; skipping remaining automation.")
            return True
        return False

    def _check_stop_after_action(self) -> bool:
        if self._controller is not None and self._controller.should_stop():
            logger.info("Stop requested after action; halting plan execution.")
            return True
        return False

    def _overlay_cursor(self, x: int, y: int) -> None:
        """Update the on-screen ring; uses Qt signals (safe from the automation worker thread)."""
        if self._overlay is None:
            return
        self._overlay.cursor_update_requested.emit(int(x), int(y))

    def _overlay_click_effect(self, x: int, y: int) -> None:
        if self._overlay is None:
            return
        self._overlay.click_effect_requested.emit(int(x), int(y))

    def _overlay_ping_at_mouse(self) -> None:
        """When the plan has no move/click (e.g. only hotkey), show the ring at the real cursor."""
        if self._overlay is None:
            return
        x, y = pyautogui.position()
        self._overlay_cursor(int(x), int(y))

    def move_smooth(self, x: int, y: int, duration: float | None = None) -> None:
        """
        Move the mouse along a smooth path, updating the optional overlay each step.

        Uses short linear segments with an ease-in-out progress curve.
        """
        if self._controller is not None and self._controller.should_stop():
            return

        target_x, target_y = int(x), int(y)
        start_x, start_y = pyautogui.position()
        if start_x == target_x and start_y == target_y:
            self._overlay_cursor(target_x, target_y)
            return

        total = float(self._MOVE_DURATION_SEC if duration is None else duration)
        steps = max(2, self._MOVE_STEPS)
        step_sleep = total / float(steps)

        for i in range(1, steps + 1):
            if self._controller is not None and self._controller.should_stop():
                logger.info("Stop requested during smooth move; aborting motion.")
                return
            t = pyautogui.easeInOutQuad(i / steps)
            nx = int(round(start_x + (target_x - start_x) * t))
            ny = int(round(start_y + (target_y - start_y) * t))
            pyautogui.moveTo(nx, ny, duration=0)
            self._overlay_cursor(nx, ny)
            time.sleep(step_sleep)

    def click(self, x: int, y: int, button: str = "left") -> None:
        """
        Move to the target, show a click effect on the overlay, then perform a real click.
        """
        self.move_smooth(x, y)
        if self._controller is not None and self._controller.should_stop():
            return
        self._overlay_click_effect(x, y)
        pyautogui.click(button=button)

    def _drag_path_smooth(self, target_x: int, target_y: int, duration: float) -> None:
        """Move pointer along an eased path while a mouse button is held (after mouseDown)."""
        tx, ty = int(target_x), int(target_y)
        start_x, start_y = pyautogui.position()
        if start_x == tx and start_y == ty:
            self._overlay_cursor(tx, ty)
            return

        steps = max(2, self._MOVE_STEPS)
        step_sleep = max(0.001, float(duration) / float(steps))

        for i in range(1, steps + 1):
            if self._controller is not None and self._controller.should_stop():
                logger.info("Stop requested during drag; releasing button.")
                break
            t = pyautogui.easeInOutQuad(i / steps)
            nx = int(round(start_x + (tx - start_x) * t))
            ny = int(round(start_y + (ty - start_y) * t))
            pyautogui.moveTo(nx, ny, duration=0)
            self._overlay_cursor(nx, ny)
            time.sleep(step_sleep)

    def drag(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        *,
        button: str = "left",
        duration_seconds: float | None = None,
    ) -> None:
        """
        Click-drag from start to end (window resize handles, sliders, selection rectangles).

        Holds `button` down, moves along a smooth path to (end_x, end_y), then releases.
        """
        self.move_smooth(start_x, start_y)
        if self._controller is not None and self._controller.should_stop():
            return

        btn = button
        dur = (
            float(duration_seconds)
            if duration_seconds is not None and duration_seconds > 0
            else self._DEFAULT_DRAG_DURATION_SEC
        )

        pyautogui.mouseDown(button=btn)
        try:
            self._drag_path_smooth(end_x, end_y, dur)
        finally:
            pyautogui.mouseUp(button=btn)

    def type_text(self, text: str) -> None:
        """Type Unicode text using the keyboard."""
        self._overlay_ping_at_mouse()
        pyautogui.write(text, interval=0.02)

    def scroll(self, amount: int) -> None:
        """Scroll vertically by `amount` (platform-dependent units)."""
        self._overlay_ping_at_mouse()
        pyautogui.scroll(int(amount))

    def hotkey(self, keys: list[str]) -> None:
        """Press a combination of keys (e.g. command+t on macOS)."""
        self._overlay_ping_at_mouse()
        pyautogui.hotkey(*keys)

    def press(self, key: str) -> None:
        """Press and release a single key."""
        self._overlay_ping_at_mouse()
        pyautogui.press(key)

    def wait(self, duration_seconds: float) -> None:
        """Sleep for the given duration (interruptible when a controller is set)."""
        self._interruptible_sleep(float(duration_seconds))

    def execute_action(self, action: Action) -> bool:
        """Execute a single validated action. Returns False if skipped (e.g. pending confirmation)."""
        if self._check_stop_before_action():
            return False

        if self._maybe_skip_risky(action):
            return False

        logger.info("Executing action: %s", action.model_dump(exclude_none=True))

        t = action.type
        if t == "move":
            assert action.x is not None and action.y is not None
            self.move_smooth(action.x, action.y)
        elif t == "click":
            assert action.x is not None and action.y is not None
            btn = action.button or "left"
            self.click(action.x, action.y, button=btn)
        elif t in ("drag", "resize"):
            assert (
                action.x is not None
                and action.y is not None
                and action.end_x is not None
                and action.end_y is not None
            )
            btn = action.button or "left"
            self.drag(
                action.x,
                action.y,
                action.end_x,
                action.end_y,
                button=btn,
                duration_seconds=action.duration_seconds,
            )
        elif t == "type":
            assert action.text is not None
            self.type_text(action.text)
        elif t == "scroll":
            assert action.amount is not None
            self.scroll(action.amount)
        elif t == "wait":
            assert action.duration_seconds is not None
            self.wait(action.duration_seconds)
        elif t == "hotkey":
            if not SETTINGS.actions.allow_hotkeys:
                logger.error(
                    "Rejected hotkey action (actions.allow_hotkeys=false): %s",
                    action.keys,
                )
                return False
            assert action.keys is not None
            self.hotkey(action.keys)
        elif t == "press":
            if not SETTINGS.actions.allow_press:
                logger.error(
                    "Rejected press action (actions.allow_press=false): %s",
                    action.key,
                )
                return False
            assert action.key is not None
            self.press(action.key)
        else:
            logger.warning("Unknown action type %r; skipping", t)
            return False

        if self._check_stop_after_action():
            return False
        return True

    def execute_plan(self, plan: AgentPlan) -> None:
        """Execute each action in order with configured delays."""
        max_n = int(SETTINGS.agent.max_actions_per_step)
        actions = plan.actions[:max_n]
        for i, action in enumerate(actions):
            ran = self.execute_action(action)
            if not ran:
                logger.warning("Action skipped or stopped: %s", action.type)
            if self._controller is not None and self._controller.should_stop():
                break
            if i < len(actions) - 1:
                self._delay_between_actions()
                if self._controller is not None and self._controller.should_stop():
                    break
