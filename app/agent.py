"""High-level orchestration: observe → plan → act loop with bounded steps and visual targeting."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from PIL import Image

from app.config import SETTINGS
from app.controller import AgentController
from app.coordinates import plan_coordinates_valid_in_resized_space, scale_plan_coordinates
from app.executor import ActionExecutor
from app.llm import OpenAIClient
from app.schema import AgentPlan
from app.targeting import (
    LocatedTarget,
    target_signature,
    validate_target_bounds,
)
from app.vision import (
    ScreenCapture,
    create_coordinate_map_screenshot,
    draw_located_target,
    save_debug_image,
)
from app.utils import project_root

if TYPE_CHECKING:
    from app.overlay import OverlayWindow

logger = logging.getLogger(__name__)


def _is_repeated_target(recent_targets: list[str], current_signature: str, threshold: int = 2) -> bool:
    """Return True if the same signature has been chosen repeatedly."""
    if not recent_targets:
        return False
    recent = recent_targets[-threshold:]
    return len(recent) == threshold and all(s == current_signature for s in recent)


def _is_inconsistent_targeting(recent_targets: list[str]) -> bool:
    """Detect oscillation across unrelated targets in recent selections."""
    if len(recent_targets) < 4:
        return False
    a, b, c, d = recent_targets[-4:]
    return a == c and b == d and a != b


class AutomationAgent:
    """
    Bounded observe–act loop: capture → optional visual targeting → OpenAI plan → execute.

    Stops when the model marks the task done, max steps is reached, coordinates stay invalid
    after retries, visual targeting fails without fallback, or the user requests an emergency stop.
    """

    def __init__(
        self,
        *,
        llm: OpenAIClient,
        executor: ActionExecutor,
        controller: AgentController,
        capture: ScreenCapture | None = None,
        overlay: OverlayWindow | None = None,
        confirm_target: Callable[[str], bool] | None = None,
    ) -> None:
        self._llm = llm
        self._executor = executor
        self._controller = controller
        self._capture = capture or ScreenCapture()
        self._overlay = overlay
        self._confirm_target = confirm_target
        self._recent_target_signatures: list[str] = []

    def _set_overlay_hidden_for_capture(self, hide: bool) -> None:
        if (
            self._overlay is not None
            and SETTINGS.vision.hide_overlay_during_capture
        ):
            self._overlay.visibility_for_capture.emit(hide)
            time.sleep(0.08 if hide else 0.05)

    def wait_after_action(self) -> None:
        """Pause between steps; uses `step_wait_seconds`, interruptible via controller."""
        deadline = time.monotonic() + float(SETTINGS.agent.step_wait_seconds)
        while time.monotonic() < deadline:
            if self._controller.should_stop():
                return
            time.sleep(min(0.05, deadline - time.monotonic()))

    def run_once(
        self,
        prompt: str,
        *,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        """Backward-compatible alias for the multi-step task runner."""
        self.run_task(prompt, on_status=on_status)

    def _visual_targeting_attempt(
        self,
        *,
        step: int,
        user_prompt: str,
        resized: Image.Image,
        resized_size: tuple[int, int],
        step_summaries: list[str],
        status: Callable[[str], None],
    ) -> Literal["click_done", "use_planner", "stop_run"]:
        """Coordinate-map targeting path: locate -> validate -> click point."""
        if not SETTINGS.targeting.use_bounding_boxes:
            return "use_planner"
        if self._controller.should_stop():
            return "use_planner"

        rw, rh = resized_size
        cursor_pos = None

        coord_map = create_coordinate_map_screenshot(
            resized,
            grid_cols=int(SETTINGS.vision.grid_cols),
            grid_rows=int(SETTINGS.vision.grid_rows),
            fine_grid_spacing=int(SETTINGS.vision.fine_grid_spacing),
            cursor_position=cursor_pos,
        )
        if SETTINGS.vision.save_coordinate_map_debug:
            save_debug_image(coord_map, f"debug_coordinate_map_step_{step:03d}")

        try:
            ann_b64 = self._capture.image_to_base64(coord_map)
        except ValueError as exc:
            logger.warning("Coordinate map encode failed: %s", exc)
            return self._targeting_fail_no_fallback(status)

        corrective: str | None = None
        max_retries = int(SETTINGS.targeting.max_locator_retries)
        chosen: LocatedTarget | None = None

        for attempt in range(max_retries + 1):
            if self._controller.should_stop():
                return "use_planner"
            status(f"Step {step}: Locating target on coordinate map ({attempt + 1}/{max_retries + 1})…")
            target = self._llm.locate_target_with_coordinate_map(
                user_task=user_prompt,
                annotated_screenshot_base64=ann_b64,
                screenshot_width=rw,
                screenshot_height=rh,
                step_index=step,
                previous_summaries=step_summaries[-5:],
                corrective=corrective,
            )
            if not target.found:
                corrective = "No target found. If visible, return one target with grid_cell and click point."
                if attempt >= max_retries:
                    break
                continue
            if target.confidence < float(SETTINGS.targeting.min_target_confidence):
                corrective = (
                    f"Confidence too low ({target.confidence:.2f}); return a clearly visible target or found=false."
                )
                if attempt >= max_retries:
                    break
                continue
            if SETTINGS.targeting.require_grid_cell and not target.grid_cell:
                corrective = "grid_cell is required; return grid_cell and full target JSON."
                if attempt >= max_retries:
                    break
                continue
            if SETTINGS.targeting.require_box_inside_screenshot and not validate_target_bounds(target, rw, rh):
                corrective = (
                    f"Box/click must be inside screenshot {rw}x{rh}, and click must be inside the box."
                )
                if attempt >= max_retries:
                    break
                continue
            if SETTINGS.targeting.require_click_point_inside_box and not validate_target_bounds(target, rw, rh):
                corrective = "click_x/click_y must be inside [x1,x2] and [y1,y2]."
                if attempt >= max_retries:
                    break
                continue
            chosen = target
            break

        if chosen is None:
            return self._targeting_fail_no_fallback(status)

        sig = target_signature(chosen)
        if _is_repeated_target(self._recent_target_signatures, sig):
            status("Repeated target detected with no progress; stopping safely.")
            logger.warning("Repeated target signature detected: %s", sig)
            return "stop_run"
        if _is_inconsistent_targeting(self._recent_target_signatures + [sig]):
            logger.warning("Inconsistent target oscillation detected; requesting re-evaluation.")
            return "use_planner"

        if SETTINGS.targeting.require_user_confirmation_before_click:
            status(f"Review target: {chosen.target_label} ({chosen.confidence:.2f})")
            if self._confirm_target is None:
                status("Target review mode enabled but no confirmation hook; stopping safely.")
                return "stop_run"
            if not self._confirm_target(f"{chosen.target_label} ({chosen.confidence:.2f})"):
                status("Target click cancelled by user.")
                return "stop_run"

        assert chosen.click_x is not None and chosen.click_y is not None
        status("Executing targeted click…")
        self._executor.click(chosen.click_x, chosen.click_y)
        self._recent_target_signatures.append(sig)
        self._recent_target_signatures = self._recent_target_signatures[-8:]

        output_path = project_root() / "screenshots" / f"debug_selected_target_step_{step:03d}.png"
        draw_located_target(coord_map, chosen, output_path)
        step_summaries.append(
            f"{chosen.target_label or 'target'} @ {chosen.grid_cell or '?'} (conf={chosen.confidence:.2f})"
        )
        return "click_done"

    def _targeting_fail_no_fallback(
        self,
        status: Callable[[str], None],
    ) -> Literal["use_planner", "stop_run"]:
        if SETTINGS.targeting.fallback_to_action_planner:
            logger.info("Visual targeting failed; falling back to action planner.")
            return "use_planner"
        status("Visual targeting could not find a valid target; stopping safely.")
        return "stop_run"

    def run_task(
        self,
        user_prompt: str,
        *,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        """Run up to `max_steps` iterations or until done / stop / failure."""

        def status(msg: str) -> None:
            if on_status:
                on_status(msg)

        self._controller.reset()
        self._recent_target_signatures = []
        logger.info("User task: %r", user_prompt)

        status("Running...")
        step_summaries: list[str] = []
        max_steps = int(SETTINGS.agent.max_steps)

        for step in range(1, max_steps + 1):
            if self._controller.should_stop():
                status("Stopped by user.")
                logger.info("Stopped by user before step %s.", step)
                return

            status(f"Step {step}/{max_steps}: Capturing screenshot...")
            self._set_overlay_hidden_for_capture(True)
            try:
                image = self._capture.capture()
            finally:
                self._set_overlay_hidden_for_capture(False)

            path = self._capture.save_screenshot(image)
            logger.info("Screenshot saved: %s", path)

            original_size = (image.width, image.height)
            resized = self._capture.resize_for_llm(image)
            resized_size = (resized.width, resized.height)
            rw, rh = resized_size
            logger.info(
                "Original capture size %sx%s; resized for LLM %sx%s",
                *original_size,
                *resized_size,
            )

            vt = self._visual_targeting_attempt(
                step=step,
                user_prompt=user_prompt,
                resized=resized,
                resized_size=resized_size,
                step_summaries=step_summaries,
                status=status,
            )
            if vt == "stop_run":
                return
            if vt == "click_done":
                if self._controller.should_stop():
                    status("Stopped by user.")
                    return
                self.wait_after_action()
                continue

            try:
                screenshot_b64 = self._capture.image_to_base64(resized)
            except ValueError as exc:
                logger.exception("Base64 encode failed")
                raise RuntimeError(str(exc)) from exc

            coordinate_retries_used = 0
            corrective: str | None = None
            plan: AgentPlan | None = None

            while True:
                status(
                    f"Step {step}/{max_steps}: Requesting action plan from OpenAI..."
                    + (" (retry)" if corrective else "")
                )
                plan = self._llm.get_action_plan(
                    user_task=user_prompt,
                    screenshot_base64=screenshot_b64,
                    screenshot_width=rw,
                    screenshot_height=rh,
                    step_summaries=step_summaries[-5:],
                    corrective_message=corrective,
                )

                ok_coords, coord_msg = plan_coordinates_valid_in_resized_space(plan, rw, rh)
                if ok_coords:
                    break

                logger.warning("Invalid coordinates in resized space: %s", coord_msg)
                if not SETTINGS.agent.retry_on_invalid_coordinates:
                    status("Invalid coordinates from model; stopping safely.")
                    logger.error("Coordinate validation failed (retries disabled).")
                    return

                if coordinate_retries_used >= int(SETTINGS.agent.max_coordinate_retries):
                    status("Invalid coordinates from model after retries; stopping safely.")
                    logger.error("Coordinate validation failed after retries.")
                    return

                coordinate_retries_used += 1
                corrective = (
                    "The previous plan had coordinates outside the screenshot bounds. "
                    f"Screenshot size is {rw} x {rh}. Return corrected JSON only."
                )

            assert plan is not None

            logger.info("Validated plan from model (resized space): %s", plan.model_dump(exclude_none=True))

            if plan.done:
                if plan.summary:
                    step_summaries.append(plan.summary)
                status("Done.")
                logger.info("Model reported task complete (done=true).")
                return

            max_n = min(int(SETTINGS.llm.max_actions), int(SETTINGS.agent.max_actions_per_step))
            actions_slice = plan.actions[:max_n]

            if not actions_slice:
                status("No actions from model; stopping.")
                logger.info("No actions in plan; stopping.")
                return

            if plan.summary:
                step_summaries.append(plan.summary)

            scaled_plan = scale_plan_coordinates(plan, original_size, resized_size)
            logger.info("Scaled plan (capture space): %s", scaled_plan.model_dump(exclude_none=True))

            scaled_actions = scaled_plan.actions[:max_n]

            status("Executing action...")
            self._executor.execute_plan(
                AgentPlan(done=scaled_plan.done, summary=scaled_plan.summary, actions=scaled_actions)
            )

            if self._controller.should_stop():
                status("Stopped by user.")
                logger.info("Stopped by user after executing step %s.", step)
                return

            self.wait_after_action()

        status("Max steps reached.")
        logger.info("Stopped after reaching max_steps=%s.", max_steps)
