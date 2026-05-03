"""High-level orchestration: observe → plan → act loop with bounded steps and visual targeting."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import pyautogui
from PIL import Image

from app.config import SETTINGS
from app.controller import AgentController
from app.coordinates import plan_coordinates_valid_in_resized_space, scale_plan_coordinates
from app.executor import ActionExecutor
from app.llm import OpenAIClient
from app.schema import AgentPlan
from app.targeting import (
    TargetBox,
    center_of_box,
    draw_target_box,
    list_template_paths,
    locate_template,
    scale_target_box_to_capture,
)
from app.vision import ScreenCapture, create_annotated_screenshot, save_debug_image

if TYPE_CHECKING:
    from app.overlay import OverlayWindow

logger = logging.getLogger(__name__)


def _cursor_on_resized(
    cap_xy: tuple[int, int],
    original_size: tuple[int, int],
    resized_size: tuple[int, int],
) -> tuple[int, int]:
    ox, oy = cap_xy
    ow, oh = original_size
    rw, rh = resized_size
    return int(round(ox * rw / ow)), int(round(oy * rh / oh))


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
    ) -> None:
        self._llm = llm
        self._executor = executor
        self._controller = controller
        self._capture = capture or ScreenCapture()
        self._overlay = overlay
        self._last_targeting_click_center: tuple[int, int] | None = None

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
        image: Image.Image,
        original_size: tuple[int, int],
        resized: Image.Image,
        resized_size: tuple[int, int],
        step_summaries: list[str],
        status: Callable[[str], None],
    ) -> Literal["click_done", "use_planner", "stop_run"]:
        """Try template + LLM bounding box before the action planner."""
        if not SETTINGS.targeting.use_bounding_boxes:
            return "use_planner"

        if self._controller.should_stop():
            return "use_planner"

        ow, oh = original_size
        rw, rh = resized_size
        min_conf = float(SETTINGS.targeting.min_target_confidence)
        max_retries = int(SETTINGS.targeting.max_targeting_retries)

        box_capture: TargetBox | None = None

        if SETTINGS.targeting.enable_template_matching:
            status(f"Step {step}: Template matching…")
            for tpl in list_template_paths():
                if self._controller.should_stop():
                    return "use_planner"
                tb = locate_template(
                    image,
                    tpl,
                    threshold=float(SETTINGS.targeting.template_match_threshold),
                )
                if tb is None:
                    continue
                if not tb.within_bounds(ow, oh):
                    logger.warning("Template box outside capture bounds; skipping %s", tpl)
                    continue
                if tb.confidence < min_conf:
                    continue
                box_capture = tb
                logger.info("Template hit: %s conf=%.3f", tpl.name, tb.confidence)
                break

        if box_capture is None:
            ann_base = resized
            cursor_pos = None
            if SETTINGS.vision.annotated_screenshots:
                cap_xy = pyautogui.position()
                if SETTINGS.vision.draw_cursor_marker:
                    cursor_pos = _cursor_on_resized((int(cap_xy[0]), int(cap_xy[1])), original_size, resized_size)
                ann_base = create_annotated_screenshot(
                    resized,
                    cursor_position=cursor_pos,
                    grid_spacing=int(SETTINGS.vision.coordinate_grid_spacing),
                    draw_labels=SETTINGS.vision.draw_grid_labels,
                )
            else:
                ann_base = resized.copy()

            try:
                ann_b64 = self._capture.image_to_base64(ann_base)
            except ValueError as exc:
                logger.warning("Annotated image encode failed: %s", exc)
                return self._targeting_fail_no_fallback(status)

            corrective: str | None = None
            for attempt in range(max_retries + 1):
                if self._controller.should_stop():
                    return "use_planner"
                status(
                    f"Step {step}: Locating target (attempt {attempt + 1}/{max_retries + 1})…"
                )
                tb = self._llm.locate_target_box(
                    user_task=user_prompt,
                    screenshot_base64=ann_b64,
                    screenshot_width=rw,
                    screenshot_height=rh,
                    previous_summaries=step_summaries[-5:],
                    corrective=corrective,
                )
                if tb is None:
                    corrective = (
                        "The previous response was missing or invalid. "
                        f"Return strict JSON with a tight box inside {rw}x{rh} or found=false."
                    )
                    if attempt >= max_retries:
                        break
                    continue

                if not tb.within_bounds(rw, rh):
                    logger.warning("LLM box outside resized bounds: %s", tb)
                    corrective = (
                        "The previous plan had coordinates outside the screenshot bounds. "
                        f"Screenshot size is {rw} x {rh}. Return corrected JSON only."
                    )
                    if attempt >= max_retries:
                        break
                    continue

                if tb.confidence < min_conf:
                    logger.warning("LLM box below min confidence %.3f < %.3f", tb.confidence, min_conf)
                    corrective = (
                        f"Confidence was too low ({tb.confidence:.2f}). "
                        "Return a higher-confidence box or found=false."
                    )
                    if attempt >= max_retries:
                        break
                    continue

                scaled = scale_target_box_to_capture(
                    tb,
                    original_size=original_size,
                    resized_size=resized_size,
                )
                if not scaled.within_bounds(ow, oh):
                    logger.warning("Scaled box outside capture bounds")
                    corrective = "Box scaled out of range; return a smaller valid box."
                    if attempt >= max_retries:
                        break
                    continue

                box_capture = scaled
                break

        if box_capture is None:
            return self._targeting_fail_no_fallback(status)

        if self._controller.should_stop():
            return "use_planner"

        cx, cy = center_of_box(box_capture)
        min_dist = int(SETTINGS.targeting.duplicate_click_min_distance_px)
        if min_dist > 0 and self._last_targeting_click_center is not None:
            lx, ly = self._last_targeting_click_center
            dist = math.hypot(float(cx - lx), float(cy - ly))
            if dist < float(min_dist):
                logger.warning(
                    "Skipping duplicate targeting click: center (%s,%s) is %.1fpx from last (%s,%s); "
                    "falling back to action planner.",
                    cx,
                    cy,
                    dist,
                    lx,
                    ly,
                )
                status("Avoiding repeat click on same spot — using action planner…")
                return "use_planner"

        status("Executing targeted click…")
        self._executor.click(cx, cy)
        self._last_targeting_click_center = (cx, cy)

        dbg = draw_target_box(image.copy(), box_capture)
        save_debug_image(dbg, f"debug_target_step_{step:03d}")

        step_summaries.append(
            f"{box_capture.label} ({box_capture.source}, conf={box_capture.confidence:.2f})"
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
        self._last_targeting_click_center = None
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
                image=image,
                original_size=original_size,
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
