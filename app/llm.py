"""OpenAI client wrapper for producing structured action plans and target localization."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import OpenAI
from openai import APIError, APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError
from pydantic import ValidationError

from app.config import OPENAI_API_KEY, SETTINGS
from app.schema import Action, AgentPlan
from app.targeting import LocatedTarget

logger = logging.getLogger(__name__)

VISION_PLANNER_SYSTEM_PROMPT = """You are a desktop automation planner. You receive a screenshot and a user task. Return the next small safe action plan for the current screen.

Return ONLY valid JSON matching this schema:

{
  "done": false,
  "summary": "brief description",
  "actions": [
    {
      "type": "move" | "click" | "type" | "scroll" | "wait" | "hotkey" | "press" | "drag" | "resize",
      "x": int | null,
      "y": int | null,
      "end_x": int | null,
      "end_y": int | null,
      "button": "left" | "right" | "middle" | null,
      "text": string | null,
      "amount": int | null,
      "duration_seconds": number | null,
      "keys": array[string] | null,
      "key": string | null,
      "reason": string | null
    }
  ]
}

Rules:

* Return only JSON. No markdown.
* This is one step in a loop, not the whole task.
* Use at most 3 actions.

Cursor-only control:

* Do NOT use hotkey or keyboard shortcuts (no Command/Control/Alt combinations).
* Do NOT use press actions. Use only move, click, type, scroll, wait, drag, or resize.
* Prefer **move** and **click** using coordinates; targets should align with prior bounding-box grounding when applicable.
* **drag** — press at (x, y), hold `button` (default left), move to (end_x, end_y), release. Use for sliders, selection marquees, dragging files, or **resize** (use type **resize** for the same fields when adjusting window/edge/corner size so the model can distinguish intent). Optional **duration_seconds** is the time to perform the drag motion (not used for **wait**).
* **resize** — same fields as **drag**; pick the handle (edge or corner) as the start and drag to the desired end.
* To open a new browser tab, locate and **click** the visible new-tab (+) control — do not use Ctrl/Cmd+T.
* For typing: first **click** to focus the field, then **type** text. Use **scroll** and **wait** as needed.

Safety:

* Coordinates must be within the screenshot dimensions (0 .. width-1 and 0 .. height-1) for all used points, including **end_x** / **end_y** for drag/resize.
* If the task is complete on the current screen, return done=true and actions=[].
* If you are unsure, return a wait action instead of guessing.
* Do not perform destructive actions.
* Do not purchase, delete, submit private forms, send messages, or confirm irreversible actions.
* Do not answer or complete graded exams, quizzes, or assessments.
"""

TARGET_LOCATOR_SYSTEM_PROMPT = """You are a visual UI target locator for a cursor-only desktop automation app.

You receive an annotated screenshot with:
* labeled grid cells like A1, B1, C1
* pixel x/y labels
* screenshot dimensions

Your job is to locate the next visible target that the cursor should click.

Return ONLY valid JSON:

{
  "found": true,
  "target_label": "short name of target",
  "grid_cell": "cell label like B3",
  "x1": int,
  "y1": int,
  "x2": int,
  "y2": int,
  "click_x": int,
  "click_y": int,
  "confidence": float,
  "source": "llm_coordinate_map",
  "reason": "brief reason"
}

If target is not visible, return:

{
  "found": false,
  "confidence": 0,
  "source": "llm_coordinate_map",
  "reason": "target not visible"
}

Rules:

* Coordinates must be in the annotated screenshot coordinate system.
* Use the visible grid and pixel labels to estimate coordinates.
* Return a bounding box around the clickable target.
* Return a click point inside the bounding box.
* The click point should usually be the center of the clickable target.
* Do not invent targets that are not visible.
* Do not return high confidence unless the target is clearly visible.
* If unsure between multiple targets, choose the safest obvious target or return found=false.
* Do not use hotkeys.
* Do not use keyboard shortcuts.
* Do not suggest keyboard commands.
* The app can only move the cursor, click, type into focused fields, scroll, and wait.

For browser navigation:
* To open a new tab, locate the visible + new tab button.
* To type a URL, locate the visible address/search bar first.
* To click a page result or visible link, locate the visible link text or button.
* If the user asks to type text, the target should be the text field where typing should happen.
"""


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if not t.startswith("```"):
        return t
    lines = t.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_plan_json(raw: str) -> AgentPlan:
    """Parse strict JSON into an AgentPlan (raises on invalid input)."""
    data: Any = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Top-level JSON must be an object")
    return AgentPlan.model_validate(data)


def parse_model_response_to_plan(content: str) -> AgentPlan:
    """Strip fences if present, parse JSON, validate into AgentPlan."""
    cleaned = _strip_code_fences(content.strip())
    return parse_plan_json(cleaned)


def _parse_locate_response(raw: str) -> LocatedTarget:
    cleaned = _strip_code_fences(raw.strip())
    data: Any = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("Locator JSON must be an object")
    return LocatedTarget.model_validate(data)


def _build_user_message_text(
    *,
    user_task: str,
    screenshot_width: int,
    screenshot_height: int,
    step_summaries: list[str] | None,
    corrective_message: str | None,
) -> str:
    lines: list[str] = [
        f"Screenshot size: {screenshot_width} x {screenshot_height}",
        "",
        f"User task: {user_task}",
        "",
        "Previous step summaries:",
    ]
    recent = (step_summaries or [])[-5:]
    if recent:
        for s in recent:
            lines.append(f"- {s}")
    else:
        lines.append("(none)")

    if corrective_message:
        lines.extend(["", corrective_message])

    lines.extend(["", "Current screenshot:"])
    return "\n".join(lines)


def _build_locator_user_message_text(
    *,
    user_task: str,
    screenshot_width: int,
    screenshot_height: int,
    step_summaries: list[str] | None,
    corrective_message: str | None,
) -> str:
    lines: list[str] = [
        "This screenshot includes grid lines and axis labels for coordinate reference.",
        f"Annotated screenshot size: {screenshot_width} x {screenshot_height}",
        "",
        f"User task: {user_task}",
        "",
        "Previous step summaries:",
    ]
    recent = (step_summaries or [])[-5:]
    if recent:
        for s in recent:
            lines.append(f"- {s}")
    else:
        lines.append("(none)")
    if corrective_message:
        lines.extend(["", corrective_message])
    lines.extend(["", "Annotated screenshot:"])
    return "\n".join(lines)


def filter_disallowed_actions(plan: AgentPlan) -> AgentPlan:
    """Remove hotkey/press when disabled in settings; insert wait if nothing executable remains."""
    allow_h = bool(SETTINGS.actions.allow_hotkeys)
    allow_p = bool(SETTINGS.actions.allow_press)
    kept: list[Action] = []
    removed = False
    for a in plan.actions:
        if a.type == "hotkey" and not allow_h:
            logger.warning("Removing disallowed hotkey action from plan (actions.allow_hotkeys=false).")
            removed = True
            continue
        if a.type == "press" and not allow_p:
            logger.warning("Removing disallowed press action from plan (actions.allow_press=false).")
            removed = True
            continue
        kept.append(a)

    if removed and not kept and not plan.done:
        kept = [
            Action(
                type="wait",
                duration_seconds=0.5,
                reason="Unsupported actions filtered; waiting safely.",
            )
        ]

    return plan.model_copy(update={"actions": kept})


class OpenAIClient:
    """Thin wrapper around the OpenAI SDK for vision + JSON action plans."""

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key if api_key is not None else OPENAI_API_KEY
        self._api_key = key.strip() if isinstance(key, str) else None
        self._client: OpenAI | None
        if self._api_key:
            self._client = OpenAI(api_key=self._api_key)
        else:
            self._client = None

    def locate_target_with_coordinate_map(
        self,
        *,
        user_task: str,
        annotated_screenshot_base64: str,
        screenshot_width: int,
        screenshot_height: int,
        step_index: int,
        previous_summaries: list[str] | None = None,
        corrective: str | None = None,
    ) -> LocatedTarget:
        """
        Ask the LLM to locate next click target using coordinate-map annotated screenshot.
        """
        if not self._client:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy `.env.example` to `.env` and add your key (see README)."
            )
        if not annotated_screenshot_base64.strip():
            raise ValueError("annotated_screenshot_base64 is required.")

        model = SETTINGS.llm.model
        temperature = float(SETTINGS.llm.temperature)

        text = _build_locator_user_message_text(
            user_task=user_task,
            screenshot_width=screenshot_width,
            screenshot_height=screenshot_height,
            step_summaries=(previous_summaries or []) + [f"Step index: {step_index}"],
            corrective_message=corrective,
        )
        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{annotated_screenshot_base64}"}},
        ]

        logger.info("Calling OpenAI target locator model=%s", model)

        try:
            completion = self._client.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": TARGET_LOCATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
        except AuthenticationError as exc:
            logger.exception("OpenAI authentication failed")
            raise RuntimeError("OpenAI authentication failed. Check OPENAI_API_KEY.") from exc
        except RateLimitError as exc:
            logger.exception("OpenAI rate limit")
            raise RuntimeError("OpenAI rate limit exceeded. Try again later.") from exc
        except APIConnectionError as exc:
            logger.exception("OpenAI connection error")
            raise RuntimeError(f"OpenAI connection error: {exc}") from exc
        except APITimeoutError as exc:
            logger.exception("OpenAI request timed out")
            raise RuntimeError("OpenAI request timed out.") from exc
        except APIError as exc:
            logger.exception("OpenAI API error")
            raise RuntimeError(f"OpenAI API error: {exc}") from exc

        raw = completion.choices[0].message.content or ""
        logger.info("Raw locator response (truncated): %s", raw[:2000])

        try:
            resp = _parse_locate_response(raw)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning("Locator JSON invalid: %s", exc)
            return LocatedTarget(
                found=False,
                confidence=0.0,
                source="llm_coordinate_map",
                reason=f"invalid locator JSON: {exc}",
            )
        if not resp.found:
            return resp.model_copy(update={"confidence": 0.0 if resp.confidence is None else resp.confidence})
        return resp

    def get_action_plan(
        self,
        *,
        user_task: str,
        screenshot_base64: str,
        screenshot_width: int,
        screenshot_height: int,
        step_summaries: list[str] | None = None,
        corrective_message: str | None = None,
    ) -> AgentPlan:
        """
        Request a vision-grounded JSON plan from OpenAI.

        Raises:
            RuntimeError: Missing API key, or wrapped API failure.
            ValueError: Invalid JSON or Pydantic validation failure.
        """
        if not self._client:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy `.env.example` to `.env` and add your key (see README)."
            )

        if screenshot_base64 is None or not screenshot_base64.strip():
            raise ValueError("screenshot_base64 is required for vision planning.")

        model = SETTINGS.llm.model
        temperature = float(SETTINGS.llm.temperature)
        max_actions = int(SETTINGS.llm.max_actions)

        text = _build_user_message_text(
            user_task=user_task,
            screenshot_width=screenshot_width,
            screenshot_height=screenshot_height,
            step_summaries=step_summaries,
            corrective_message=corrective_message,
        )

        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{screenshot_base64}"},
            },
        ]

        logger.info("Calling OpenAI model=%s max_actions=%s", model, max_actions)

        try:
            completion = self._client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": VISION_PLANNER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
        except AuthenticationError as exc:
            logger.exception("OpenAI authentication failed")
            raise RuntimeError("OpenAI authentication failed. Check OPENAI_API_KEY.") from exc
        except RateLimitError as exc:
            logger.exception("OpenAI rate limit")
            raise RuntimeError("OpenAI rate limit exceeded. Try again later.") from exc
        except APIConnectionError as exc:
            logger.exception("OpenAI connection error")
            raise RuntimeError(f"OpenAI connection error: {exc}") from exc
        except APITimeoutError as exc:
            logger.exception("OpenAI request timed out")
            raise RuntimeError("OpenAI request timed out.") from exc
        except APIError as exc:
            logger.exception("OpenAI API error")
            raise RuntimeError(f"OpenAI API error: {exc}") from exc

        raw = completion.choices[0].message.content or ""
        logger.info("Raw model response (truncated): %s", raw[:4000])

        try:
            plan = parse_model_response_to_plan(raw)
        except json.JSONDecodeError as exc:
            snippet = re.sub(r"\s+", " ", raw).strip()
            if len(snippet) > 500:
                snippet = snippet[:500] + "…"
            logger.exception("Invalid JSON from model; snippet=%r", snippet)
            raise ValueError(f"Model returned invalid JSON: {exc}") from exc
        except ValidationError as exc:
            logger.exception("Plan failed Pydantic validation")
            raise ValueError(f"Plan validation failed: {exc}") from exc

        plan = filter_disallowed_actions(plan)

        trimmed = AgentPlan(
            done=plan.done,
            summary=plan.summary,
            actions=plan.actions[:max_actions],
        )

        if trimmed.done and not trimmed.actions:
            logger.info("Validated terminal plan: done=true actions=[]")
            return trimmed

        if not trimmed.actions:
            logger.info("Validated plan with no actions (done=%s)", trimmed.done)
            return trimmed

        logger.info("Validated plan: %s", trimmed.model_dump(exclude_none=True))
        return trimmed
