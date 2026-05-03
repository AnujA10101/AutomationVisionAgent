"""OpenAI client wrapper for producing structured action plans and target localization."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import OpenAI
from openai import APIError, APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError
from pydantic import BaseModel, ValidationError

from app.config import OPENAI_API_KEY, SETTINGS
from app.schema import Action, AgentPlan
from app.targeting import TargetBox

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
* Do NOT use press except where explicitly needed for text entry after focusing a field (prefer avoiding press entirely).
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

TARGET_LOCATOR_SYSTEM_PROMPT = """You are a visual UI target locator. You receive an annotated screenshot (grid labels) and a user task. Identify the single best visible element that should be clicked next.

Return ONLY valid JSON:

{
  "found": true,
  "label": "short target name",
  "x1": int,
  "y1": int,
  "x2": int,
  "y2": int,
  "confidence": float,
  "source": "llm_visual",
  "reason": "short reason"
}

Or when nothing suitable is visible:

{
  "found": false,
  "reason": "why no target"
}

Rules:

* Coordinates must be in the annotated screenshot coordinate system (pixels).
* Use grid labels to estimate positions.
* Return a tight bounding box around the clickable region (not just a point).
* **Multi-step tasks:** Read “Previous step summaries”. If the **next** step should be typing, scrolling, or focusing a **different** control (address bar, search field), locate **that** element — not the same control again (e.g. do not keep picking “new tab” once a tab is open if the user still needs to navigate or type).
* If the correct **next** click target is unclear or the UI already reflects the last action, return **found=false** so another subsystem can plan typing/scrolling.
* Prefer the visible clickable control that advances the **remaining** user task.
* If the target is not visible or confidence would be low, set found=false.
* Do not guess wildly — prefer found=false over a junk box.
* Do not suggest hotkeys or keyboard shortcuts. Locating targets for mouse clicking only.

Confidence must be between 0 and 1 when found=true.
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


class LocateTargetResponse(BaseModel):
    found: bool
    label: str | None = None
    x1: int | None = None
    y1: int | None = None
    x2: int | None = None
    y2: int | None = None
    confidence: float | None = None
    source: str = "llm_visual"
    reason: str | None = None


def _parse_locate_response(raw: str) -> LocateTargetResponse:
    cleaned = _strip_code_fences(raw.strip())
    data: Any = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("Locator JSON must be an object")
    return LocateTargetResponse.model_validate(data)


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

    def locate_target_box(
        self,
        *,
        user_task: str,
        screenshot_base64: str,
        screenshot_width: int,
        screenshot_height: int,
        previous_summaries: list[str] | None = None,
        corrective: str | None = None,
    ) -> TargetBox | None:
        """
        Ask the LLM for a single bounding box in annotated screenshot coordinates.

        Returns None if the model reports no suitable target or parsing/validation fails.
        """
        if not self._client:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy `.env.example` to `.env` and add your key (see README)."
            )
        if not screenshot_base64.strip():
            raise ValueError("screenshot_base64 is required.")

        model = SETTINGS.llm.model
        temperature = float(SETTINGS.llm.temperature)

        text = _build_locator_user_message_text(
            user_task=user_task,
            screenshot_width=screenshot_width,
            screenshot_height=screenshot_height,
            step_summaries=previous_summaries,
            corrective_message=corrective,
        )
        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_base64}"}},
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
            return None

        if not resp.found:
            logger.info("Locator reported found=false: %s", resp.reason)
            return None

        if (
            resp.x1 is None
            or resp.y1 is None
            or resp.x2 is None
            or resp.y2 is None
            or resp.label is None
        ):
            logger.warning("Locator missing box fields when found=true")
            return None

        conf = resp.confidence if resp.confidence is not None else 0.75

        try:
            box = TargetBox(
                label=resp.label,
                x1=int(resp.x1),
                y1=int(resp.y1),
                x2=int(resp.x2),
                y2=int(resp.y2),
                confidence=float(conf),
                source=resp.source or "llm_visual",
                reason=resp.reason,
            )
        except ValueError as exc:
            logger.warning("TargetBox validation failed: %s", exc)
            return None

        return box

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
