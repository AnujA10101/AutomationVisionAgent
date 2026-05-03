"""Pydantic models describing LLM action plans."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

ActionType = Literal["move", "click", "type", "scroll", "wait", "hotkey", "press", "drag", "resize"]
MouseButton = Literal["left", "right", "middle"]


class Action(BaseModel):
    """A single UI automation step."""

    type: ActionType
    x: int | None = None
    y: int | None = None
    end_x: int | None = None
    end_y: int | None = None
    button: MouseButton | None = None
    text: str | None = None
    amount: int | None = None
    duration_seconds: float | None = None
    keys: list[str] | None = None
    key: str | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def validate_fields_for_type(self) -> Action:
        t = self.type
        if t in ("move", "click"):
            if self.x is None or self.y is None:
                raise ValueError(f'Action type "{t}" requires both x and y coordinates')
        if t in ("drag", "resize"):
            if self.x is None or self.y is None:
                raise ValueError(f'Action type "{t}" requires start x and y')
            if self.end_x is None or self.end_y is None:
                raise ValueError(f'Action type "{t}" requires end_x and end_y (drag end point)')
        if t == "type":
            if self.text is None or self.text == "":
                raise ValueError('Action type "type" requires non-empty text')
        if t == "scroll":
            if self.amount is None:
                raise ValueError('Action type "scroll" requires amount')
        if t == "wait":
            if self.duration_seconds is None:
                raise ValueError('Action type "wait" requires duration_seconds')
        if t == "hotkey":
            if not self.keys or len(self.keys) < 1:
                raise ValueError('Action type "hotkey" requires keys with at least one key')
        if t == "press":
            if self.key is None or self.key.strip() == "":
                raise ValueError('Action type "press" requires key')
        return self


class AgentPlan(BaseModel):
    """Structured plan returned by the model."""

    done: bool = False
    summary: str | None = None
    actions: list[Action] = Field(default_factory=list)
