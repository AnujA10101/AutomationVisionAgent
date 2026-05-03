"""Action / plan schema validation (no GUI or mouse automation)."""

from __future__ import annotations

import pytest

from app.llm import parse_model_response_to_plan
from app.schema import Action, AgentPlan


def test_valid_hotkey_action() -> None:
    a = Action(type="hotkey", keys=["command", "t"], reason="new tab")
    assert a.keys == ["command", "t"]


def test_invalid_hotkey_empty_keys() -> None:
    with pytest.raises(ValueError, match="hotkey"):
        Action(type="hotkey", keys=[])


def test_valid_press_action() -> None:
    a = Action(type="press", key="enter")
    assert a.key == "enter"


def test_invalid_press_missing_key() -> None:
    with pytest.raises(ValueError, match="press"):
        Action(type="press", key="")


def test_agent_plan_done_true_empty_actions_parse() -> None:
    raw = '{"done":true,"summary":"Finished loading page","actions":[]}'
    plan = parse_model_response_to_plan(raw)
    assert plan.done is True
    assert plan.actions == []
    assert plan.summary == "Finished loading page"


def test_agent_plan_defaults() -> None:
    plan = AgentPlan(actions=[Action(type="wait", duration_seconds=1.0)])
    assert plan.done is False


def test_valid_drag_action() -> None:
    a = Action(
        type="drag",
        x=10,
        y=20,
        end_x=100,
        end_y=80,
        button="left",
        duration_seconds=0.4,
    )
    assert a.end_x == 100


def test_drag_missing_end_fails() -> None:
    with pytest.raises(ValueError, match="end_x"):
        Action(type="drag", x=0, y=0)


def test_resize_same_as_drag_shape() -> None:
    a = Action(type="resize", x=5, y=5, end_x=200, end_y=5, reason="widen")
    assert a.type == "resize"
