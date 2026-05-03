"""Coordinate scaling and bounds validation (no mouse automation)."""

from __future__ import annotations

import pytest

from app.coordinates import (
    clamp_xy,
    plan_coordinates_valid_in_resized_space,
    scale_action_coordinates,
    scale_plan_coordinates,
    scale_xy,
)
from app.schema import Action, AgentPlan


def test_scale_xy_example_from_spec() -> None:
    """3000x2000 original, 1500x1000 resized -> (750,500) maps to (1500,1000)."""
    ox, oy = scale_xy(750, 500, (3000, 2000), (1500, 1000))
    assert ox == 1500
    assert oy == 1000


def test_clamp_xy_negative_and_overflow() -> None:
    x, y = clamp_xy(-5, 3000, 800, 600)
    assert x == 0
    assert y == 599


def test_scale_xy_invalid_resized_raises() -> None:
    with pytest.raises(ValueError, match="Invalid resized"):
        scale_xy(1, 1, (100, 100), (0, 50))


def test_scale_plan_moves_click_coordinates() -> None:
    plan = AgentPlan(
        actions=[
            Action(type="move", x=100, y=50, reason="t"),
            Action(type="wait", duration_seconds=0.1),
            Action(type="click", x=100, y=50, button="left", reason="c"),
        ]
    )
    orig = (2000, 1000)
    resized = (1000, 500)
    scaled = scale_plan_coordinates(plan, orig, resized)
    assert scaled.actions[0].x == 200
    assert scaled.actions[0].y == 100
    assert scaled.actions[2].x == 200
    assert scaled.actions[2].y == 100


def test_coordinate_validation_rejects_x_greater_than_resized_width() -> None:
    """Model returns x beyond resized width — reject before scaling (no blind clamp)."""
    plan = AgentPlan(
        actions=[
            Action(type="click", x=1453, y=100, button="left"),
        ]
    )
    ok, msg = plan_coordinates_valid_in_resized_space(plan, 1280, 831)
    assert ok is False
    assert msg is not None
    assert "1453" in msg or "outside" in msg


def test_coordinate_validation_accepts_valid_xy() -> None:
    plan = AgentPlan(
        actions=[
            Action(type="click", x=1279, y=830, button="left"),
        ]
    )
    ok, _msg = plan_coordinates_valid_in_resized_space(plan, 1280, 831)
    assert ok is True


def test_scale_valid_coordinates_no_clamping() -> None:
    """Valid resized coords scale proportionally to original capture space."""
    plan = AgentPlan(actions=[Action(type="click", x=640, y=450, button="left")])
    scaled = scale_plan_coordinates(plan, (2000, 1500), (1280, 900))
    assert scaled.actions[0].x == 1000
    assert scaled.actions[0].y == 750


def test_scale_action_preserves_non_spatial_types() -> None:
    plan = AgentPlan(actions=[Action(type="hotkey", keys=["command", "t"])])
    out = scale_plan_coordinates(plan, (1920, 1080), (1280, 720))
    assert out.actions[0].keys == ["command", "t"]


def test_drag_endpoints_validated_in_resized_space() -> None:
    plan = AgentPlan(
        actions=[
            Action(
                type="drag",
                x=10,
                y=10,
                end_x=2000,
                end_y=10,
            )
        ]
    )
    ok, _msg = plan_coordinates_valid_in_resized_space(plan, 1280, 831)
    assert ok is False


def test_scale_plan_drags_both_endpoints() -> None:
    plan = AgentPlan(
        actions=[
            Action(type="drag", x=100, y=50, end_x=200, end_y=100, reason="d"),
        ]
    )
    orig = (2000, 1000)
    resized = (1000, 500)
    scaled = scale_plan_coordinates(plan, orig, resized)
    assert scaled.actions[0].x == 200
    assert scaled.actions[0].y == 100
    assert scaled.actions[0].end_x == 400
    assert scaled.actions[0].end_y == 200
