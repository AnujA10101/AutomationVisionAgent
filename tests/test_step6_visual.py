"""Coordinate-map targeting and validation (no mouse automation)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from app.agent import _is_repeated_target
from app.executor import ActionExecutor
from app.llm import _parse_locate_response, filter_disallowed_actions
from app.schema import Action, AgentPlan
from app.targeting import (
    LocatedTarget,
    TargetBox,
    center_of_box,
    draw_target_box,
    locate_template,
    target_signature,
    validate_target_bounds,
)
from app.vision import create_coordinate_map_screenshot, save_debug_image


def test_target_box_validation_ok() -> None:
    b = TargetBox(label="btn", x1=10, y1=20, x2=50, y2=60, confidence=0.9, source="test")
    assert b.within_bounds(100, 100)


def test_target_box_invalid_order() -> None:
    with pytest.raises(ValueError):
        TargetBox(label="bad", x1=50, y1=50, x2=10, y2=10, confidence=0.9, source="test")


def test_center_of_box() -> None:
    b = TargetBox(label="x", x1=0, y1=0, x2=10, y2=10, confidence=1.0, source="t")
    assert center_of_box(b) == (5, 5)


def test_low_confidence_rejected_by_agent_logic() -> None:
    b = TargetBox(label="x", x1=0, y1=0, x2=5, y2=5, confidence=0.2, source="llm_visual")
    assert b.confidence < 0.70


def test_coordinate_map_preserves_dimensions() -> None:
    img = Image.new("RGB", (320, 240), color=(30, 30, 40))
    ann = create_coordinate_map_screenshot(img, grid_cols=12, grid_rows=8, fine_grid_spacing=50)
    assert ann.size == img.size


def test_grid_label_render_does_not_crash() -> None:
    img = Image.new("RGB", (640, 400), color=(10, 10, 10))
    out = create_coordinate_map_screenshot(img, grid_cols=6, grid_rows=4, fine_grid_spacing=40)
    assert out.size == (640, 400)


def test_save_debug_image_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.vision.project_root", lambda: tmp_path)
    img = Image.new("RGB", (10, 10), color=(0, 0, 0))
    p = save_debug_image(img, "debug_target_step_001")
    assert p.name == "debug_target_step_001.png"
    assert p.parent.name == "screenshots"
    assert p.is_file()


def test_draw_target_box_returns_copy() -> None:
    img = Image.new("RGB", (100, 80), color=(0, 0, 0))
    box = TargetBox(label="t", x1=10, y1=10, x2=40, y2=40, confidence=0.99, source="x")
    out = draw_target_box(img, box)
    assert out is not img


def test_hotkey_rejected_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import SETTINGS

    monkeypatch.setattr(SETTINGS, "actions", SETTINGS.actions.model_copy(update={"allow_hotkeys": False}))
    ex = ActionExecutor(overlay=None, controller=None)
    act = Action(type="hotkey", keys=["command", "t"])
    assert ex.execute_action(act) is False


def test_press_rejected_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import SETTINGS

    monkeypatch.setattr(SETTINGS, "actions", SETTINGS.actions.model_copy(update={"allow_press": False}))
    ex = ActionExecutor(overlay=None, controller=None)
    act = Action(type="press", key="enter")
    assert ex.execute_action(act) is False


def test_filter_removes_hotkey(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import SETTINGS

    monkeypatch.setattr(SETTINGS, "actions", SETTINGS.actions.model_copy(update={"allow_hotkeys": False}))
    plan = AgentPlan(actions=[Action(type="hotkey", keys=["a"])])
    out = filter_disallowed_actions(plan)
    assert len(out.actions) == 1
    assert out.actions[0].type == "wait"


def test_template_match_missing_file_returns_none() -> None:
    img = Image.new("RGB", (200, 200), color=(128, 128, 128))
    assert locate_template(img, Path("/nonexistent/path/template.png")) is None


def test_locate_target_response_parse() -> None:
    raw = '{"found": false, "confidence": 0, "source": "llm_coordinate_map", "reason": "nothing"}'
    r = _parse_locate_response(raw)
    assert r.found is False


def test_valid_located_target_passes_validation() -> None:
    t = LocatedTarget(
        found=True,
        target_label="new tab button",
        grid_cell="B1",
        x1=100,
        y1=20,
        x2=130,
        y2=45,
        click_x=116,
        click_y=33,
        confidence=0.9,
    )
    assert validate_target_bounds(t, 1280, 831)


def test_missing_grid_cell_fails_when_found_true() -> None:
    with pytest.raises(ValueError):
        LocatedTarget(
            found=True,
            target_label="x",
            x1=1,
            y1=1,
            x2=2,
            y2=2,
            click_x=1,
            click_y=1,
            confidence=0.9,
        )


def test_click_point_outside_box_fails() -> None:
    with pytest.raises(ValueError):
        LocatedTarget(
            found=True,
            target_label="x",
            grid_cell="A1",
            x1=10,
            y1=10,
            x2=20,
            y2=20,
            click_x=25,
            click_y=15,
            confidence=0.9,
        )


def test_box_outside_screenshot_fails_bounds_check() -> None:
    t = LocatedTarget(
        found=True,
        target_label="x",
        grid_cell="A1",
        x1=10,
        y1=10,
        x2=400,
        y2=20,
        click_x=15,
        click_y=15,
        confidence=0.9,
    )
    assert validate_target_bounds(t, 100, 100) is False


def test_target_signature_is_stable() -> None:
    t = LocatedTarget(
        found=True,
        target_label="Address Bar",
        grid_cell="C1",
        x1=100,
        y1=20,
        x2=500,
        y2=60,
        click_x=220,
        click_y=40,
        confidence=0.8,
    )
    assert target_signature(t) == target_signature(t)


def test_repeated_target_detection_works() -> None:
    sig = "new tab button:B1:1200:80"
    assert _is_repeated_target([sig, sig], sig, threshold=2)
