"""Visual grounding, targeting, annotations (no mouse automation)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from app.executor import ActionExecutor
from app.llm import filter_disallowed_actions
from app.schema import Action, AgentPlan
from app.targeting import TargetBox, center_of_box, draw_target_box, locate_template
from app.vision import create_annotated_screenshot, save_debug_image


def test_target_box_validation_ok() -> None:
    b = TargetBox(
        label="btn",
        x1=10,
        y1=20,
        x2=50,
        y2=60,
        confidence=0.9,
        source="test",
    )
    assert b.within_bounds(100, 100)


def test_target_box_invalid_order() -> None:
    with pytest.raises(ValueError):
        TargetBox(
            label="bad",
            x1=50,
            y1=50,
            x2=10,
            y2=10,
            confidence=0.9,
            source="test",
        )


def test_center_of_box() -> None:
    b = TargetBox(
        label="x",
        x1=0,
        y1=0,
        x2=10,
        y2=10,
        confidence=1.0,
        source="t",
    )
    assert center_of_box(b) == (5, 5)


def test_low_confidence_rejected_by_agent_logic() -> None:
    b = TargetBox(
        label="x",
        x1=0,
        y1=0,
        x2=5,
        y2=5,
        confidence=0.2,
        source="llm_visual",
    )
    assert b.confidence < 0.55


def test_annotated_same_dimensions() -> None:
    img = Image.new("RGB", (320, 240), color=(30, 30, 40))
    ann = create_annotated_screenshot(img, grid_spacing=50)
    assert ann.size == img.size


def test_save_debug_image_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.vision.project_root", lambda: tmp_path)
    img = Image.new("RGB", (10, 10), color=(0, 0, 0))
    p = save_debug_image(img, "debug_target_step_001")
    assert p.name == "debug_target_step_001.png"
    assert p.parent.name == "screenshots"
    assert p.is_file()


def test_draw_target_box_returns_copy() -> None:
    img = Image.new("RGB", (100, 80), color=(0, 0, 0))
    box = TargetBox(
        label="t",
        x1=10,
        y1=10,
        x2=40,
        y2=40,
        confidence=0.99,
        source="x",
    )
    out = draw_target_box(img, box)
    assert out is not img


def test_hotkey_rejected_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import SETTINGS

    monkeypatch.setattr(
        SETTINGS,
        "actions",
        SETTINGS.actions.model_copy(update={"allow_hotkeys": False}),
    )
    ex = ActionExecutor(overlay=None, controller=None)
    act = Action(type="hotkey", keys=["command", "t"])
    assert ex.execute_action(act) is False


def test_press_rejected_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import SETTINGS

    monkeypatch.setattr(
        SETTINGS,
        "actions",
        SETTINGS.actions.model_copy(update={"allow_press": False}),
    )
    ex = ActionExecutor(overlay=None, controller=None)
    act = Action(type="press", key="enter")
    assert ex.execute_action(act) is False


def test_filter_removes_hotkey(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import SETTINGS

    monkeypatch.setattr(
        SETTINGS,
        "actions",
        SETTINGS.actions.model_copy(update={"allow_hotkeys": False}),
    )
    plan = AgentPlan(
        actions=[Action(type="hotkey", keys=["a"])],
    )
    out = filter_disallowed_actions(plan)
    assert len(out.actions) == 1
    assert out.actions[0].type == "wait"


def test_template_match_missing_file_returns_none() -> None:
    img = Image.new("RGB", (200, 200), color=(128, 128, 128))
    assert locate_template(img, Path("/nonexistent/path/template.png")) is None


def test_locate_target_response_parse() -> None:
    from app.llm import _parse_locate_response

    raw = '{"found": false, "reason": "nothing"}'
    r = _parse_locate_response(raw)
    assert r.found is False
