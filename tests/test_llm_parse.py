"""JSON plan parsing without calling OpenAI."""

from __future__ import annotations

import json

import pytest

from app.llm import parse_model_response_to_plan


def test_parse_minimal_valid_json() -> None:
    raw = (
        '{"done":false,"summary":"pause","actions":[{"type":"wait","duration_seconds":1,"reason":"hold"}]}'
    )
    plan = parse_model_response_to_plan(raw)
    assert len(plan.actions) == 1
    assert plan.actions[0].type == "wait"


def test_parse_rejects_invalid_json() -> None:
    with pytest.raises(json.JSONDecodeError):
        parse_model_response_to_plan("not json")
