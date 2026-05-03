"""Load environment variables and YAML application settings."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from app.utils import project_root


class AppSection(BaseModel):
    name: str = "AutomationVisionAgent"
    debug: bool = False


class ScreenSection(BaseModel):
    monitor_index: int = 1
    screenshot_width: int = 1280


class AgentSection(BaseModel):
    max_steps: int = 8
    max_actions_per_step: int = 3
    action_delay_seconds: float = 0.35
    step_wait_seconds: float = 1.0
    retry_on_invalid_coordinates: bool = True
    max_coordinate_retries: int = 1


class LlmSection(BaseModel):
    model: str = "gpt-4.1-mini"
    max_actions: int = 3
    temperature: float = 0.0


class SafetySection(BaseModel):
    pyautogui_failsafe: bool = True
    emergency_stop_key: str = "escape"
    require_confirmation_for_risky_actions: bool = True


class VisionSection(BaseModel):
    coordinate_map_enabled: bool = True
    grid_cols: int = 12
    grid_rows: int = 8
    fine_grid_spacing: int = 50
    save_coordinate_map_debug: bool = True
    annotated_screenshots: bool = True
    coordinate_grid_spacing: int = 100
    draw_cursor_marker: bool = True
    draw_grid_labels: bool = True
    hide_overlay_during_capture: bool = True


class TargetingSection(BaseModel):
    use_bounding_boxes: bool = True
    require_grid_cell: bool = True
    require_click_point_inside_box: bool = True
    require_box_inside_screenshot: bool = True
    min_target_confidence: float = 0.70
    max_locator_retries: int = 2
    click_box_center: bool = True
    enable_template_matching: bool = True
    template_match_threshold: float = 0.78
    max_targeting_retries: int = 1
    fallback_to_action_planner: bool = True
    duplicate_click_min_distance_px: int = 28
    require_user_confirmation_before_click: bool = False


class ActionsSection(BaseModel):
    allow_hotkeys: bool = False
    allow_press: bool = False


class Settings(BaseModel):
    """Typed view of `config/settings.yaml`."""

    app: AppSection = Field(default_factory=AppSection)
    screen: ScreenSection = Field(default_factory=ScreenSection)
    agent: AgentSection = Field(default_factory=AgentSection)
    llm: LlmSection = Field(default_factory=LlmSection)
    safety: SafetySection = Field(default_factory=SafetySection)
    vision: VisionSection = Field(default_factory=VisionSection)
    targeting: TargetingSection = Field(default_factory=TargetingSection)
    actions: ActionsSection = Field(default_factory=ActionsSection)


def _settings_path() -> Path:
    return project_root() / "config" / "settings.yaml"


def load_settings(path: Path | None = None) -> Settings:
    """Load and validate settings from YAML. Raises FileNotFoundError with guidance if missing."""
    resolved = path or _settings_path()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Settings file not found: {resolved}\n"
            "Create `config/settings.yaml` at the project root, or copy it from the repository template."
        )
    try:
        raw_text = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Could not read settings file {resolved}: {exc}") from exc

    try:
        data: Any = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {resolved}: {exc}") from exc

    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"Settings root must be a mapping (dict), got {type(data).__name__}")

    try:
        return Settings.model_validate(data)
    except Exception as exc:
        raise ValueError(f"Settings failed validation ({resolved}): {exc}") from exc


# Use override=True so values in `.env` win over empty or stale env vars
# (the default `override=False` often blocks the key the user just added to `.env`).
_env_file = project_root() / ".env"
_dotenv_loaded = load_dotenv(_env_file, override=True, encoding="utf-8-sig")

try:
    SETTINGS: Settings = load_settings()
except Exception as exc:
    print(f"ERROR: Failed to load configuration: {exc}", file=sys.stderr)
    raise

OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY") or None
if OPENAI_API_KEY is not None and not OPENAI_API_KEY.strip():
    OPENAI_API_KEY = None

if SETTINGS.app.debug:
    _has_key = bool(OPENAI_API_KEY)
    print(
        f"[config] Loaded .env from {_env_file.resolve()} "
        f"(file_exists={_env_file.is_file()}, dotenv_ok={_dotenv_loaded}, OPENAI_API_KEY_loaded={_has_key})"
    )
    if not _has_key:
        print(
            "[config] Note: `echo $OPENAI_API_KEY` in the terminal will usually be empty — "
            "shells do not auto-load `.env`; only this Python process reads it via dotenv.",
            file=sys.stderr,
        )
