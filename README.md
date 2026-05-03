# AutomationVisionAgent

Local desktop app for an AI-assisted **screen-control agent**: capture the display, **visually ground** click targets (grid-annotated screenshot + optional OpenCV template match + LLM bounding box), then move/click/type/scroll in a **bounded** loop (emergency stop, max steps).

## Current status

**Step 4 + Step 6 (implemented):** **Cursor-only** browser/system control: no hotkeys in settings by default (`actions.allow_hotkeys: false`, `actions.allow_press: false`). The **action planner** and **target locator** prompts forbid Command/Ctrl shortcuts; execution rejects `hotkey` / `press` when disabled.

**Visual grounding (Step 6):** Every step creates a **coordinate-map screenshot** and runs a strict locator:

1. Build map overlay with coarse cells (A1, B1…), fine pixel grid, axis labels, and dimensions text.
2. Send this map image to the locator model and require:
   - `grid_cell`
   - bounding box (`x1,y1,x2,y2`)
   - explicit click point (`click_x, click_y`) inside the box
   - confidence
3. Validate target shape + bounds (no clamping), reject low-confidence targets, and retry with corrective feedback.
4. Click only the validated click point.

**Debug images**

- Coordinate map snapshot per step: `screenshots/debug_coordinate_map_step_001.png`
- Selected target overlay per step: `screenshots/debug_selected_target_step_001.png`

**Why this is stronger than raw x/y guessing:** the model references labeled cells and pixel axes, then must return both box and in-box click point, which is validated before execution.

**Overlay during capture:** If `vision.hide_overlay_during_capture` is `true`, the transparent overlay is **hidden** for one frame so it is not part of the screenshot.

**Safety / control**

- **Stop** and **Escape** (application-wide) = emergency stop; same as before.
- **PyAutoGUI failsafe** (default on): top-left corner aborts.
- **No infinite loop** — `agent.max_steps`. No Canvas- or quiz-specific logic in prompts.

## Requirements

- Python **3.11+**
- macOS/Windows/Linux with a display (PyQt6 + mss + pyautogui)
- **OpenAI API key** for the vision model in `config/settings.yaml` → `llm.model`
- **opencv-python-headless** (for template matching; optional for “locator + planner only” use)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### OpenAI API key

```bash
cp .env.example .env
```

Edit `.env`:

```bash
OPENAI_API_KEY=sk-...
```

Do **not** commit `.env`. The shell will **not** show the key with `echo $OPENAI_API_KEY` unless you export it yourself; the app loads `.env` via `python-dotenv`.

### Settings (summary)

`config/settings.yaml` includes:

- `screen` — monitor index, max width for downscaled “LLM” image  
- `llm` — model, `max_actions`, `temperature`  
- `agent` — `max_steps`, delays, coordinate retries  
- `vision` — coordinate-map overlay (`grid_cols`, `grid_rows`, fine grid spacing), debug map saves, hide overlay while capturing  
- `targeting` — required grid cell/click-point checks, confidence threshold, locator retries, consistency checks, optional review mode  
- `actions` — `allow_hotkeys=false`, `allow_press=false` by default (cursor + type/scroll only)  
- `safety` — failsafe, optional risky-action skipping  

## Run the app

```bash
python -m app.main
```

Example (cursor-oriented; no hotkeys in default config):

```text
Open a new tab in the browser
```

You should see: capture (overlay hidden if configured) → **coordinate-map screenshot** → locator JSON with `grid_cell + box + click point` → validation → cursor move/click → debug PNGs under `screenshots/`.

## Templates under `assets/templates/`

- Add one or more **PNG** crops of the control (same scale as on your display as much as possible).  
- Filename is arbitrary (e.g. `new_tab_chrome.png`).  
- If **no** template is present, template matching is skipped.  
- Browser chrome **varies** by browser, theme, and zoom — you may need different crops per machine.

## Target Review Mode

`targeting.require_user_confirmation_before_click` is available (default `false`).

- When `true`, the app prepares the selected target and requires a confirmation hook before clicking.
- If review mode is on but no confirmation hook is wired, the run stops safely.

## Limitations

- **LLM** can still misread very small icons.  
- Grid overlays can obscure tiny UI details in dense chrome.  
- Very small controls may still need zoom/crop workflows.  
- Confidence can still be overestimated by the model.  
- Browser chrome varies by theme/window size/platform.  
- **Multi-monitor** / **HiDPI** alignment between mss and pyautogui is not perfect.  
- **No** screenshot-diff “wait until UI changed” yet (only `step_wait_seconds`).

## Run tests

```bash
pytest
```

Tests do **not** move the mouse, call OpenAI, or open the full GUI (except what import side effects allow).

## Project layout (main pieces)

- `app/main.py` — Qt, worker, overlay passed into the agent for capture hide  
- `app/agent.py` — observe–act loop, coordinate-map locator retries, consistency protection  
- `app/llm.py` — coordinate-map target locator + action planner, disallowed-action filtering  
- `app/targeting.py` — strict `LocatedTarget` model + bounds/signature helpers  
- `app/vision.py` — coordinate-map rendering + selected-target debug drawing  
- `app/executor.py` — pyautogui; blocks `hotkey`/`press` when config says so  
- `app/overlay.py` — `visibility_for_capture` to hide during mss grab  

## What to implement next

- Optional **wait-for-visual-change** after each action  
- Finer **per-step** policy (e.g. only run targeting when the user intent is “click X”)  
- Optional confirmation for risky actions when `require_confirmation_for_risky_actions: true`  
