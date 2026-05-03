# AutomationVisionAgent

Local desktop app for an AI-assisted **screen-control agent**: capture the display, **visually ground** click targets (grid-annotated screenshot + optional OpenCV template match + LLM bounding box), then move/click/type/scroll in a **bounded** loop (emergency stop, max steps).

## Current status

**Step 4 + Step 6 (implemented):** **Cursor-only** browser/system control: no hotkeys in settings by default (`actions.allow_hotkeys: false`, `actions.allow_press: false`). The **action planner** and **target locator** prompts forbid Command/Ctrl shortcuts; execution rejects `hotkey` / `press` when disabled.

**Visual grounding (Step 6):** Each step can use, in order:

1. **Template matching** — optional `opencv` match on the **full capture** for PNGs in `assets/templates/` (e.g. a crop of the new-tab `+` icon). No file → skipped safely.
2. **LLM bounding box** — an **annotated** screenshot (grid + axis labels, optional cursor marker) is sent to a dedicated **target locator** model call (`response_format: json_object`). The model returns a box in **resized** image space; the app validates confidence, maps the box to **capture** space, and **clicks the center** (no raw point guess for that path).
3. **Action planner fallback** — if `targeting.fallback_to_action_planner` is `true` and targeting fails, the existing JSON **action plan** runs (move/click/type/scroll/wait only in instructions).

**Debug images** after a successful target decision: `screenshots/debug_target_step_001.png`, `..._002.png`, … (drawn box + label).

**Why boxes instead of a single (x,y):** A tight **rectangle** around a control is easier to hit than one mis-guessed pixel; the app uses the **box center** for the click when `targeting.click_box_center` is `true`.

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
- `vision` — annotated screenshots, grid spacing, hide overlay while capturing  
- `targeting` — bounding-box path, `min_target_confidence`, template threshold, retries, `fallback_to_action_planner`  
- `actions` — `allow_hotkeys`, `allow_press` (default false = cursor + type/scroll only)  
- `safety` — failsafe, optional risky-action skipping  

## Run the app

```bash
python -m app.main
```

Example (cursor-oriented; no hotkeys in default config):

```text
Open a new tab in the browser
```

You should see: capture (overlay hidden if configured) → template pass (if you add a template) or **annotated** image to the **locator** → validated **click** at box center → debug PNG under `screenshots/`, or **planner** fallback on failure if enabled.

## Templates under `assets/templates/`

- Add one or more **PNG** crops of the control (same scale as on your display as much as possible).  
- Filename is arbitrary (e.g. `new_tab_chrome.png`).  
- If **no** template is present, template matching is skipped.  
- Browser chrome **varies** by browser, theme, and zoom — you may need different crops per machine.

## Limitations

- **Tiny icons** and **Retina / scaling** still make matching hard.  
- **LLM** can still pick the **wrong** box; always use **Stop** if it drifts.  
- **Template** quality and **threshold** (`targeting.template_match_threshold`) must be tuned per environment.  
- **Multi-monitor** / **HiDPI** alignment between mss and pyautogui is not perfect.  
- **No** screenshot-diff “wait until UI changed” yet (only `step_wait_seconds`).

## Run tests

```bash
pytest
```

Tests do **not** move the mouse, call OpenAI, or open the full GUI (except what import side effects allow).

## Project layout (main pieces)

- `app/main.py` — Qt, worker, overlay passed into the agent for capture hide  
- `app/agent.py` — observe–act loop, **targeting** then **planner** fallback  
- `app/llm.py` — target locator + action planner, `filter_disallowed_actions`  
- `app/targeting.py` — `TargetBox`, template match, `draw_target_box`  
- `app/vision.py` — capture, **annotated** screenshot, `save_debug_image`  
- `app/executor.py` — pyautogui; blocks `hotkey`/`press` when config says so  
- `app/overlay.py` — `visibility_for_capture` to hide during mss grab  

## What to implement next

- Optional **wait-for-visual-change** after each action  
- Finer **per-step** policy (e.g. only run targeting when the user intent is “click X”)  
- Optional confirmation for risky actions when `require_confirmation_for_risky_actions: true`  
