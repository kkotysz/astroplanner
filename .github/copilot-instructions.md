# Copilot Instructions for AstroPlanner

## Build, Test, and Lint Commands

- **Preferred environment:** `conda activate astroplanner` (Python 3.12)
- **Run the app:** `python astro_planner.py` — on macOS: `./run.command`
- **Minimal syntax validation:** `python3 -m py_compile astro_planner.py`
- **Run a single test file:** `pytest tests/test_<module>.py`
- **Run all tests:** `pytest`
- **Useful Makefile targets:** `make help`, `make llm-check`, `make up-seestar-sim`

## High-Level Architecture

`astro_planner.py` is a compatibility shim. All real code lives in `astroplanner/`:

- **`astroplanner/main_window.py`** — desktop shell, owns global app state, timers, worker lifecycles, and the highest-risk refresh paths (Aladin, chart, cutout)
- **`astroplanner/models.py`** — stable domain models (`Target`, `Site`, `SessionSettings`) using Pydantic v2 `BaseModel`
- **`astroplanner/scoring.py`** — deterministic per-target night scoring (`TargetNightMetrics`)
- **`astroplanner/storage/app.py`** — SQLite-only settings/state/cache/observatory/plan repositories; **QSettings and .ini fallbacks are gone**
- **`astroplanner/ui/`** — extracted dialog and widget modules
- **`astroplanner/*_coordinator.py`** — orchestration layer; each coordinator receives a `MainWindow` instance to reduce the size of `main_window.py` without faking independence
- **`astroplanner/ai.py` + `ai_context.py` + `ai_panel_coordinator.py`** — LLM config, knowledge-note loading, prompt assembly, streaming chat
- **`knowledge/`** — short AI grounding notes (not planning logic); template at `knowledge/_templates/note.md`

The dependency direction is intentionally one-way: `astro_planner.py` → `astroplanner/main_window.py` → feature modules. Feature modules must not import `astro_planner.py` at runtime. Coordinators may reference `MainWindow` only under `TYPE_CHECKING`.

## Key Conventions

**Python style:**
- Every module starts with `from __future__ import annotations`
- Domain models extend Pydantic v2 `BaseModel`; prefer `Field(...)` for new fields
- Use ASCII unless the file already contains Unicode
- Add comments only for non-obvious logic; prefer small helpers over ad-hoc conditionals

**Architecture rules (from `docs/architecture.md`):**
- Do not add runtime imports from `astroplanner/` back to `astro_planner.py`
- Do not create a generic `utils.py`; put helpers in the relevant feature module
- Preserve class names, Qt signal names, settings keys, and callback signatures during refactors
- Each extraction must leave the codebase in a compilable/importable state

**MainWindow and UI:**
- Preserve existing signal/slot patterns and naming
- Avoid unnecessary full refreshes — they reload Aladin or disturb chat/streaming
- UI strings should use `set_translated_text` / `set_translated_tooltip` from `astroplanner/i18n.py`

**AI/Knowledge:**
- `Describe Object`, `Suggest Targets`, scoring, and filtering must stay deterministic and local
- LLM chat is optional; uses any OpenAI-compatible `POST /v1/chat/completions` endpoint
- New knowledge notes go under `knowledge/object-classes/`, `knowledge/observing/`, or `knowledge/sources/`; follow `knowledge/_templates/note.md`

**External integrations:**
- BHTOM token is in app settings only — never committed
- Prefer cached BHTOM data over re-fetching within a session
- Respect user-edited target names in resolver lookups (don't silently overwrite)

**Screenshots:**
- Refresh only when a UI change materially affects the documented views:
  ```bash
  python scripts/capture_readme_screenshots.py --views all
  python scripts/generate_readme_overlays.py --views all
  ```

**Commits:** use `fix:`, `feat:`, or `docs:` prefixes; keep scope clear; split unrelated changes.

## Settings & data locations

- Runtime settings and app DB live under the platform config directory resolved by Qt/QStandardPaths or XDG (e.g. `~/.config/krzkot/AstroPlanner` on Linux). The SQLite DB file is `app.db` and cached assets are in `assets-cache` inside that directory (`astroplanner/app_config.py` and `astroplanner/storage/app.py`).
- Override the settings directory for tests or ephemeral runs with: `export ASTROPLANNER_CONFIG_DIR=/path/to/tmpdir`.

## Headless GUI tests

- Many tests import PySide6. For CI/headless runs, set an offscreen Qt platform or use Xvfb:
  - Offscreen example (single test):
    `QT_QPA_PLATFORM=offscreen ASTROPLANNER_CONFIG_DIR=$(mktemp -d) pytest tests/test_visibility_matplotlib.py::test_visibility_matplotlib_target_color_key_prefers_stable_ids -q`
  - Full suite with Xvfb:
    `xvfb-run -a pytest`

## Quick validation & examples

- Compile-check entire package: `python3 -m compileall astroplanner`
- Minimal syntax check: `python3 -m py_compile astro_planner.py`
- Run with an example plan: `python astro_planner.py --plan examples/plan_targets.json`
- Capture README screenshots (UI): `python scripts/capture_readme_screenshots.py --views all`

## LLM / integration helpers

- Useful Makefile helpers: `make llm-up-docker`, `make llm-pull`, `make llm-check`, `make up-seestar-sim` (see `Makefile`).

## Where to look next

- Architecture & dependency rules: `docs/architecture.md`
- AI grounding notes & templates: `knowledge/_templates/note.md` and `knowledge/`
- README and AGENTS.md contain setup and integration examples for LLMs and Seestar.

If you'd like, apply these additions or trim/expand any section.
