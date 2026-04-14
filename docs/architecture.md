# AstroPlanner Architecture

AstroPlanner is still centered around the `MainWindow` in `astro_planner.py`, but the codebase is now being split into focused support modules under `astroplanner/`.

## Current boundaries

- `astro_planner.py`
  Keeps the application entry point, `MainWindow`, most orchestration logic, and the highest-risk refresh paths.
- `astroplanner/models.py`
  Stable observation-domain models such as `Target`, `Site`, `SessionSettings`, and lightweight calculation metadata.
- `astroplanner/bhtom.py`
  BHTOM API payload parsing, candidate building, local suggestion ranking, and BHTOM worker threads.
- `astroplanner/ui/common.py`
  Shared UI helpers that are safe to reuse across dialogs and widgets.
- `astroplanner/ui/theme_utils.py`
  Minimal theme-aware widget helpers used by extracted UI modules.
- `astroplanner/ui/suggestions.py`
  `Suggest Targets` table model, delegate, and dialog.
- `astroplanner/seestar.py`
  Seestar-specific domain logic and adapters.
- `astroplanner/storage/`
  SQLite-backed settings, cache, and app state helpers.

## Refactor rules

- `MainWindow` remains the shell and orchestrator until a later pass.
- New extracted modules must not import `MainWindow`.
- Prefer moving complete feature slices or stable helpers over partial cross-imports.
- When a new module needs a small helper, move or duplicate that helper into a focused shared module instead of importing back from `astro_planner.py`.
- UI-visible behavior, settings keys, and worker signal contracts should stay unchanged during structural refactors.

## Near-term direction

The intended extraction order is:

1. stable domain models
2. BHTOM suggestion logic and workers
3. shared UI helpers
4. `Suggest Targets`
5. observatory and add-target flows
6. Seestar session UI
7. AI assistant internals
8. settings dialogs
9. weather stack

This keeps the riskiest areas (`MainWindow`, weather refresh paths, and the main targets table) for later, after the supporting seams are already in place.
