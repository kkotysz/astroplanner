# AstroPlanner Architecture

This document describes the current repository shape after the staged split of
`astro_planner.py`. It is a map of the code that exists now, not a future refactor
plan.

## Runtime Shape

AstroPlanner is still launched from `astro_planner.py`. That file owns the
application entry point, `MainWindow`, top-level widget composition, the highest
risk refresh paths, and compatibility glue for older code that has not been moved
yet.

The `astroplanner/` package now contains the extracted domain, UI, storage, worker,
and coordinator modules. The dependency direction is intentionally one-way:

- `astro_planner.py` imports and composes modules from `astroplanner/`.
- Feature modules do not import `astro_planner.py` at runtime.
- Coordinator modules may reference `MainWindow` only behind `TYPE_CHECKING` for
  type hints.
- Extracted dialogs communicate back through callbacks, settings adapters,
  repositories, worker signals, or explicit parent objects instead of importing
  `MainWindow`.

## Core Modules

- `astroplanner/models.py`
  Stable domain models: `Target`, `Site`, `SessionSettings`, `CalcRunStats`, and
  `targets_match`.
- `astroplanner/app_config.py`
  App settings directory resolution, SQLite settings factory, and obsolete-key
  cleanup.
- `astroplanner/scoring.py`
  Deterministic per-target night scoring and `TargetNightMetrics`.
- `astroplanner/astronomy.py`
  Background calculation workers, including `AstronomyWorker` and `ClockWorker`.
- `astroplanner/bhtom.py`
  BHTOM API fetch/parse helpers, candidate deduplication, local suggestion ranking,
  and BHTOM worker threads.
- `astroplanner/seestar.py`
  Seestar domain logic, queue generation, ALP schedule payloads, handoff rendering,
  ALP clients, and device adapters.
- `astroplanner/weather.py`
  Weather provider/cache/worker logic used by the weather UI.
- `astroplanner/visibility_plotly.py`
  Pure Plotly HTML builder and airmass-axis constants for the interactive
  visibility chart.
- `astroplanner/ai.py`
  Knowledge-note parsing, deterministic AI intent helpers, OpenAI-compatible model
  discovery, LLM config, and LLM request workers.
- `astroplanner/storage/`
  SQLite-backed settings, state, cache, observatories, plans, templates,
  observation log, and chat history repositories.
- `astroplanner/exporters.py`
  File export helpers for metrics, calendar handoff, Seestar payloads, CSV, and
  checklist markdown.
- `astroplanner/parsing.py`, `astroplanner/qt_helpers.py`
  Small focused helpers that are shared without becoming a generic catch-all module.

## UI Modules

- `astroplanner/ui/common.py`
  Shared UI primitives such as `TargetTableView`, table-width distribution, dialog
  sizing, and loading skeleton widgets.
- `astroplanner/ui/theme_utils.py`
  Narrow theme-aware helpers used by extracted UI modules.
- `astroplanner/ui/widgets.py`
  Shared custom widgets used by the shell, including the neon toggle and zoomable
  cover image label.
- `astroplanner/ui/targets.py`
  Main target table model and delegate: `TargetTableModel` and
  `TargetTableGlowDelegate`.
- `astroplanner/ui/suggestions.py`
  `Suggest Targets` dialog, table model, and delegate.
- `astroplanner/ui/observatory.py`
  Observatory manager dialog, add-observatory dialog, and lookup worker.
- `astroplanner/ui/add_target.py`
  Add-target dialog, metadata lookup worker, and finder chart worker.
- `astroplanner/ui/seestar.py`
  Seestar session planning dialog.
- `astroplanner/ui/settings.py`
  Table and general settings dialogs.
- `astroplanner/ui/plans.py`
  Saved-plan picker and observation-history dialog.
- `astroplanner/ui/weather.py`
  Weather workspace dialog, tabs, plotting helpers, and UI refresh logic.

## Coordinators

The current coordinator layer extracts orchestration that is tightly coupled to
`MainWindow` state but does not need to live directly inside the main file.

- `astroplanner/targets_coordinator.py`
  Main target table setup, table settings application, visibility filtering,
  column-width refreshes, selection scheduling, and related table lifecycle glue.
- `astroplanner/visibility_coordinator.py`
  Visibility plot refresh scheduling, selected-target/cutout synchronization,
  Plotly web rendering/cache, Plotly selection styling, polar selection, and
  show/hide orchestration.
- `astroplanner/observatory_coordinator.py`
  Observatory persistence, default config loading, coordinate lookup, and combo
  refresh glue.
- `astroplanner/ai_panel_coordinator.py`
  AI Assistant window composition, chat transcript rendering, warm-up lifecycle,
  LLM dispatch, streaming updates, and AI status handling.
- `astroplanner/plan_coordinator.py`
  Plan snapshots, workspace autosave, named-plan load/save, JSON plan import,
  per-plan AI chat persistence, and observation-log writes.

These coordinators deliberately accept a `MainWindow` instance because their job is
to reduce the size of `astro_planner.py` without pretending that the underlying
state is independent yet.

## Data And Configuration

- `config/default_observatories.json`
  Runtime seed data for first-run/default observatories. This file is still needed
  by the application and is intentionally kept under `config/`.
- `examples/plan_targets.json`
  Example plan for demos, tests, and screenshot capture.
- `knowledge/`
  Local grounding notes used by the AI Assistant. These remain short prompt snippets
  and do not replace deterministic planning logic.
- `assets/icon.png`
  Repository-level static app asset.
- `docs/`
  Architecture, setup, and screenshot documentation.

## MainWindow Responsibilities

`MainWindow` is currently still responsible for:

- composing the main desktop UI shell;
- owning global app state, settings, storage, timers, and worker lifecycles;
- connecting menus, toolbars, dialogs, and coordinators;
- handling the highest-risk visibility, cutout, Aladin/finder, and refresh paths;
- preserving existing signal/slot behavior and settings compatibility.

Long-term, `MainWindow` should keep shrinking, but it is not currently a thin shell.
The present architecture favors safe extraction of stable slices over a big-bang
rewrite.

## Dependency Rules

- Do not add runtime imports from `astroplanner/` modules back to `astro_planner.py`.
- Keep new shared helpers focused; do not create a broad `utils.py`.
- Prefer feature modules for feature-specific helpers.
- Preserve class names, Qt signal names, settings keys, callback signatures, and
  visible UI behavior during structural refactors.
- End each extraction in a compilable/importable state.
- Avoid moving controls or changing screenshots unless the task is intentionally a
  visible UI change.

## Remaining Monolith Areas

The main file is smaller but still contains substantial legacy surface area. The
largest remaining candidates are:

- high-level `MainWindow` orchestration and startup/shutdown lifecycle;
- visibility/cutout/finder rendering internals that are still sensitive to refresh
  timing, especially Matplotlib, cutout, and finder preview paths;
- Aladin integration and preview cache glue;
- remaining menu/action wiring and app-wide settings application;
- cross-feature workflows that still span table state, selected target state,
  weather, Seestar, and AI.

Future cleanup should keep following the same rule: extract cohesive, tested slices
only when the new module can avoid runtime dependence on `astro_planner.py`.
