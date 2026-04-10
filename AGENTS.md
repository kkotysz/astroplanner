# AGENTS.md

## Project overview

AstroPlanner is a desktop observation planner built mainly around a single large PySide6 application file:

- main entry point: `astro_planner.py`
- AI grounding notes: `knowledge/`
- Docker helpers and local services: `docker/`, `scripts/`, `Makefile`

The app combines:

- visibility planning and scoring
- target management and metadata enrichment
- BHTOM-based target suggestions
- SIMBAD / TNS / Gaia / NED resolver integrations
- optional local LLM chat through an OpenAI-compatible backend

## Working environment

Prefer the Conda environment named `astroplanner`.

Typical commands:

```bash
conda activate astroplanner
python astro_planner.py
python3 -m py_compile astro_planner.py
```

Useful local stack commands:

```bash
make help
make llm-install-help
make llm-pull
make llm-check
make llm-up-docker
make up-seestar-sim
```

## Validation

Minimum validation after code changes:

```bash
python3 -m py_compile astro_planner.py
```

Use heavier checks only when relevant and available:

- run focused verification for touched code paths
- avoid full GUI/manual flows unless the change actually touches them
- if a check cannot be run, say so explicitly

## Repository workflow

This repository may be in a dirty state. Treat local changes carefully.

- never revert unrelated user changes
- stage selectively
- do not use destructive git commands such as `git reset --hard`
- do not amend commits unless explicitly asked
- if a file has unexpected changes that conflict with your work, stop and ask

## Code style and implementation guidelines

- keep changes focused and local
- prefer deterministic logic over LLM behavior for anything correctness-critical
- use ASCII unless the file already needs Unicode
- add comments only when they clarify non-obvious logic
- prefer small helpers over spreading ad-hoc conditionals through the UI code

`astro_planner.py` is large. When editing it:

- preserve existing naming and signal/slot patterns
- avoid broad refactors unless explicitly requested
- be careful with UI refresh paths; unnecessary full refreshes can reload Aladin or disturb chat/streaming behavior

## AI and knowledge-base rules

The app has two different classes of AI-like behavior:

1. Deterministic/local features:
- `Describe Object`
- `Suggest Targets`
- scoring, windows, ordering, filtering

These should stay local and deterministic unless the user explicitly asks otherwise.

2. LLM chat in the AI Assistant:
- uses an OpenAI-compatible backend
- should stay lightweight
- latency matters more than elaborate prompting

Current backend examples:

- Ollama:
  - URL: `http://localhost:11434`
  - model: `gemma4:e4b`
- Docker Model Runner:
  - URL: `http://localhost:12434/engines`
  - model: `docker.io/ai/gemma4:E4B`

Reasoning / thinking behavior depends on backend and model. Do not assume every backend supports it the same way.

Warm-up is implemented as a lightweight chat-completions request. It is broadly portable across OpenAI-compatible backends, but the performance benefit depends on whether the backend keeps the model hot.

## Knowledge notes

The AI Assistant uses local grounding notes from `knowledge/`.

- keep notes short and practical
- prefer heuristics and caveats over encyclopedia-style text
- do not duplicate metrics already computed by code
- do not replace deterministic planning logic with markdown knowledge

When adding notes:

- use the existing structure under `knowledge/object-classes/`, `knowledge/observing/`, and `knowledge/sources/`
- follow the template in `knowledge/_templates/note.md`
- optimize for small prompt snippets, not long-form prose

## External integrations

### BHTOM

- `Suggest Targets` and `Quick Targets` rely on the BHTOM API
- token is configured in app settings, not committed to the repo
- if BHTOM data is already cached in-session, prefer using the cache instead of re-fetching

### SIMBAD and resolvers

- prefer exact/name-based matches first
- coordinate fallback is useful, but should be surfaced clearly when used
- if user-edited target names differ from source names, respect the edited name in lookup attempts

## UI expectations

- preserve the current desktop-first workflow
- do not move controls casually; UI changes should be intentional
- AI chat should remain readable, responsive, and stable during streaming
- if adding configurable appearance options, ensure they actually affect the rendered UI

## Screenshots and README refresh

If a UI change affects the documented layout or visible controls, consider whether README screenshots should be refreshed.

Relevant commands:

```bash
python scripts/capture_readme_screenshots.py --views all
python scripts/generate_readme_overlays.py --views all
```

Guidelines:

- refresh screenshots only when the UI change is user-visible and materially changes the documented views
- if you update screenshots, also check the matching annotated overlays and README captions/legends
- do not regenerate screenshots as part of unrelated backend-only or logic-only changes

## Commit style

Use short, factual commit messages, typically:

- `fix: ...`
- `feat: ...`
- `docs: ...`

Keep scope clear. Split unrelated changes into separate commits when asked.
