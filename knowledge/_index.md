# AstroPlanner Knowledge Base

Summary: Local markdown knowledge notes used to ground AI Assistant answers with practical astronomy guidance.
Tags: knowledge-base, ai, grounding

## Purpose

- Provide concise, practical heuristics for AstroPlanner AI answers.
- Keep reasoning grounded in project-specific terminology and observing workflow.
- Avoid using this folder for deterministic scoring, ranking, or night-window calculations.

## Note Types

- `object-classes/`: practical guidance for classes such as SN or QSO.
- `observing/`: cross-cutting heuristics such as moonlight or best-window interpretation.
- `sources/`: meaning and caveats of external catalogs and fields.

## Retrieval Rules

- Prefer at most 1-3 notes per answer.
- Prefer `Summary`, `Key heuristics`, and `Caveats`.
- Do not inject notes when they are not relevant to the question.
