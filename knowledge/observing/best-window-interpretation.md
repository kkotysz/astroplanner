# Best Window Interpretation

Summary: How to interpret `best window`, `over limit`, `score`, and `min airmass` in AstroPlanner.
Tags: best-window, score, over-limit, airmass, altitude, planning

## When this note applies

- The user asks which target is best tonight.
- The answer should explain why one target ranks above another.
- The question mentions best window, score, order, airmass, or altitude.

## Key heuristics

- `Best window` is the most useful continuous observing segment under the current filters, not the whole night.
- `Over limit` tells you how much usable time exists above the altitude threshold, which is often more practical than a single current altitude snapshot.
- `Min airmass` in the best window is a stronger practical quality signal than score when comparing similar targets.

## What to tell the user

- Explain which metric is actually deciding the recommendation: window timing, over-limit hours, min airmass, or brightness.
- Avoid pretending that score alone is the full answer.
- If a target has a short but urgent window, say that explicitly.

## Caveats

- A high score does not guarantee the target is optimal at the current moment.
- A target with a later or shorter best window may still be better if it is much brighter or cleaner from moonlight.
