# Choosing Between Similar Targets

Summary: Practical tie-breakers for selecting between targets of similar class or interest.
Tags: choosing-between-similar-targets, planning, best-window, practical-observing

## When this note applies

- The user asks which of several similar targets is best.
- The candidates have similar score or similar source importance.
- The assistant needs to justify a recommendation instead of listing everything equally.

## Key heuristics

- Prefer the target with the best combination of brightness, lower min airmass, and a cleaner or earlier-closing useful window.
- If two targets are similar, an earlier-closing window is a valid practical reason to recommend one first.
- Moon separation should break ties more strongly for faint or diffuse objects than for bright point sources.

## What to tell the user

- State the deciding metric explicitly: brightness, min airmass, best window timing, moon separation, or over-limit time.
- If the choice is close, say that it is close and explain the tie-breaker.
- Avoid pretending that score alone settled the question.

## Caveats

- A single metric rarely decides every comparison.
- A source-side importance value should not dominate an obviously weaker geometry case.
