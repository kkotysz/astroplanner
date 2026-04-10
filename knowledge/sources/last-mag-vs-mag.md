# Last Mag vs Mag

Summary: How to distinguish time-sensitive `Last Mag` from catalog `Mag` in AstroPlanner answers.
Tags: last-mag-vs-mag, bhtom, simbad, magnitude, source-metadata

## When this note applies

- The question is about brightness or magnitude.
- The answer mixes BHTOM and catalog-derived metadata.
- The user asks why one target appears brighter or fainter than another.

## Key heuristics

- `Last Mag` is a latest reported brightness-like value from a source such as BHTOM and should be treated as more time-sensitive than a generic catalog `Mag`.
- Plain `Mag` is usually catalog metadata and may be less representative of the current state for transients.
- When both are relevant, prefer `Last Mag` for practical transient ranking and `Mag` for static reference context.

## What to tell the user

- Say explicitly whether a number is `Last Mag` or `Mag`.
- Do not blur a time-sensitive transient brightness with a static catalog value.
- If brightness is central to the recommendation, mention the label, not just the number.

## Caveats

- `Last Mag` may still be stale depending on the cadence of the source feed.
- Different sources may report brightness in different contexts or bands.
