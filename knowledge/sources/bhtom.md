# BHTOM Fields and Caveats

Summary: Meaning and limitations of BHTOM fields used by AstroPlanner suggestions.
Tags: bhtom, importance, last-mag, transient-sources

## When this note applies

- The question mentions BHTOM.
- The answer relies on `importance`, `Last Mag`, or BHTOM-only candidates.
- The user asks why a BHTOM target is being recommended.

## Key heuristics

- `Importance` is a source-side prioritization hint, not a direct visibility metric.
- `Last Mag` is observationally useful, but it should be weighed together with altitude, airmass, and available window.
- A BHTOM suggestion should still be justified using AstroPlanner night metrics rather than source metadata alone.

## What to tell the user

- Say when a recommendation is strong because both BHTOM metadata and tonight's visibility agree.
- If BHTOM importance is high but tonight's geometry is poor, say that explicitly.
- Use `Last Mag` as practical context, especially for transient objects.

## Caveats

- BHTOM metadata can be stale or incomplete.
- `Importance` should not be treated as equivalent to physical observability.
