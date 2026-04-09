# SIMBAD Metadata Caveats

Summary: How to interpret SIMBAD-derived type, photometry, and classification fields in AstroPlanner.
Tags: simbad, source-metadata, classification

## When this note applies

- The question refers to SIMBAD.
- The answer relies on SIMBAD type, photometry, spectral type, or object classification.
- The user asks why AstroPlanner describes an object in a certain way.

## Key heuristics

- SIMBAD object type is usually the best authoritative classification available in AstroPlanner for broad object class questions.
- SIMBAD photometry is useful as factual context, but it may mix measurements from different epochs or bands and should not be treated as a live brightness report.
- If SIMBAD and another source disagree, use SIMBAD for class identification and use time-sensitive catalogs for current transient state.

## What to tell the user

- Say clearly when the answer is grounded in SIMBAD classification.
- Distinguish between object class and current observing state.
- If a match was by coordinates rather than name, mention that it is a coordinate-based match.

## Caveats

- Name resolution can fail or differ because of aliases and spacing.
- SIMBAD is not a live transient feed, so it should not be treated as the best source for current brightness changes.
