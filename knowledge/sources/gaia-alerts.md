# Gaia Alerts Caveats

Summary: How to interpret Gaia Alerts objects and alert metadata when AstroPlanner resolves a target from that source.
Tags: gaia-alerts, gaia, transient-sources, alert-streams

## When this note applies

- The question mentions Gaia Alerts.
- The object source is Gaia Alerts.
- The user asks what a Gaia alert means or how much to trust it operationally.

## Key heuristics

- Gaia Alerts are useful for discovering interesting transient or variable behavior, but they are not a guarantee that the object is an easy observing target tonight.
- Use Gaia Alerts as a discovery and context source; use AstroPlanner's geometry, min airmass, and best window to judge practical observability.
- If another source provides a better current classification, treat Gaia Alerts as supporting context rather than the final word on type.

## What to tell the user

- Explain whether Gaia Alerts contributes discovery context, classification context, or just cross-identification.
- Be explicit when a Gaia alert is interesting scientifically but not necessarily optimal tonight.
- Keep the answer grounded in current observing conditions rather than alert novelty alone.

## Caveats

- Alert labels can be provisional or incomplete.
- Gaia Alerts metadata is not a substitute for current geometry or for a well-resolved object type from a more authoritative catalog.
