#!/bin/bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"

# Use explicit PYTHON env var if provided, otherwise fall back to python3.
PY="${PYTHON:-python3}"

# Forward optional CLI args (e.g. --plan examples/plan_targets.json).
exec "$PY" "$APP_DIR/astro_planner.py" "$@"
