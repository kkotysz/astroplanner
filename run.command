#!/bin/bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"

# Use the venv's python directly (no need to "activate")
PY="/Users/krzkot/miniconda3/envs/astroplanner/bin/python"

# Run your program (adjust entrypoint)
exec "$PY" "$APP_DIR/astro_planner.py"
