#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_NAME="${ASTROPLANNER_COMPOSE_PROJECT_NAME:-astroplanner}"

export COMPOSE_PROJECT_NAME="${PROJECT_NAME}"
export COMPOSE_IGNORE_ORPHANS="${COMPOSE_IGNORE_ORPHANS:-True}"

exec docker compose \
  -p "${PROJECT_NAME}" \
  -f "${ROOT_DIR}/docker-compose.yml" \
  "$@"
