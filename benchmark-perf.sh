#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is required. Install it from https://docs.astral.sh/uv/" >&2
  exit 1
fi

exec uv run --project . python src/benchmark_performance.py "$@"
