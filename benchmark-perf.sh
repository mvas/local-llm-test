#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is required. Install it from https://docs.astral.sh/uv/" >&2
  exit 1
fi

# Load .env from the project root if it exists (e.g. HF_TOKEN for gated HuggingFace repos).
if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

exec uv run --project . python src/benchmark_performance.py "$@"
