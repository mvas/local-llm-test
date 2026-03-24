#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is required but not found in PATH" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -d .venv-bfcl ]]; then
  echo "Using existing virtual environment at: .venv-bfcl"
else
  echo "Creating virtual environment at: .venv-bfcl"
  uv venv .venv-bfcl
fi
echo "Installing latest bfcl-eval"
uv pip install --python .venv-bfcl/bin/python bfcl-eval soundfile

echo
echo "BFCL environment is ready."
echo "  Env path: .venv-bfcl"
echo "  BFCL CLI: .venv-bfcl/bin/bfcl"
echo
echo "Example:"
echo "  BFCL_PROJECT_ROOT={suite_dir} LOCAL_SERVER_ENDPOINT={host} LOCAL_SERVER_PORT={port} \\"
echo "  .venv-bfcl/bin/bfcl generate --model <BFCL_MODEL_ID> --skip-server-setup"
