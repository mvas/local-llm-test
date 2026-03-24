# Pre-requisites

[Huggingface CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)

[UV > 0.10.12](https://docs.astral.sh/uv/getting-started/installation/)

# Initialization
```bash
uv python install
uv sync
```

# BFCL in Separate Environment

`bfcl-eval` currently conflicts with this project's EvalPlus dependency (`tree-sitter` version mismatch).
Use an isolated virtual environment for BFCL and call its executable directly from benchmark commands.

## 1) Create and install BFCL environment

```bash
# from repository root
scripts/setup_bfcl_env.sh
```

## 2) Configure BFCL output location

BFCL writes under its project root (`result/` and `score/`). For benchmark runs, point it to your
suite output directory:

```bash
export BFCL_PROJECT_ROOT=/absolute/path/to/run/bfcl
```

In benchmark templates, set `BFCL_PROJECT_ROOT` inline per model run so outputs are isolated.

## 3) Use llama-server endpoint from main benchmark run

When llama-server is already running from `src/benchmark_performance.py`, instruct BFCL to reuse it:

- `--skip-server-setup`
- `LOCAL_SERVER_ENDPOINT=<host>`
- `LOCAL_SERVER_PORT=<port>`

Optional for remote/custom endpoints:

- `REMOTE_OPENAI_BASE_URL=http://<host>:<port>/v1`
- `REMOTE_OPENAI_API_KEY=<token>`

## 4) Example BFCL command template (for external suite execution)

Use the BFCL executable from the isolated environment (preferred over shell activation):

```bash
BFCL_PROJECT_ROOT={suite_dir} \
LOCAL_SERVER_ENDPOINT={host} \
LOCAL_SERVER_PORT={port} \
.venv-bfcl/bin/bfcl generate \
  --model <BFCL_MODEL_ID> \
  --test-category simple,parallel,multiple \
  --skip-server-setup
```

Then evaluate:

```bash
BFCL_PROJECT_ROOT={suite_dir} \
.venv-bfcl/bin/bfcl evaluate \
  --model <BFCL_MODEL_ID> \
  --test-category simple,parallel,multiple
```

## 5) Run from benchmark script

```bash
uv run src/benchmark_performance.py models/models.txt \
  --run-bfcl \
  --bfcl-env-path .venv-bfcl \
  --bfcl-model-id meta-llama/Llama-3.1-8B-Instruct-FC \
  --bfcl-test-categories simple_python,parallel,multiple \
  --bfcl-limit 20
```

`--bfcl-limit` is implemented via BFCL `--run-ids` (deterministic first N test IDs per selected category).

## 6) Notes

- Do not run `uv sync` inside `.venv-bfcl`; keep this environment independent.
- Keep `.venv-bfcl` out of version control.
