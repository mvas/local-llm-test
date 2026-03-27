# Pre-requisites

[Huggingface CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)

[UV > 0.10.12](https://docs.astral.sh/uv/getting-started/installation/)

For Aider:
Docker


# Initialization
```bash
uv python install
uv sync
```

# BFCL benchmark

`bfcl-eval` currently conflicts with this project's EvalPlus dependency (`tree-sitter` version mismatch).
Use an isolated virtual environment for BFCL and call its executable directly from benchmark commands.

## 1) Create and install BFCL environment

```bash
# from repository root
scripts/setup_bfcl_env.sh
```

## 2) Prepare model files

- `models/models.txt` - list of models to run benchmark on.
- `models/bfcl-model-ids.txt` - mapping to BFCL internal model configurations

## 3) Run benchmark

```bash
uv run src/benchmark_performance.py models/models.txt \
  --run-bfcl \
  --bfcl-model-id-map-file models/bfcl-model-ids.txt \
  --bfcl-limit 20
```

## Notes

- `--bfcl-limit` is implemented via BFCL `--run-ids` (deterministic first N test IDs per selected category).
- Do not run `uv sync` inside `.venv-bfcl`; keep this environment independent.
- Keep `.venv-bfcl` out of version control.

# Aider Benchmark

The Aider suite runs inside the `aider-benchmark` Docker image to provide isolated environment for generated code. Models still to be served with local `llama-server`.

## 1) Clone code and build docker container

```bash
# from repository root
scripts/setup_aider_env.sh
```

## 2) Prepare model files

- `models/models.txt` - list of models to run benchmark on.

## 3) Run Aider benchmark

```bash
uv run src/benchmark_performance.py models/models.txt \
  --run-aider \
  --aider-limit 20
```

## Notes:
- `--full-mode` makes Aider run the full benchmark set (`--num-tests -1`).
