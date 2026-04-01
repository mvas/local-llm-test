This repository contains a scaffolding to run a series of benchmarks on LLMs, serving them on localhost.

## Pre-requisites

[Huggingface CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)

[UV > 0.10.12](https://docs.astral.sh/uv/getting-started/installation/)

For Aider:
Docker


## Initialization
```bash
uv python install
uv sync
```

## BFCL benchmark specifics

`bfcl-eval` currently conflicts with this project's EvalPlus dependency (`tree-sitter` version mismatch).
Use an isolated virtual environment for BFCL and call its executable directly from benchmark commands.

### 1) Create and install BFCL environment

```bash
# from repository root
scripts/setup_bfcl_env.sh
```

### 2) Prepare model files

- `models/bfcl-model-ids.txt` - mapping to BFCL internal model configurations

### 3) Additional options

`models/bfcl-model-ids.txt` is default file name. If using file located anywhere else or named differently - set `--bfcl-model-id-map-file`:

```bash
uv run src/benchmark_performance.py models/models.txt \
  --run-bfcl \
  --bfcl-model-id-map-file models/bfcl-model-ids.txt \
  --bfcl-limit 20
```

### Notes

- `--bfcl-limit` is implemented via BFCL `--run-ids` (deterministic first N test IDs per selected category).
- `--bfcl-timeout` sets the timeout in seconds for the generate step (default: 5400 / 1.5 hours). Reasoning models with long chain-of-thought outputs may need a higher value.
- Do not run `uv sync` inside `.venv-bfcl`; keep this environment independent.
- Keep `.venv-bfcl` out of version control.

## Aider benchmark specifics

The Aider suite runs inside the `aider-benchmark` Docker image to provide isolated environment for generated code. Models still to be served with local `llama-server`.

### 1) Clone code and build docker container

```bash
# from repository root
scripts/setup_aider_env.sh
```

### Notes:
- `--full-mode` makes Aider run the full benchmark set (`--num-tests -1`).
- `--aider-timeout` sets the timeout in seconds for the Docker benchmark run (default: 5400 / 1.5 hours). Increase for slow models or large test limits.

## Models

Select the models you want to benchmark, download gguf files for them and put their paths into models list file (example: `models/models.txt`). This file will be input for benchmarking.

Update (or create new) tokenizers map - `models/tokenizers.txt`. While it is optional, ?????


## Running

### Quality benchmarks

```bash
uv run src/benchmark_performance.py <models_file> [options]
```

`models_file` — path to a list of GGUF model paths (see `models/models.txt` for format).

**Suite flags** — at least one must be provided:

| Flag | Description | Default limit |
|---|---|---|
| `--run-lm-evals` | GSM8K, MMLU, IfEval via LM-Eval | default limits of 30,20,30 apply when `--full-mode` is not set |
| `--run-humaneval` | HumanEval+ via EvalPlus | `--humaneval-limit 40` |
| `--run-mbpp` | MBPP+ via EvalPlus | `--mbpp-limit 20` |
| `--run-bfcl` | Function-calling via BFCL | `--bfcl-limit 20` |
| `--run-aider` | Aider coding benchmark (Docker) | `--aider-limit 20` |

**Common options:**

| Option | Default | Description |
|---|---|---|
| `-t / --tokenizer-map-file` | — | `MODEL_PATH=TOKENIZER_ID` map; recommended for LM-Eval |
| `--ngl` | `99` | GPU layers to offload |
| `--ctx` | `8192` | Context length |
| `--full-mode` | off | Remove per-suite sample limits for LM eval and run full set of tests for BFCL and Aider |
| `--server-port` | `8082` | llama-server port |
| `--out-dir-base` | `results/performance` | Output directory root |

Results are written to `results/performance/<timestamp>/` as `summary.csv`, `suite_runs.csv`, and `metrics.csv`.

**Examples:**

```bash
# LM-Eval suites with tokenizer map
uv run src/benchmark_performance.py models/models1.txt -t models/tokenizers.txt --run-lm-evals

# HumanEval, 10 problems
uv run src/benchmark_performance.py models/models1.txt --run-humaneval --humaneval-limit 10

# BFCL, 20 problems
uv run src/benchmark_performance.py models/models1.txt --run-bfcl --bfcl-limit 20 --bfcl-model-id-map-file models/bfcl-model-ids.txt

# Full Aider run
uv run src/benchmark_performance.py models/models1.txt --run-aider --full-mode
```

---

## Tuning

### `--n-predict` — capping generation length

Reasoning/thinking models (e.g. DeepSeek-R1 distills) can produce extremely long chain-of-thought outputs — tens of thousands of tokens per exercise — which causes requests to time out before generation finishes, leading to silent hangs rather than failed tests.

`--n-predict N` sets a hard server-side cap on tokens generated per request (`-1` = unlimited, the default).

**Recommended values:**

| Model size | Suggested value | Notes |
|---|---|---|
| Small distilled (7B–14B) | `16384` | Covers most exercises; hard ones get truncated and fail quickly |
| Larger distilled (32B–70B) | `32768` | More headroom for complex problems |
| Frontier / non-reasoning | `-1` (default) | No cap needed; rely on `--aider-timeout` instead |

Example:

```bash
uv run src/benchmark_performance.py models/models1.txt --run-aider --n-predict 16384
```

---

### Speed benchmarks

```bash
uv run src/benchmark_speed.py <models_file> [prompts_file] [options]
```

`models_file` — path to a list of GGUF model paths.  
`prompts_file` — optional; prompts with `[short]`/`[medium]`/`[long]` sections (default: `models/prompts.txt`).

**Key options:**

| Option | Default | Description |
|---|---|---|
| `--ngl` | `99` | GPU layers to offload |
| `--ctx` | `8192` | Context length |
| `--lat-maxtokens` | `256` | Max tokens per latency request |
| `--lat_repeats` | `3` | Latency measurement repetitions |
| `--bench-input-tokens` | `512` | llama-bench input token count |
| `--bench-output-tokens` | `256` | llama-bench output token count |
| `--bench-repetitions` | `3` | llama-bench repetitions |
| `--server-port` | `8081` | llama-server port |
| `--out-dir-base` | `results/speed` | Output directory root |

Results are written to `results/speed/<timestamp>/` as `summary.csv` and `latency_by_prompt.csv`.

**Example:**

```bash
uv run src/benchmark_speed.py models/models1.txt models/prompts.txt
```

