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
| `--reasoning-budget` | — | Token budget for thinking (see [Reasoning models](#reasoning-models)) |
| `--reasoning` | `auto` | Enable/disable thinking: `on`, `off`, `auto` |

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

### Reasoning models

Thinking/reasoning models (Qwen3.5, DeepSeek-R1, QwQ, etc.) generate hidden chain-of-thought tokens before producing visible output. On local inference this can mean 10,000–20,000+ thinking tokens per request, making benchmarks orders of magnitude slower and often causing them to appear stuck.

**`--reasoning-budget`** is the recommended first lever. It caps thinking tokens at the server level while preserving the model's ability to reason:

| Flag | Effect |
|---|---|
| `--reasoning-budget 4096` | Allow up to 4096 thinking tokens per response (recommended for 7B–14B) |
| `--reasoning-budget 0` | Suppress thinking entirely (thinking tag immediately closed) |
| (not set, default) | Unrestricted — model decides how much to think |

**`--reasoning off`** completely disables the thinking chat-template path. Useful when the model's thinking mode is causing format issues or when you want pure non-reasoning behavior.

**Example — 9B reasoning model on Aider:**

```bash
uv run src/benchmark_performance.py models/models1.txt \
  --run-aider --ctx 32768 \
  --reasoning-budget 4096 \
  --n-predict 8192 \
  --aider-litellm-timeout 300
```

### `--n-predict` — capping total generation length

`--n-predict N` sets a hard server-side cap on **all** tokens generated per request, including both thinking and visible output (`-1` = unlimited, the default). This is a broader alternative to `--reasoning-budget` that also limits non-thinking output.

**Recommended values:**

| Model type | Suggested value | Notes |
|---|---|---|
| Reasoning (7B–14B) | `8192`–`16384` | Use alongside or instead of `--reasoning-budget` |
| Reasoning (32B–70B) | `16384`–`32768` | More headroom for complex problems |
| Non-reasoning | `-1` (default) | No cap needed; rely on `--aider-timeout` instead |

### `--aider-litellm-timeout` — per-request API timeout

`LITELLM_REQUEST_TIMEOUT` controls how long the litellm HTTP client inside the Aider container waits for a single API response before giving up. When it fires, llama-server cancels the in-progress generation and the same request is retried from scratch.

`--aider-litellm-timeout N` sets this value in seconds (default: `300`). It should be large enough for a full capped generation to complete:

```
estimated_timeout ≈ max_tokens ÷ tokens_per_second  (with some margin)
```

For example, with `--n-predict 8192` on a 9B model at ~30 tok/s: `8192 ÷ 30 ≈ 273s` → `300s` default provides sufficient margin.

With `LITELLM_NUM_RETRIES=1` (hardcoded), worst-case time per exercise is `2 × timeout`. Setting a very high timeout without `--reasoning-budget` or `--n-predict` is not recommended.

**Example — small reasoning model, fully tuned:**

```bash
uv run src/benchmark_performance.py models/models1.txt \
  --run-aider \
  --ctx 32768 \
  --reasoning-budget 4096 \
  --n-predict 8192 \
  --aider-litellm-timeout 300
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

