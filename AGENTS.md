# AGENTS.md

## Project Purpose

`local-llm` runs repeatable LLM benchmarks (quality + speed) locally against GGUF models.
The engine to host inference is `llama.cpp`. Results are written to `results/`.

Core workflow:
- Start/stop local inference server per model safely.
- Run benchmark suites (LM-Eval, EvalPlus, BFCL, optional templated suites).
- Persist machine-readable CSV summaries and raw logs for each run.

## Setup

1. `uv` is used as environment and package manager
2. BFCL requires separate virtual environment because of conflicts (set up script `scripts/setup_bfcl_env.sh`)
3. Entrypoints:
   - `src/benchmark_performance.py` (quality pipeline)
   - `src/benchmark_speed.py` (throughput + latency pipeline)

## Code Map (Where Things Live)

- `src/benchmark_performance.py`
  - Orchestrates per-model quality runs.
  - Starts `llama-server`, executes suites, writes `summary.csv`, `suite_runs.csv`, `metrics.csv`.
- `src/benchmark_speed.py`
  - Runs `llama-bench` and streaming latency measurements.
  - Writes speed-focused `summary.csv` and `latency_by_prompt.csv`.
- `src/common.py`
  - Shared dataclasses, command constants, server lifecycle, CSV helpers, errors.
- `src/suite_common.py`
  - Common subprocess logging and metric selection helpers.
- `src/suite_lm_eval.py`
  - LM-Eval suites (GSM8K, MMLU, IfEval): execution, parsing outputs.
- `src/suite_evalplus.py`
  - EvalPlus suites (HumanEval, MBPP): subset creation, execution, parsing outputs.
- `src/suite_bfcl.py`
  - BFCL benchmark: mapping resolution, subset generation, generate/evaluate execution, score parsing.
- `src/suite_templated.py`
  - Generic shell-template suite integration.
- `models/`
  - `models.txt`: example models list. Such file is input to the benchmarking script.
  - `tokenizers.txt`: example of tokenizers mapping, needed for LM-Eval suites.
  - `bfcl-model-ids.txt`: example of BFCL model-id mapping, needed for BFCL benchmark.

## Rules and patterns

- Use `BenchmarkError` for expected operational failures; avoid ad-hoc exceptions for user-facing run failures.
- Keep runs deterministic by preserving/propagating defaults (`seed`, `temp`, `top_p`, limits).
- Write suite logs to per-suite files inside each model raw directory.
- Emit structured metrics via `Metric` dataclass rows, then append to CSV.
- Ensure server lifecycle is symmetric: start once per model, always cleanup in `finally`.
- Prefer explicit paths and resolved model paths (see `expand_model_path`).
- Prefer reasonable defaults to argument(s) in entrypoint parser. Use constants instead of hardcoded literals.
- Keep subprocess execution through `run_logged_command`.
- Keep outputs in the established run directory shape.
- Return `(primary_name, primary_value, metric_rows, runtime_s)` pattern for suite integrations.
- Keep docs updated - at minimum, update `README.md` for user-facing invocation changes.
- Do not hardcode machine-specific model paths beyond existing `models/*.txt` conventions.
- Do not bypass `.venv-bfcl`; BFCL is intentionally isolated due to dependency conflicts.
- Keep timeouts explicit and centralized (`SUITE_TIMEOUT` or suite-specific config).
- Preserve backwards compatibility for existing flags unless intentionally deprecating.
- Prefer additive changes over silent behavior changes.

## Examples of commands

LM_eval benchmarks:
`sh benchmark-perf.sh models/models1.txt -t models/tokenizers.txt --run-lm-evals`

HumanEval benchmarks:

`sh benchmark-perf.sh models/models1.txt -t models/tokenizers.txt --run-humaneval --humaneval-limit 5`

`sh benchmark-perf.sh models/models1.txt -t models/tokenizers.txt --run-mbpp --mbpp-limit 5`

Aider benchmark (reasoning model):
`sh benchmark-perf.sh models/models1.txt --run-aider --ctx 32768 --reasoning-budget 4096 --n-predict 8192 --aider-litellm-timeout 300`

BFCL benchmark:
`sh benchmark-perf.sh models/models1.txt --run-bfcl --bfcl-limit 2 --bfcl-model-id-map-file models/bfcl-model-ids.txt`

Speed benchmark (example):
- `sh benchmark-speed.sh models/models1.txt models/prompts.txt`

## Results Contract

Quality runs create:
- `results/performance/<timestamp>/summary.csv`
- `results/performance/<timestamp>/suite_runs.csv`
- `results/performance/<timestamp>/metrics.csv`
- `results/performance/<timestamp>/raw/<model_slug>/<suite>/...logs...`

Speed runs create:
- `results/speed/<timestamp>/summary.csv`
- `results/speed/<timestamp>/latency_by_prompt.csv`
- `results/speed/<timestamp>/raw/<model_slug>/...logs...`
