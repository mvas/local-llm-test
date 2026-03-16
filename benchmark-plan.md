# Local LLM Benchmark Plan for M4 Pro 48GB

## Goal

Build a benchmark workflow for local LLMs that answers two questions:

1. How well does the model work?
2. How fast does the model work on this machine?

This plan is optimized for:

- Apple Silicon: `M4 Pro`
- Unified memory: `48GB`
- Local inference only
- Two use cases:
  - general-purpose assistant / reasoning
  - coding assistant / coding agent

The plan is split into a fast first pass and a deeper evaluation pass.

## Benchmark Philosophy

Do not use one benchmark for everything. Separate evaluation into:

- raw model quality
- coding quality
- agent quality
- inference speed

A model can score well on static QA benchmarks and still be too slow for agent loops. A smaller model can score slightly worse but deliver a better real-world experience because it is much faster and more responsive.

Keep comparisons fair by fixing:

- inference backend
- prompt template / chat format
- temperature
- max output tokens
- context length
- quantization family

If any of those change, treat the result as a different configuration.

## Recommended Tool Stack

Use these tools as the default benchmark stack:

- `llama.cpp`
  - For local inference and raw speed tests
  - Use `llama-bench` for throughput benchmarking
- `lm-evaluation-harness`
  - For general-purpose quality benchmarks
- `EvalPlus`
  - For coding benchmarks with stronger tests than plain HumanEval
- `SWE-bench Verified`
  - For coding-agent evaluation
- Optional: `Terminal-Bench`
  - For terminal-oriented agents
- Optional: `GAIA`
  - For general-purpose tool-using agents

## Runtime Recommendation

Start with one runtime and keep it fixed for the first comparison round.

Preferred order:

1. `llama.cpp`
2. `MLX`
3. `Ollama` only if convenience matters more than measurement purity

Reasoning:

- `llama.cpp` gives the cleanest raw benchmarking and straightforward throughput metrics.
- `MLX` is worth checking later because Apple Silicon performance can be very strong.
- `Ollama` is convenient, but it adds another layer that can obscure exact runtime comparisons.

## Models to Compare

Use three model size classes. Compare one or two candidates per class.

### Fast class

Use for chat responsiveness and agent loops.

Suggested range:

- `7B`
- `8B`

Example candidates:

- `Qwen2.5-Coder 7B`
- `Qwen2.5 7B Instruct`
- `Llama 3.1 8B Instruct`

### Balanced class

Likely best tradeoff on this machine.

Suggested range:

- `14B`
- `16B`
- `20B`

Example candidates:

- `Qwen2.5-Coder 14B`
- `Qwen2.5 14B Instruct`
- `DeepSeek-Coder v2 Lite`
- `Mistral Small` class models if available in your preferred format

### Strong class

Use only if latency is still acceptable.

Suggested range:

- `32B`

Example candidates:

- `Qwen2.5-Coder 32B`
- `QwQ 32B` for reasoning experiments

Avoid making `70B` the center of the plan. It may run in aggressive quantization, but it is often too slow for coding-agent workflows on a laptop.

## Quantization Strategy

For each model, test only one or two quantization points at first.

Suggested starting points:

- smaller models: `Q4_K_M` or similar practical 4-bit quant
- larger models: one lower-memory quant and one quality-oriented quant if feasible

Do not compare:

- one model in a strong 6-bit quant
- another in a weak 3-bit quant

unless the goal is explicitly "best quality under memory limit."

## Metrics to Record

Record all results in one table or CSV.

### Speed metrics

- time to first token (`TTFT`)
- prompt processing speed (`prompt tok/s`)
- generation speed (`decode tok/s`)
- end-to-end latency for fixed prompts
- peak memory usage
- tokens generated before slowdown or failure at larger context sizes

### Quality metrics

- benchmark score per suite
- pass rate
- exact match / accuracy where applicable
- failure mode notes

### Agent metrics

- success rate
- median time per task
- average steps or tool calls
- timeout rate
- common failure reason:
  - reasoning failure
  - bad edit
  - test failure
  - timeout
  - tool misuse

## Phase 1: Quick Benchmark (about 2 hours)

Goal: identify which models deserve deeper testing.

Test 3 to 5 model configurations total.

Recommended matrix:

- 1 fast model
- 2 balanced models
- optional 1 strong model

Example:

- `Qwen2.5 7B Instruct`
- `Qwen2.5-Coder 14B`
- `Qwen2.5 14B Instruct`
- `Qwen2.5-Coder 32B` if latency is acceptable

### Step 1: Raw speed with llama.cpp

Run `llama-bench` for each model with the same settings.

Capture:

- prompt speed
- generation speed
- batch sensitivity if you test it

Primary outcome:

- rule out models that are too slow before running heavier quality suites

### Step 2: Small general-quality suite

Run `lm-evaluation-harness` on a compact task set:

- `hellaswag`
- `arc_challenge`
- `winogrande`
- `gsm8k`
- `ifeval`

Why this set:

- `HellaSwag`: common-sense completion
- `ARC-Challenge`: harder multiple-choice reasoning
- `Winogrande`: pronoun/common-sense reasoning
- `GSM8K`: math and chain-of-thought-adjacent reasoning
- `IFEval`: instruction following

If runtime is tight, use a small sample count first.

### Step 3: Small coding suite

Run:

- `EvalPlus HumanEval+`
- `EvalPlus MBPP+`

Primary outcome:

- identify which model is best at code generation under local constraints

### Step 4: Manual prompt smoke test

Use a small fixed prompt pack of 10 to 20 prompts that reflect real usage:

- explain an unfamiliar code snippet
- write a small utility function
- refactor a function for readability
- diagnose a failing test from error text
- summarize a long technical document
- generate a shell command from a task description

Record:

- usefulness
- hallucination rate
- verbosity fit
- whether responses are fast enough to feel interactive

This matters because benchmark leaders are not always the best day-to-day assistants.

## Phase 2: Deep Benchmark (overnight or weekend)

Goal: compare the top 2 or 3 model configurations in more realistic settings.

### General-purpose evaluation

Expand `lm-evaluation-harness` to include:

- `mmlu` or `mmlu_pro`
- `truthfulqa`
- `bbh` subset if you want harder reasoning

Do not run every possible benchmark. Use a curated set that covers:

- knowledge
- reasoning
- instruction following
- truthfulness

### Coding evaluation

Keep `EvalPlus`, then add one of:

- `LiveCodeBench` for broader and newer coding tasks
- a private prompt pack derived from your own work

The private prompt pack is often more valuable than another public benchmark if your goal is real productivity.

### Coding-agent evaluation

Run a small subset of `SWE-bench Verified`.

Suggested starting size:

- `25` tasks for the first pass
- `50` tasks if the setup is stable

Do not start with the full benchmark. It is heavy, slow, and operationally expensive.

Use it to answer:

- can the model navigate a repo?
- can it make correct edits?
- can it satisfy tests?
- can it recover from partial failure?

### Terminal-agent evaluation

If terminal tool use matters, add `Terminal-Bench`.

This is especially useful if you care about:

- shell-heavy workflows
- repo inspection
- build and test loops
- file manipulation through tools

### General-purpose agent evaluation

If you want broad tool-using behavior beyond coding, add `GAIA`.

Treat `GAIA` as optional. It is useful, but less clean for strict local apples-to-apples measurement because tasks can depend on external retrieval and environment details.

## Suggested Prompt Packs

Create three local prompt packs and keep them fixed for every run.

### Pack A: General assistant

Use 20 to 30 prompts:

- summarization
- factual QA
- planning
- extraction
- instruction following
- math and logic

### Pack B: Coding assistant

Use 20 to 30 prompts:

- write functions
- explain code
- repair bugs from stack traces
- generate tests
- refactor code
- regex and shell tasks

### Pack C: Agent tasks

Use 10 to 20 tasks:

- inspect repo structure
- identify bug source from logs
- patch a broken function
- update tests after a spec change
- trace a failing command

Score each item with a simple rubric:

- `0` = failed
- `1` = partly useful
- `2` = correct and useful

This gives a lightweight benchmark that reflects your real workflows better than public suites alone.

## Exact Evaluation Flow

For each model configuration:

1. Run `llama-bench`
2. Run a fixed latency test on 5 to 10 prompts
3. Run `lm-evaluation-harness` small suite
4. Run `EvalPlus`
5. Run prompt pack A and B
6. If promising, run agent subset:
   - `SWE-bench Verified`
   - optional `Terminal-Bench`

Stop early if a model is clearly too slow or weak.

## Decision Rules

Use these rules to choose winners.

### Best daily general assistant

Optimize for:

- good `IFEval`
- good `MMLU` / `ARC`
- low latency
- low hallucination rate in prompt pack A

### Best coding assistant

Optimize for:

- strongest `EvalPlus`
- good code explanation and bug-fix performance in prompt pack B
- acceptable latency for iterative prompting

### Best coding agent

Optimize for:

- `SWE-bench Verified` subset success rate
- median task completion time
- stability across multiple tasks

The best coding assistant and the best coding agent may not be the same model.

## Practical Thresholds for This Machine

Use these rough thresholds when deciding whether to keep a model in contention.

### Keep for interactive use

- `TTFT` feels responsive
- generation speed is comfortable for multi-turn chat
- no memory pressure issues at your target context

### Keep for agent use

- latency remains acceptable over repeated tool loops
- long prompts do not collapse performance too much
- failure rate is not dominated by timeouts

As a rule of thumb, agent workflows punish slow models more than static benchmarks do.

## Suggested First Comparison Set

Start with this:

- `Qwen2.5 7B Instruct`
- `Qwen2.5-Coder 14B`
- `Qwen2.5 14B Instruct`
- `Qwen2.5-Coder 32B` if it fits and remains usable

Alternative set if you want reasoning emphasis:

- `Llama 3.1 8B Instruct`
- `Qwen2.5 14B Instruct`
- `QwQ 32B`

## What to Skip Initially

Do not spend early time on:

- full `SWE-bench Verified`
- too many quantization variants
- too many runtimes at once
- giant context experiments before baseline results exist
- leaderboard chasing without testing your own prompt packs

## Deliverables

At the end of the benchmark, produce:

- one spreadsheet or CSV with all numeric metrics
- one short write-up per model:
  - best use case
  - main weakness
  - recommended runtime and quantization
- one final ranking:
  - best general assistant
  - best coding assistant
  - best coding agent
  - best speed / quality tradeoff

## Minimal Execution Plan

If you want the leanest possible version, do this:

1. Benchmark `3` models with `llama-bench`
2. Run `hellaswag`, `arc_challenge`, `gsm8k`, `ifeval`
3. Run `EvalPlus`
4. Run a `25`-prompt private prompt pack
5. Run `25` tasks from `SWE-bench Verified` on the top `2` models

This is the highest-signal low-overhead version of the plan.

## Recommended Next Step

Implement the benchmark in two scripts:

- `quick-benchmark.sh`
- `deep-benchmark.sh`

And store outputs in:

- `results/speed/`
- `results/quality/`
- `results/agent/`

That keeps the process repeatable and makes it easy to add new models later.
