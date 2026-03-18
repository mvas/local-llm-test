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

- raw model quality (reasoning, knowledge, instruction following)
- coding quality (generation and editing)
- tool calling quality
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
  - For general-purpose quality benchmarks (`GSM8K`, `MMLU`, `IFEval`)
- `EvalPlus`
  - For coding benchmarks with stronger tests than plain HumanEval (`HumanEval+`, `MBPP+`)
- `Aider`
  - For code editing evaluation across multiple languages
- `BFCL`
  - For tool/function calling evaluation

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

### Step 1: Raw speed with llama.cpp

Run `llama-bench` for each model with the same settings. Rule out models that are too slow

Capture:

- prompt speed
- generation speed
- batch sensitivity if you test it


### Step 2: General-quality suite

Run `lm-evaluation-harness`, using limited subset of examples, for limited task set:

- `GSM8K`, math and chain-of-thought reasoning
- `MMLU`, Log-likelihood, knowledge benchmark
- `IFEval`, instruction following

Older benchmarks like `HellaSwag` and `Winogrande` are largely saturated by modern instruction-tuned models at 7B+ and no longer discriminate well between candidates.


### Step 3: Specific tests

- `EvalPlus HumanEval+` - code generation
- `Aider` subset, multi-turn code generation
- `BFCL` subset for tool calling


## Phase 2: Deep Benchmark (overnight or weekend)

Goal: compare the top 2 or 3 model configurations in more realistic settings.

### Step 1: General-purpose evaluation

Expand `lm-evaluation-harness` to include:

- full `GSM8K`: math and chain-of-thought reasoning
- full `IFEval` (instruction following)
- `mmlu_pro` (log-likelihood, general knowledge)
- `truthfulqa` (log-likelihood, hallucination resistance)
- `bbh`


### Step 2: Specific benchmarks

- `EvalPlus MBPP+`
- full `Aider` (code editing)
- `LiveCodeBench` for broader and newer coding tasks
- Full `BFCL` (tool calling)
- a private prompt pack


## Decision Rules

Use these rules to choose winners.

### Best daily general assistant

Optimize for:

- good `IFEval`
- good `MMLU`
- good `GSM8K`
- low latency

### Best coding assistant

Optimize for:

- strongest `EvalPlus`
- strongest `Aider` score
- acceptable latency for iterative prompting

### Best coding agent

Optimize for:

- `Aider` code editing success rate
- `BFCL` tool calling accuracy


The best coding assistant and the best coding agent may not be the same model.

### Keep for interactive use

- `TTFT` feels responsive
- generation speed is comfortable for multi-turn chat
- no memory pressure issues at your target context

### Keep for agent use

- latency remains acceptable over repeated tool loops
- long prompts do not collapse performance too much
- failure rate is not dominated by timeouts

As a rule of thumb, agent workflows punish slow models more than static benchmarks do.


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


## Recommended Next Step

Implement the benchmark in two scripts:

- `quick-benchmark.sh`
- `deep-benchmark.sh`

And store outputs in:

- `results/speed/`
- `results/quality/`
- `results/agent/`

That keeps the process repeatable and makes it easy to add new models later.
