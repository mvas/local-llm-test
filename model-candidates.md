## Models to Compare

Use three model size classes. Compare one or two candidates per class.

### Fast class

Use for chat responsiveness and agent loops.

Suggested range: from `7B` up to `10B` size.

### Balanced class

Likely best tradeoff on this machine.

Suggested range: from `10B` to `27B` size.

### Strong class

Use only if latency is still acceptable.

Suggested range: from `27B` to `34B`

Avoid making `70B` the center of the plan. It may run in aggressive quantization, but it is often too slow for coding-agent workflows on a laptop.

## Suggested options

Research date: 2026-03-17.
Sources used: Aider polyglot leaderboard, Aider code editing (Python) leaderboard,
HuggingFace Open LLM Leaderboard 2, AwesomeAgents open-source leaderboard,
InsiderLLM M4 Mac guide, Ministral 3 HuggingFace model card, Mistral AI announcement,
Qwen3 technical report, model release announcements.

---

### Note on Instruct vs Reasoning variants

Several models in this list come in two flavours: a standard Instruct variant and
a Reasoning (thinking) variant. They require different benchmark strategies.

**Use Instruct (non-thinking) for:**

- All speed benchmarks: TTFT, prompt tok/s, decode tok/s
- Agent loop tests: Aider, BFCL
- Coding benchmarks: EvalPlus, HumanEval+
- IFEval instruction following

**Use Reasoning (thinking) for quality-only benchmarks where latency is not measured:**

- GSM8K, MATH, MMLU, MMLU-Pro
- GPQA
- AIME-style math/reasoning suites

**Should you test both variants of the same model?**
One run with Instruct is the primary target — it reflects what you would actually
deploy. One run with Reasoning gives you the quality ceiling at the same parameter
count. That is a useful comparison but doubles the work. Suggested approach: run
Instruct for everything first. Add Reasoning runs only for models where quality on
math/reasoning benchmarks is the deciding factor (Phase 2).

**Model-specific behaviour:**


| Model                              | Thinking mode                                                                | Notes                                                                                                 |
| ---------------------------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Qwen3 8B / 14B / 32B               | Toggle via `/think` or `/no_think` in prompt, or `thinking_budget` API param | Single checkpoint. No separate download.                                                              |
| Qwen3.5 9B / 27B / 35B-A3B         | Toggle via `chat_template_kwargs: {"enable_thinking": false}` API param      | Single checkpoint. Does NOT support the `/think` `/nothink` prompt switch that Qwen3 uses.            |
| Ministral 3 8B / 14B               | Separate Instruct and Reasoning checkpoints                                  | Download Instruct as primary. Reasoning is a separate model pull.                                     |
| Phi-4-reasoning-plus               | Always on — cannot be disabled                                               | Thinking tokens are always generated. Speed benchmarks will reflect the mandatory reasoning overhead. |
| DeepSeek-R1-Distill-Qwen-14B / 32B | Always on — cannot be disabled                                               | Based on R1 training, thinking tokens are always emitted. Same caveat as Phi-4-reasoning-plus.        |


---

### Fast class candidates

Memory at Q4_K_M on M4 Pro: roughly 4–6 GB model weight, leaving headroom for
long context KV caches. Expected generation speed: 70–120 tok/s.


| Model                     | Size | Why include                                                                                                                                                                                                                                                                                                                                                              |
| ------------------------- | ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Qwen2.5-Coder-7B-Instruct | 7B   | 57.9 % on Aider Python benchmark (via API); 63.9 % at Q8_0 via Ollama. Highest-scoring open-source model locally runnable at 7B for coding. Aider edit leaderboard.                                                                                                                                                                                                      |
| Llama 3.1 8B Instruct     | 8B   | 37.6 % Aider Python. Widely used reference baseline. Important for cross-comparison since most public hardware benchmarks include this model. Meta open release.                                                                                                                                                                                                         |
| Gemma 3 4B                | 4B   | Below the suggested range, but Google's technical report says it outperforms Gemma 2 27B on standard benchmarks. Useful as an extreme-speed data point for agent loops. InsiderLLM Gemma guide.                                                                                                                                                                          |
| Qwen3.5-9B                | 9B   | Released Feb 2026. MMLU-Pro 82.5, GPQA Diamond 81.7, LiveCodeBench v6 65.6 %, BFCL-V4 66.1 % — a 9B model outperforming Qwen3-30B-A3B on MMLU-Pro and BFCL. Thinking mode toggleable via API. 262K context. No Aider score yet; keep Qwen2.5-Coder-7B as the coding baseline. ⚠️ Gated DeltaNet architecture — see ref below this table. Qwen3.5 HuggingFace model card. |
| Ministral 3 8B Instruct   | 8B   | Arena Hard 0.509, WildBench 66.8, MATH 0.876 (Mistral model card, Dec 2025). Competes directly with Qwen3-8B on instruct tasks. Apache 2.0. 256K context, native function calling. Different training lineage from all other Fast class candidates. Mistral AI announcement.                                                                                            |


Yi-Coder 9B was considered and rejected: qualitative tests and Aider Python score
(54.1 %) both show it is weaker than Qwen2.5-Coder-7B (57.9 %) despite being larger.
Llama 3.2 11B was rejected: it is a vision/multimodal model with no useful coding
benchmark data.
Qwen3-8B is not a primary candidate — use as the llama.cpp fallback for Qwen3.5-9B
if the Gated DeltaNet architecture is unsupported. Official GGUF at
Qwen/Qwen3-8B-GGUF; Q4_K_M 5.03 GB, Q8_0 8.71 GB.

---

### Balanced class candidates

Memory at Q4_K_M: roughly 9–16 GB model weight. Expected generation speed:
20–55 tok/s depending on exact size. This is the sweet spot on 48 GB.


| Model                        | Size | Why include                                                                                                                                                                                                                                                                                                                                                    |
| ---------------------------- | ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Qwen2.5-Coder-14B-Instruct   | 14B  | 69.2 % Aider Python via API; 61.7 % via Ollama local. Best locally runnable coding model at 14B. Aider edit leaderboard.                                                                                                                                                                                                                                       |
| Phi-4-reasoning-plus 14B     | 14B  | Microsoft. Thinking always on — cannot be disabled. Outperforms DeepSeek-R1-Distill-Llama-70B (5× larger) on reasoning benchmarks. Trained on high-quality synthetic data + reinforcement learning. Expect slow TTFT. Microsoft AI blog, Unsloth Phi-4 docs.                                                                                                   |
| DeepSeek-R1-Distill-Qwen-14B | 14B  | AIME 2024 pass@1 69.7; MATH-500 93.9. Thinking always on. Reasoning distillation of R1 into a 14B Qwen2.5 shell. Comparison point: distilled reasoning vs non-distilled at identical size. DeepSeek paper Jan 2025.                                                                                                                                            |
| Gemma 3 12B                  | 12B  | 128 K context. Google's mid-size model (March 2025). Different training lineage from all other Balanced candidates. InsiderLLM Gemma guide.                                                                                                                                                                                                                    |
| Devstral-Small-2507          | 24B  | #1 open-source model on SWE-bench (Jul 2025). Finetuned from Mistral-Small-3.1 for agentic coding tasks, built with All Hands AI. Text-only (vision encoder removed). No reasoning overhead — pure instruct. 128K context. Q4_K_M 14.3 GB. Official GGUF works directly with llama.cpp. Apache 2.0. mistralai/Devstral-Small-2507_gguf HuggingFace model card. |
| Qwen3 14B Instruct           | 14B  | Next-generation Qwen (May 2025). Thinking mode toggleable. Beats Qwen2.5-14B across all benchmarks. Provides generational comparison: 2025 Qwen base vs 2024 Qwen Coder. Qwen3 technical report.                                                                                                                                                               |
| Ministral 3 14B Instruct     | 14B  | Arena Hard 0.551, WildBench 68.5, MATH 0.904 (Dec 2025). Beats Qwen3-14B Non-Thinking on all three metrics. AIME25 85 % with Reasoning variant. Apache 2.0. 256K context. Mistral claims performance comparable to Mistral Small 3.2 24B at 14B. Ministral 3 HuggingFace model card.                                                                           |


If time is limited, prioritise:
Qwen2.5-Coder-14B → Ministral 3 14B → Qwen3 14B → Phi-4-reasoning-plus → DeepSeek-R1-Distill-Qwen-14B.
The others fill architecture/lineage gaps.



Qwen2.5-14B-Instruct - dropped in favor of Qwen3 14B Instruct.

---

### Strong class candidates

Memory at Q4_K_M: roughly 20–22 GB model weight. Comfortable on 48 GB even at
Q8_0 (~34 GB). Expected generation speed: 15–25 tok/s.


| Model                        | Size    | Why include                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ---------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Qwen2.5-Coder-32B-Instruct   | 32B     | 71.4 % Aider Python benchmark (via API); 72.9 % via Ollama (whole format). Highest-scoring open-source model on the Aider code editing leaderboard overall. The current best reference for local coding. Aider edit leaderboard.                                                                                                                                                                                                                                                                 |
| Qwen3 32B                    | 32B     | 40.0 % Aider polyglot leaderboard (new, harder, multi-language benchmark). Best locally runnable dense model on the polyglot test. MMLU-Pro 73.5 %, Chatbot Arena ELO 1238. Thinking mode toggleable. AwesomeAgents leaderboard, Aider polyglot leaderboard.                                                                                                                                                                                                                                     |
| DeepSeek-R1-Distill-Qwen-32B | 32B     | AIME 2024 72.6, MATH-500 94.3, Codeforces rating 1691. Thinking always on. Outperforms OpenAI o1-mini on several benchmarks. DeepSeek paper Jan 2025. Community Aider test: ~15–16 % polyglot locally at Q4_K_M.                                                                                                                                                                                                                                                                                 |
| Qwen3.5-27B                  | 27B     | Released Feb 2026. MMLU-Pro 86.1, GPQA Diamond 85.5, IFEval 95.0, LiveCodeBench v6 80.7 %, SWE-bench Verified 72.4 %, BFCL-V4 68.5. Outperforms every other Strong class candidate on all benchmarks. ~17 GB at Q4_K_M. ⚠️ Gated DeltaNet architecture — if llama.cpp does not support it, use MLX or vLLM. **llama.cpp fallback: Qwen3-32B** (confirmed llama.cpp + GGUF, thinking toggleable, 40.0 % Aider polyglot). Qwen3.5 HuggingFace model card.                                          |
| Qwen3.5-35B-A3B              | 35B MoE | Released Feb 2026. MMLU-Pro 85.3, LiveCodeBench v6 74.6 %, SWE-bench Verified 69.2 %, BFCL-V4 67.3. MoE: 35B total params (~22 GB at Q4_K_M), only 3B active — very fast generation. Supersedes Qwen3-30B-A3B on every benchmark (+10 pp LiveCodeBench). ⚠️ Gated DeltaNet + MoE architecture — if llama.cpp does not support it, use MLX or vLLM. **llama.cpp fallback: Qwen3-30B-A3B-GGUF** (official GGUF, confirmed Metal/llama.cpp support, same MoE slot). Qwen3.5 HuggingFace model card. |


Note on Qwen3.5 architecture and inference engine selection:
Qwen3.5-9B, Qwen3.5-27B, and Qwen3.5-35B-A3B use a Gated DeltaNet hybrid
architecture (not a standard transformer). Before running, check whether
llama.cpp has added a qwen3_5 backend (check llama.cpp releases or
gguf-my-repo listings on HuggingFace).

Inference engine options in priority order for M4 Pro:

1. llama.cpp — preferred (used for all other models; enables apples-to-apples

speed comparison). Use only if qwen3_5 GGUF support is confirmed.
2. MLX / mlx_lm — best Apple Silicon alternative. Install: pip install mlx-lm.
Run: mlx_lm.generate --model mlx-community/Qwen3.5-9B-4bit.
Qwen3.5 MLX weights are typically on mlx-community within days of release.
3. vLLM — most complete API compatibility, but requires a separate process and
is less convenient for quick benchmarking.

Speed numbers from different engines are NOT directly comparable. If Qwen3.5
ends up on MLX, run Qwen3-32B on both engines to get a rough calibration
factor and annotate results with the engine used.

Note on Qwen2.5-Coder-32B polyglot score: it scores only 16.4 % on the Aider
polyglot leaderboard versus 71.4 % on the older Python-only benchmark. The
polyglot test is substantially harder (6 languages, 225 exercises vs 133 Python
only). The drop is partly explained by the "whole" edit format used locally,
which is less efficient than diff-based formats. Do not discard this model on
the basis of the polyglot number alone — run both benchmarks and compare.

---

### Reach candidate

This model exceeds the stated 32B ceiling but fits in 48 GB at Q4_K_M.
Include only if you want to know how much headroom a 70B model buys and if
agent-loop latency is acceptable.


| Model                  | Size | Why include                                                                                                                                                                                                              |
| ---------------------- | ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Llama 3.3 70B Instruct | 70B  | 59.4 % Aider Python benchmark. Q4_K_M fits in ~42 GB, leaving ~6 GB for KV cache. Generation speed roughly 7–12 tok/s on M4 Pro — borderline for agent loops. Important as a ceiling data point. Aider edit leaderboard. |


---

### Models ruled out (with reasons)


| Model                                 | Reason                                                                                                                                                                                                   |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Gemma 3 27B                           | 4.9 % on Aider polyglot; ranks #100/123 on BenchLM general leaderboard. Not competitive for coding agent use.                                                                                            |
| Llama 4 Maverick                      | 15.6 % Aider polyglot. MoE model, 128 experts total (400B params, 17B active). Requires multi-GPU for full inference. Not suitable as a primary local candidate.                                         |
| Qwen3-Coder-Next (80B MoE, 3B active) | Theoretically fits at ~46 GB Q4, but leaves almost nothing for KV cache at 256K context. Very new; llama.cpp GGUF support needs verification. Revisit if memory budget increases or MLX support matures. |
| DeepSeek V3 / R1 (671B)               | Too large. Would require 4–8× 80 GB GPUs. No local path on this machine.                                                                                                                                 |
| Codestral 22B                         | 48.1 % Aider Python, 11.1 % Aider polyglot. Weaker than Qwen2.5-Coder-14B at a larger size. Poor return on memory investment.                                                                            |
| Mistral Small 3.1 24B                 | Replaced by Devstral-Small-2507 (same base, purpose-built for coding agents, #1 SWE-bench). Ministral 3 14B also likely outperforms it on general benchmarks at half the size and 40 % faster.           |
| Qwen3-30B-A3B Coder                   | Not a primary candidate — use as llama.cpp fallback for Qwen3.5-35B-A3B if the Gated DeltaNet architecture is unsupported. Official GGUF available; confirmed Metal backend on Apple Silicon.            |
| Qwen3-8B                              | Not a primary candidate — use as llama.cpp fallback for Qwen3.5-9B if the Gated DeltaNet architecture is unsupported. Official GGUF at Qwen/Qwen3-8B-GGUF; Q4_K_M 5.03 GB, Q8_0 8.71 GB.                 |


