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
InsiderLLM M4 Mac guide, model release announcements.

---

### Fast class candidates

Memory at Q4_K_M on M4 Pro: roughly 4–6 GB model weight, leaving headroom for
long context KV caches. Expected generation speed: 70–120 tok/s.

| Model | Size | Why include |
|---|---|---|
| Qwen2.5-Coder-7B-Instruct | 7B | 57.9 % on Aider Python benchmark (via API); 63.9 % at Q8_0 via Ollama. Highest-scoring open-source model locally runnable at 7B for coding. Aider edit leaderboard. |
| Llama 3.1 8B Instruct | 8B | 37.6 % Aider Python. Widely used reference baseline. Important for cross-comparison since most public hardware benchmarks include this model. Meta open release. |
| Gemma 3 4B | 4B | Below the suggested range, but Google's technical report says it outperforms Gemma 2 27B on standard benchmarks. Useful as an extreme-speed data point for agent loops. InsiderLLM Gemma guide. |

Rationale for stopping at three: the 7–9B space is dominated by Qwen2.5-Coder-7B
for coding tasks. A second coding model at this size (e.g., Yi-Coder-9B at 54 %)
does not add much signal. Llama 3.1 8B gives a neutral general-purpose reference.

---

### Balanced class candidates

Memory at Q4_K_M: roughly 9–16 GB model weight. Expected generation speed:
20–55 tok/s depending on exact size. This is the sweet spot on 48 GB.

| Model | Size | Why include |
|---|---|---|
| Qwen2.5-Coder-14B-Instruct | 14B | 69.2 % Aider Python via API; 61.7 % via Ollama local. Best locally runnable coding model at 14B. Aider edit leaderboard. |
| Qwen2.5-14B-Instruct | 14B | IFEval 81.43 on HuggingFace Open LLM Leaderboard 2. Strong instruction following at 14B. Good pairing with the Coder variant to isolate how much coding specialisation helps. |
| Phi-4 14B (or Phi-4-reasoning-plus 14B) | 14B | Microsoft. Outperforms DeepSeek-R1-Distill-Llama-70B (5× larger) on several reasoning benchmarks. Trained on high-quality synthetic data rather than internet crawl. Strong for math and chain-of-thought. Available on Ollama (Q4_K_M ~11 GB). Microsoft AI blog, Unsloth Phi-4 docs. |
| DeepSeek-R1-Distill-Qwen-14B | 14B | AIME 2024 pass@1 69.7; MATH-500 93.9. Reasoning-focused distillation of R1 into a 14B shell based on Qwen2.5. Interesting comparison point against non-distilled models of the same size. DeepSeek paper Jan 2025. |
| Gemma 3 12B | 12B | 128 K context window. Google's mid-size model from March 2025. Fills the 12B slot with a model from a different training lineage. InsiderLLM Gemma guide. |
| Mistral Small 3.1 24B | 24B | 128 K context. Multimodal. Mistral's latest small model (March 2025), claims to beat Gemma 3 and GPT-4o Mini across text, multilingual, and long-context benchmarks. Fits at Q4_K_M ~15 GB. Mistral AI announcement, HuggingFace model card. |

Six candidates is a lot for one class. If time is limited, prioritise:
Qwen2.5-Coder-14B → Phi-4 14B → DeepSeek-R1-Distill-Qwen-14B.
The others fill gaps but are lower priority.

---

### Strong class candidates

Memory at Q4_K_M: roughly 20–22 GB model weight. Comfortable on 48 GB even at
Q8_0 (~34 GB). Expected generation speed: 15–25 tok/s.

| Model | Size | Why include |
|---|---|---|
| Qwen2.5-Coder-32B-Instruct | 32B | 71.4 % Aider Python benchmark (via API); 72.9 % via Ollama (whole format). Highest-scoring open-source model on the Aider code editing leaderboard overall. The current best reference for local coding. Aider edit leaderboard. |
| Qwen3 32B | 32B | 40.0 % Aider polyglot leaderboard (new, harder, multi-language benchmark). Best locally runnable model on the polyglot test. MMLU-Pro 73.5 %, Chatbot Arena ELO 1238. AwesomeAgents leaderboard, Aider polyglot leaderboard. |
| DeepSeek-R1-Distill-Qwen-32B | 32B | AIME 2024 72.6, MATH-500 94.3, Codeforces rating 1691. Reasoning specialist. Outperforms OpenAI o1-mini on several benchmarks despite being locally runnable. DeepSeek paper Jan 2025. Community Aider test: ~15–16 % polyglot locally at Q4_K_M (hard benchmark, matches expectations for a reasoning model used with diff format). |

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

| Model | Size | Why include |
|---|---|---|
| Llama 3.3 70B Instruct | 70B | 59.4 % Aider Python benchmark. Q4_K_M fits in ~42 GB, leaving ~6 GB for KV cache. Generation speed roughly 7–12 tok/s on M4 Pro — borderline for agent loops. Important as a ceiling data point. Aider edit leaderboard. |

---

### Models ruled out (with reasons)

| Model | Reason |
|---|---|
| Gemma 3 27B | 4.9 % on Aider polyglot; ranks #100/123 on BenchLM general leaderboard. Not competitive for coding agent use. |
| Llama 4 Maverick | 15.6 % Aider polyglot. MoE model, 128 experts total (400B params, 17B active). Requires multi-GPU for full inference. Not suitable as a primary local candidate. |
| Qwen3-Coder-Next (80B MoE, 3B active) | Theoretically fits at ~46 GB Q4, but leaves almost nothing for KV cache at 256K context. Very new; llama.cpp GGUF support needs verification. Revisit if memory budget increases or MLX support matures. |
| DeepSeek V3 / R1 (671B) | Too large. Would require 4–8× 80 GB GPUs. No local path on this machine. |
| Codestral 22B | 48.1 % Aider Python, 11.1 % Aider polyglot. Weaker than Qwen2.5-Coder-14B at a larger size. Poor return on memory investment. |

