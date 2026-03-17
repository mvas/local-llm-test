GSM8K - for reasoning	            
MMLU  - for knowledge	            
IFEval- for instruction following	
HumanEval - for coding	                
MBPP  - for coding reliability	    
Aider - for code editing	        
BFCL  - for tool calling	        
llama.cpp - for speed	                



These are rough estimates for a **14B Q4_K_M model on M4 Pro 48GB** running via llama.cpp. Actual times vary with model size, generation length, and backend tuning.


| Benchmark | Samples | Runtime estimate | Notes |
|---|---|---|---|
| **GSM8K** | ~1,319 | 1.5 – 3 hours | 5-shot CoT, each answer is 200-400 tokens |
| **MMLU** | ~14,000 | 1 – 3 hours | Multiple-choice, generation is short (one letter), but prompts are long with 5-shot |
| **MMLU Pro** | ~12,000 | 1.5 – 3 hours | 10-choice instead of 4, slightly harder and longer prompts |
| **IFEval** | ~540 | 30 – 60 min | Full-length generations checked for constraint adherence |
| **EvalPlus** | ~560 | 1 – 2 hours | HumanEval+ (164) + MBPP+ (399), includes code execution overhead |
| **Aider subset** | ~50–100 | 30 – 60 min | Python exercises only |
| **Aider full** | ~225+ | 2 – 4 hours | All languages in polyglot suite |
| **BFCL subset** | ~200–400 | 15 – 30 min | Simple + multiple function calling categories |
| **BFCL full** | ~2,000 | 1 – 2 hours | All categories including parallel, nested, multi-language |
| **TruthfulQA** | ~817 | 20 – 40 min | Multiple-choice scoring, short generations |
| **BBH** | ~6,500 | 2 – 4 hours | 23 tasks, 3-shot CoT; a subset of tasks cuts this significantly |
| **LiveCodeBench** | ~400+ | 2 – 4 hours | Code generation + execution, newer/harder problems |


**Scaling rules of thumb:**
- **7B model**: roughly half the times above
- **32B model**: roughly 2–3x the times above
- Batch size and prompt caching in your backend can help significantly with multiple-choice benchmarks (MMLU, TruthfulQA)
