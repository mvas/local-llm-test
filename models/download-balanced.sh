#!/bin/bash

# Qwen2.5-Coder-14B-Instruct
# https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct-GGUF
hf download Qwen/Qwen2.5-Coder-14B-Instruct-GGUF --dry-run
hf download Qwen/Qwen2.5-Coder-14B-Instruct-GGUF qwen2.5-coder-14b-instruct-q4_k_m.gguf  --local-dir ~/models/balanced

# Qwen3 14B Instruct
# https://huggingface.co/Qwen/Qwen3-14B-GGUF
hf download Qwen/Qwen3-14B-GGUF --dry-run
hf download Qwen/Qwen3-14B-GGUF Qwen3-14B-Q4_K_M.gguf --local-dir ~/models/balanced

# DeepSeek-R1-Distill-Qwen-14B
# https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF
# https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF
hf download bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF --dry-run
hf download bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf --local-dir ~/models/balanced

# Phi-4-reasoning-plus 14B
# https://huggingface.co/microsoft/phi-4-gguf
hf download microsoft/phi-4-gguf --dry-run
hf download microsoft/phi-4-gguf phi-4-Q4_K.gguf --local-dir ~/models/balanced

# Gemma 3 12B
# https://huggingface.co/unsloth/gemma-3-12b-it-GGUF
hf download unsloth/gemma-3-12b-it-GGUF --dry-run
hf download unsloth/gemma-3-12b-it-GGUF gemma-3-12b-it-Q4_K_M.gguf --local-dir ~/models/balanced

# Devstral Small 1.1
# https://huggingface.co/mistralai/Devstral-Small-2507_gguf
hf download mistralai/Devstral-Small-2507_gguf --dry-run
hf download mistralai/Devstral-Small-2507_gguf Devstral-Small-2507-Q4_K_M.gguf --local-dir ~/models/balanced

# Ministral 3 14B Instruct
# https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512-GGUF
hf download mistralai/Ministral-3-14B-Instruct-2512-GGUF --dry-run
hf download mistralai/Ministral-3-14B-Instruct-2512-GGUF Ministral-3-14B-Instruct-2512-Q4_K_M.gguf --local-dir ~/models/balanced