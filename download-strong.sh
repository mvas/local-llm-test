#!/bin/bash

# Qwen2.5-Coder-32B-Instruct
# https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct-GGUF
# https://huggingface.co/bartowski/Qwen2.5-Coder-32B-Instruct-GGUF
hf download Qwen/Qwen2.5-Coder-32B-Instruct-GGUF --dry-run
hf download Qwen/Qwen2.5-Coder-32B-Instruct-GGUF qwen2.5-coder-32b-instruct-q4_k_m.gguf --local-dir ~/models

# Qwen3 32B
# https://huggingface.co/Qwen/Qwen3-32B-GGUF
hf download Qwen/Qwen3-32B-GGUF --dry-run
hf download Qwen/Qwen3-32B-GGUF Qwen3-32B-Q4_K_M.gguf --local-dir ~/models


# Qwen3.5-27B
# https://huggingface.co/bartowski/Qwen_Qwen3.5-27B-GGUF
hf download bartowski/Qwen_Qwen3.5-27B-GGUF --dry-run
hf download bartowski/Qwen_Qwen3.5-27B-GGUF Qwen_Qwen3.5-27B-Q4_K_M.gguf --local-dir ~/models

# Qwen3.5-35B-A3B
# https://huggingface.co/bartowski/Qwen_Qwen3.5-35B-A3B-GGUF
hf download bartowski/Qwen_Qwen3.5-35B-A3B-GGUF --dry-run
hf download bartowski/Qwen_Qwen3.5-35B-A3B-GGUF Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf --local-dir ~/models

# DeepSeek-R1-Distill-Qwen-32B
# https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF
hf download bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF --dry-run
hf download bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf --local-dir ~/models