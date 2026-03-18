#!/bin/bash

# Qwen2.5-Coder 7B 
# https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF
# https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF
# https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF
# https://huggingface.co/bartowski/Qwen2.5.1-Coder-7B-Instruct-GGUF

hf download bartowski/Qwen2.5.1-Coder-7B-Instruct-GGUF --dry-run
hf download bartowski/Qwen2.5.1-Coder-7B-Instruct-GGUF Qwen2.5.1-Coder-7B-Instruct-Q4_K_M.gguf --local-dir ~/models

# Qwen3.5-9B model
# https://huggingface.co/bartowski/Qwen_Qwen3.5-9B-GGUF

hf download bartowski/Qwen_Qwen3.5-9B-GGUF --dry-run
hf download bartowski/Qwen_Qwen3.5-9B-GGUF Qwen_Qwen3.5-9B-Q4_K_L.gguf  --local-dir ~/models

# Llama 3.1 8B Instruct
# https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF

hf download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF --dry-run
hf download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --local-dir ~/models

# Gemma 3 4B
# https://huggingface.co/bartowski/google_gemma-3n-E4B-it-GGUF
# https://huggingface.co/unsloth/gemma-3n-E4B-it-GGUF

hf download bartowski/google_gemma-3n-E4B-it-GGUF --dry-run
hf download bartowski/google_gemma-3n-E4B-it-GGUF google_gemma-3n-E4B-it-Q4_K_M.gguf --local-dir ~/models

# Ministral 3 8B Instruct
# https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-GGUF
# or
# https://huggingface.co/collections/unsloth/ministral-3

hf download mistralai/Ministral-3-8B-Instruct-2512-GGUF --dry-run
hf download mistralai/Ministral-3-8B-Instruct-2512-GGUF Ministral-3-8B-Instruct-2512-Q4_K_M.gguf  --local-dir ~/models