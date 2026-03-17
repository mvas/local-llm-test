#!/bin/bash

# Download Qwen3.5-9B model

hf download bartowski/Qwen_Qwen3.5-9B-GGUF --dry-run
hf download bartowski/Qwen_Qwen3.5-9B-GGUF Qwen_Qwen3.5-9B-Q4_K_L.gguf  --local-dir ~/models
