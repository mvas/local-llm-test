#! /usr/bin/env bash

uv run llama-server \
    -m ~/models/fast/Qwen2.5.1-Coder-7B-Instruct-Q4_K_M.gguf \
    -ngl 99 \
    -c 8192 \
    --reasoning off \
    --host 127.0.0.1 \
    --port 8082