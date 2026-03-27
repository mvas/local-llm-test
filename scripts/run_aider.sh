#! /usr/bin/env bash

python3 /aider/benchmark/benchmark.py \
    2026-01-01-004-run \
    --model openai/Qwen2.5.1-Coder-7B-Instruct \
    --edit-format whole \
    --threads 1 \
    --num-tests 20 \
    --exercises-dir polyglot-benchmark \
    --languages python