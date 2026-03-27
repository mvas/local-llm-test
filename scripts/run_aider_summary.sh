#! /usr/bin/env bash

# Use this inside the Aider benchmark interactive container to generate a summary.

python3 /aider/benchmark/benchmark.py \
    --stats tmp.benchmarks/2026-01-01-004-run | tee tmp.benchmarks/2026-01-01-004-run.summary.yml