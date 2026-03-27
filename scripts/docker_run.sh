#!/usr/bin/env bash

# Use this script to run the Aider benchmark container interactively.

docker run \
       -it --rm \
       --memory=8g \
       --memory-swap=8g \
       --add-host=host.docker.internal:host-gateway \
       -v `pwd`/../aider:/aider \
       -v `pwd`/../aider/tmp.benchmarks:/benchmarks \
       -e OPENAI_API_KEY=local-benchmark \
       -e OPENAI_API_BASE=http://host.docker.internal:8082/v1 \
       -e AIDER_DOCKER=1 \
       -e AIDER_BENCHMARK_DIR=/benchmarks \
       aider-benchmark \
       bash