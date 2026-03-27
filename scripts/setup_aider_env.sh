#! /usr/bin/env bash

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker is required but not found in PATH" >&2
  exit 1
fi

cd ../
git clone https://github.com/Aider-AI/aider.git
cd aider
mkdir -p tmp.benchmarks
git clone https://github.com/Aider-AI/polyglot-benchmark tmp.benchmarks/polyglot-benchmark
./benchmark/docker_build.sh

docker build -t aider-benchmark ../aider/benchmark

echo
echo "Aider container environment is ready."