#!/bin/bash

set -e

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BENCHMARK_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

# Build the Docker image
docker build -t ghcr.io/opendevin/eval-polyglot:v1.0.0 -f "${BENCHMARK_DIR}/Dockerfile" "${BENCHMARK_DIR}"

echo "Docker image built successfully: ghcr.io/opendevin/eval-polyglot:v1.0.0"