#!/bin/bash
set -e

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BENCHMARK_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# Build the Docker image
echo "Building Docker image for MATH-500 benchmark..."
docker build -t openhands-math500:latest "$BENCHMARK_DIR"

echo "Docker image built successfully: openhands-math500:latest"