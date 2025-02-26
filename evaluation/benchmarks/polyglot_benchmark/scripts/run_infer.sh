#!/bin/bash

set -e

# Default values
MODEL_CONFIG=${1:-"eval_gpt4_1106_preview"}
GIT_VERSION=${2:-"HEAD"}
AGENT=${3:-"CodeActAgent"}
EVAL_LIMIT=${4:-"-1"}
EVAL_NUM_WORKERS=${5:-"1"}
EVAL_IDS=${6:-""}
EVAL_LANGUAGES=${7:-""}

# Set environment variables
export USE_UNIT_TESTS=${USE_UNIT_TESTS:-"true"}
export NO_DOCKER=${NO_DOCKER:-"false"}

# Check if we have a local Docker image env file
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BENCHMARK_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"
DOCKER_ENV_FILE="${BENCHMARK_DIR}/docker_image.env"

if [ -f "$DOCKER_ENV_FILE" ]; then
  echo "Loading Docker image configuration from $DOCKER_ENV_FILE"
  source "$DOCKER_ENV_FILE"
else
  # If no local image is available, use the default
  export POLYGLOT_DOCKER_IMAGE=${POLYGLOT_DOCKER_IMAGE:-"ghcr.io/opendevin/eval-polyglot:v1.0.0"}
  
  # Check if we need to build a local Docker image
  if [ "$BUILD_LOCAL_DOCKER" = "true" ]; then
    echo "Building local Docker image..."
    "${SCRIPT_DIR}/build_local_docker.sh"
    source "$DOCKER_ENV_FILE"
  fi
fi

echo "Using Docker image: $POLYGLOT_DOCKER_IMAGE"

# Try to find the polyglot-benchmark repository
if [ -z "$POLYGLOT_BENCHMARK_PATH" ]; then
  # Check common locations
  POSSIBLE_PATHS=(
    "/workspace/polyglot-benchmark"
    "$HOME/polyglot-benchmark"
    "$HOME/thereal/polyglot-benchmark"
    "$(git rev-parse --show-toplevel)/polyglot-benchmark"
    "$(pwd)/polyglot-benchmark"
  )
  
  for path in "${POSSIBLE_PATHS[@]}"; do
    if [ -d "$path" ]; then
      export POLYGLOT_BENCHMARK_PATH="$path"
      echo "Found polyglot-benchmark repository at: $POLYGLOT_BENCHMARK_PATH"
      break
    fi
  done
fi

# If still not found, try to clone it
if [ -z "$POLYGLOT_BENCHMARK_PATH" ] || [ ! -d "$POLYGLOT_BENCHMARK_PATH" ]; then
  echo "Polyglot benchmark repository not found. Attempting to clone it..."
  CLONE_DIR="$(git rev-parse --show-toplevel)/polyglot-benchmark"
  git clone https://github.com/Aider-AI/polyglot-benchmark.git "$CLONE_DIR"
  if [ $? -eq 0 ]; then
    export POLYGLOT_BENCHMARK_PATH="$CLONE_DIR"
    echo "Successfully cloned polyglot-benchmark to $POLYGLOT_BENCHMARK_PATH"
  else
    echo "Failed to clone polyglot-benchmark. Please set POLYGLOT_BENCHMARK_PATH manually."
    exit 1
  fi
fi

# Add additional arguments based on provided parameters
ARGS="--agent-cls ${AGENT} --llm-config ${MODEL_CONFIG} --max-iterations 30 --eval-num-workers ${EVAL_NUM_WORKERS}"

if [ "${EVAL_LIMIT}" != "-1" ]; then
  ARGS="${ARGS} --eval-n-limit ${EVAL_LIMIT}"
fi

if [ -n "${EVAL_IDS}" ]; then
  ARGS="${ARGS} --eval-ids ${EVAL_IDS}"
fi

if [ -n "${EVAL_LANGUAGES}" ]; then
  ARGS="${ARGS} --eval-languages ${EVAL_LANGUAGES}"
fi

# Run the evaluation
cd "$(git rev-parse --show-toplevel)"
poetry run python -m evaluation.benchmarks.polyglot_benchmark.run_infer ${ARGS}