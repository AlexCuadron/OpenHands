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

# Set BUILD_LOCAL_DOCKER to true by default if not specified
export BUILD_LOCAL_DOCKER=${BUILD_LOCAL_DOCKER:-"true"}

if [ -f "$DOCKER_ENV_FILE" ]; then
  echo "Loading Docker image configuration from $DOCKER_ENV_FILE"
  source "$DOCKER_ENV_FILE"
else
  # If no local image is available, use the default
  export POLYGLOT_DOCKER_IMAGE=${POLYGLOT_DOCKER_IMAGE:-"ghcr.io/opendevin/eval-polyglot:v1.0.0"}
  
  # Try to pull the image first
  echo "Trying to pull Docker image: $POLYGLOT_DOCKER_IMAGE"
  if ! docker pull "$POLYGLOT_DOCKER_IMAGE" 2>/dev/null; then
    echo "Failed to pull Docker image: $POLYGLOT_DOCKER_IMAGE"
    
    # Build a local Docker image if pulling fails and BUILD_LOCAL_DOCKER is true
    if [ "$BUILD_LOCAL_DOCKER" = "true" ]; then
      echo "Building local Docker image..."
      "${SCRIPT_DIR}/build_local_docker.sh"
      source "$DOCKER_ENV_FILE"
    else
      echo "WARNING: Docker image not found and BUILD_LOCAL_DOCKER is not set to true."
      echo "You can build a local Docker image by running:"
      echo "  ${SCRIPT_DIR}/build_local_docker.sh"
      echo "Or set BUILD_LOCAL_DOCKER=true to build it automatically."
    fi
  else
    echo "Successfully pulled Docker image: $POLYGLOT_DOCKER_IMAGE"
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