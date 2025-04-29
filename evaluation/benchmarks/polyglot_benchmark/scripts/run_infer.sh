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
export POLYGLOT_BENCHMARK_PATH=${POLYGLOT_BENCHMARK_PATH:-"/workspace/polyglot-benchmark"}
export USE_UNIT_TESTS=${USE_UNIT_TESTS:-"true"}

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