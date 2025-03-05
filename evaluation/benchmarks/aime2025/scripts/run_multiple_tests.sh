#!/usr/bin/env bash
set -eo pipefail

# This script runs multiple tests from the AIME2025 benchmark
# Usage: bash evaluation/benchmarks/aime2025/scripts/run_multiple_tests.sh <llm-config> <commit-hash> <agent-cls> <eval-limit> <num-workers> <eval-ids> <run-evaluation> <allowed-tools>

# Default values
MODEL_CONFIG=${1:-"togetherDeepseek"}
COMMIT_HASH=${2:-"HEAD"}
AGENT=${3:-"CodeActAgent"}
EVAL_LIMIT=${4:-5}
NUM_WORKERS=${5:-1}
EVAL_IDS=${6:-"0,1,2,3,4"}
RUN_EVALUATION=${7:-"eval"}
ALLOWED_TOOLS=${8:-"ipython_only"}

# Print the parameters
echo "Running AIME2025 multiple tests with the following parameters:"
echo "MODEL_CONFIG: $MODEL_CONFIG"
echo "COMMIT_HASH: $COMMIT_HASH"
echo "AGENT: $AGENT"
echo "EVAL_LIMIT: $EVAL_LIMIT"
echo "NUM_WORKERS: $NUM_WORKERS"
echo "EVAL_IDS: $EVAL_IDS"
echo "RUN_EVALUATION: $RUN_EVALUATION"
echo "ALLOWED_TOOLS: $ALLOWED_TOOLS"

# Check if the dataset exists
if [ ! -d "AIME2025" ]; then
  echo "AIME2025 dataset not found locally. Attempting to download from Hugging Face..."
  git clone https://huggingface.co/datasets/opencompass/AIME2025 || echo "Failed to download dataset. The benchmark will attempt to download it automatically."
fi

# Run the benchmark with the specified parameters
bash evaluation/benchmarks/aime2025/scripts/run_infer.sh "$MODEL_CONFIG" "$COMMIT_HASH" "$AGENT" "$EVAL_LIMIT" "$NUM_WORKERS" "$EVAL_IDS" "$RUN_EVALUATION" "$ALLOWED_TOOLS"