#!/usr/bin/env bash
set -eo pipefail

# This script runs multiple tests for the AIME2025 benchmark
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
OVERTHINKING_THRESHOLD=${9:-""}

# Create a temporary directory for the results
TMP_DIR=$(mktemp -d)
echo "Created temporary directory: $TMP_DIR"

# Function to clean up temporary files
cleanup() {
  if [ -n "$TMP_DIR" ] && [ -d "$TMP_DIR" ]; then
    rm -rf "$TMP_DIR"
    echo "Cleaned up temporary directory: $TMP_DIR"
  fi
}

# Register the cleanup function to be called on exit
trap cleanup EXIT

# Run the tests
echo "Running tests with MODEL_CONFIG=$MODEL_CONFIG, AGENT=$AGENT, EVAL_LIMIT=$EVAL_LIMIT, NUM_WORKERS=$NUM_WORKERS, EVAL_IDS=$EVAL_IDS"
bash evaluation/benchmarks/aime2025/scripts/run_infer.sh "$MODEL_CONFIG" "$COMMIT_HASH" "$AGENT" "$EVAL_LIMIT" "$NUM_WORKERS" "$EVAL_IDS" "$RUN_EVALUATION" "$ALLOWED_TOOLS" "$OVERTHINKING_THRESHOLD"

# Get the output directory
OUTPUT_DIR=$(find evaluation/evaluation_outputs -path "*/AIME2025/$AGENT/*" -type d | sort -r | head -n 1)
if [ -z "$OUTPUT_DIR" ]; then
  echo "Error: Could not find output directory"
  exit 1
fi

echo "Output directory: $OUTPUT_DIR"

# Copy the results to the temporary directory
cp -r "$OUTPUT_DIR" "$TMP_DIR/"

# Run the tests with a different model
echo ""
echo "Running tests with a different model..."
MODEL_CONFIG="togetherLlama3"
echo "Running tests with MODEL_CONFIG=$MODEL_CONFIG, AGENT=$AGENT, EVAL_LIMIT=$EVAL_LIMIT, NUM_WORKERS=$NUM_WORKERS, EVAL_IDS=$EVAL_IDS"
bash evaluation/benchmarks/aime2025/scripts/run_infer.sh "$MODEL_CONFIG" "$COMMIT_HASH" "$AGENT" "$EVAL_LIMIT" "$NUM_WORKERS" "$EVAL_IDS" "$RUN_EVALUATION" "$ALLOWED_TOOLS" "$OVERTHINKING_THRESHOLD"

# Get the output directory
OUTPUT_DIR=$(find evaluation/evaluation_outputs -path "*/AIME2025/$AGENT/*" -type d | sort -r | head -n 1)
if [ -z "$OUTPUT_DIR" ]; then
  echo "Error: Could not find output directory"
  exit 1
fi

echo "Output directory: $OUTPUT_DIR"

# Copy the results to the temporary directory
cp -r "$OUTPUT_DIR" "$TMP_DIR/"

# Run the tests with a different agent
echo ""
echo "Running tests with a different agent..."
MODEL_CONFIG="togetherDeepseek"
AGENT="ThinkingAgent"
echo "Running tests with MODEL_CONFIG=$MODEL_CONFIG, AGENT=$AGENT, EVAL_LIMIT=$EVAL_LIMIT, NUM_WORKERS=$NUM_WORKERS, EVAL_IDS=$EVAL_IDS"
bash evaluation/benchmarks/aime2025/scripts/run_infer.sh "$MODEL_CONFIG" "$COMMIT_HASH" "$AGENT" "$EVAL_LIMIT" "$NUM_WORKERS" "$EVAL_IDS" "$RUN_EVALUATION" "$ALLOWED_TOOLS" "$OVERTHINKING_THRESHOLD"

# Get the output directory
OUTPUT_DIR=$(find evaluation/evaluation_outputs -path "*/AIME2025/$AGENT/*" -type d | sort -r | head -n 1)
if [ -z "$OUTPUT_DIR" ]; then
  echo "Error: Could not find output directory"
  exit 1
fi

echo "Output directory: $OUTPUT_DIR"

# Copy the results to the temporary directory
cp -r "$OUTPUT_DIR" "$TMP_DIR/"

# Print a summary of the results
echo ""
echo "Summary of results:"
echo "==================="
echo "Results saved to: $TMP_DIR"
echo ""
echo "To compare the results, run:"
echo "  ls -la $TMP_DIR"
echo "  cat $TMP_DIR/*/analysis/summary.txt"