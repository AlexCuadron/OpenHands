#!/usr/bin/env bash
set -eo pipefail

# This script evaluates the results of an AIME2025 benchmark run
# Usage: bash evaluation/benchmarks/aime2025/scripts/eval_infer.sh <path-to-output-jsonl> [output-directory]

OUTPUT_FILE=$1
OUTPUT_DIR=$2

if [ -z "$OUTPUT_FILE" ]; then
  echo "Error: No output file specified"
  echo "Usage: bash evaluation/benchmarks/aime2025/scripts/eval_infer.sh <path-to-output-jsonl> [output-directory]"
  exit 1
fi

if [ ! -f "$OUTPUT_FILE" ]; then
  echo "Error: Output file not found: $OUTPUT_FILE"
  exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
  # If no output directory is specified, use the directory of the output file
  OUTPUT_DIR=$(dirname "$OUTPUT_FILE")/analysis
fi

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Evaluating results in: $OUTPUT_FILE"
echo "Saving results to: $OUTPUT_DIR"

# Run the analysis
poetry run python evaluation/benchmarks/aime2025/scripts/analyze_results.py "$OUTPUT_FILE" --output-dir "$OUTPUT_DIR"

echo ""
echo "Evaluation complete. Results saved to: $OUTPUT_DIR"