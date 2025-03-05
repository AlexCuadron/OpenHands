#!/usr/bin/env bash
set -eo pipefail

# This script debugs answers from the AIME2025 benchmark
# Usage: bash evaluation/benchmarks/aime2025/scripts/debug_answers.sh <path-to-output-jsonl> [output-directory]

OUTPUT_FILE=$1
OUTPUT_DIR=$2

if [ -z "$OUTPUT_FILE" ]; then
  echo "Error: No output file specified."
  echo "Usage: bash evaluation/benchmarks/aime2025/scripts/debug_answers.sh <path-to-output-jsonl> [output-directory]"
  exit 1
fi

if [ ! -f "$OUTPUT_FILE" ]; then
  echo "Error: Output file not found: $OUTPUT_FILE"
  exit 1
fi

# If no output directory is specified, use the directory of the output file
if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="$(dirname "$OUTPUT_FILE")/debug"
fi

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Debugging answers in: $OUTPUT_FILE"
echo "Saving debug results to: $OUTPUT_DIR"

# Run the debug script
poetry run python evaluation/benchmarks/aime2025/scripts/debug_answers.py "$OUTPUT_FILE" --output-dir "$OUTPUT_DIR"

echo ""
echo "Debug complete. Results saved to: $OUTPUT_DIR"