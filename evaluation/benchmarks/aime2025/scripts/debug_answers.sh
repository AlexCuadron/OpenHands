#!/usr/bin/env bash
set -eo pipefail

# This script debugs the answers extracted from an AIME2025 benchmark run
# Usage: bash evaluation/benchmarks/aime2025/scripts/debug_answers.sh <path-to-output-jsonl> [output-file]

OUTPUT_FILE=$1
DEBUG_OUTPUT_FILE=$2

if [ -z "$OUTPUT_FILE" ]; then
  echo "Error: No output file specified"
  echo "Usage: bash evaluation/benchmarks/aime2025/scripts/debug_answers.sh <path-to-output-jsonl> [output-file]"
  exit 1
fi

if [ ! -f "$OUTPUT_FILE" ]; then
  echo "Error: Output file not found: $OUTPUT_FILE"
  exit 1
fi

if [ -z "$DEBUG_OUTPUT_FILE" ]; then
  # If no output file is specified, use the directory of the output file
  DEBUG_OUTPUT_FILE=$(dirname "$OUTPUT_FILE")/debug_answers.csv
fi

echo "Debugging answers in: $OUTPUT_FILE"
echo "Saving results to: $DEBUG_OUTPUT_FILE"

# Run the debug script
poetry run python evaluation/benchmarks/aime2025/scripts/debug_answers.py "$OUTPUT_FILE" --output-file "$DEBUG_OUTPUT_FILE"

echo ""
echo "Debugging complete. Results saved to: $DEBUG_OUTPUT_FILE"