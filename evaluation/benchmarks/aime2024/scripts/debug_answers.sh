#!/usr/bin/env bash
set -eo pipefail

# Check if an output file is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <path-to-output-jsonl>"
  echo "Example: $0 ./evaluation/evaluation_outputs/AIME2024/CodeActAgent/v0.26.0/output.jsonl"
  exit 1
fi

OUTPUT_FILE=$1

echo "======================================"
echo "Debugging answer extraction for AIME2024"
echo "======================================"
echo "Input file: $OUTPUT_FILE"
echo "======================================"

# Run the debug script
poetry run python evaluation/benchmarks/aime2024/scripts/debug_answers.py "$OUTPUT_FILE" --save-csv

echo ""
echo "======================================"
echo "Debugging complete!"
echo "======================================"