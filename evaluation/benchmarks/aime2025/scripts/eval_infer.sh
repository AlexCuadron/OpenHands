#!/usr/bin/env bash
set -eo pipefail

# This script analyzes the results of the AIME2025 benchmark
# Usage: bash evaluation/benchmarks/aime2025/scripts/eval_infer.sh <path-to-output-jsonl> [output-directory]

OUTPUT_FILE=$1
OUTPUT_DIR=$2

if [ -z "$OUTPUT_FILE" ]; then
  echo "Error: No output file specified."
  echo "Usage: bash evaluation/benchmarks/aime2025/scripts/eval_infer.sh <path-to-output-jsonl> [output-directory]"
  exit 1
fi

if [ ! -f "$OUTPUT_FILE" ]; then
  echo "Error: Output file not found: $OUTPUT_FILE"
  exit 1
fi

# Check if the file is empty
if [ ! -s "$OUTPUT_FILE" ]; then
  echo "Error: Output file is empty: $OUTPUT_FILE"
  exit 1
fi

# If no output directory is specified, use the directory of the output file
if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="$(dirname "$OUTPUT_FILE")/analysis"
fi

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Analyzing results in: $OUTPUT_FILE"
echo "Saving analysis to: $OUTPUT_DIR"

# Check if required Python packages are installed
if ! python -c "import pandas, matplotlib" &> /dev/null; then
  echo "Installing required Python packages..."
  pip install pandas matplotlib
fi

# Run the analysis script
poetry run python evaluation/benchmarks/aime2025/scripts/analyze_results.py "$OUTPUT_FILE" --output-dir "$OUTPUT_DIR"

echo ""
echo "Analysis complete. Results saved to: $OUTPUT_DIR"