#!/usr/bin/env bash
set -eo pipefail

# Check if an output file is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <path-to-output-jsonl> [output-directory]"
  echo "Example: $0 ./evaluation/evaluation_outputs/AIME2024/CodeActAgent/v0.26.0/output.jsonl"
  exit 1
fi

OUTPUT_FILE=$1
OUTPUT_DIR=${2:-"$(dirname "$OUTPUT_FILE")/analysis"}

echo "======================================"
echo "Running evaluation on AIME2024 results"
echo "======================================"
echo "Input file: $OUTPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "======================================"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the evaluation
poetry run python evaluation/benchmarks/aime2024/scripts/analyze_results.py "$OUTPUT_FILE" --output-dir "$OUTPUT_DIR"

echo ""
echo "======================================"
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "======================================"

# Display summary if available
SUMMARY_FILE="$OUTPUT_DIR/summary.json"
if [ -f "$SUMMARY_FILE" ]; then
  echo ""
  echo "Summary:"
  cat "$SUMMARY_FILE" | python -m json.tool
fi

echo ""
echo "To view detailed results, check the CSV file: $OUTPUT_DIR/detailed_results.csv"