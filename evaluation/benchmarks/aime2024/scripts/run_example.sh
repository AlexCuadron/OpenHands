#!/usr/bin/env bash
set -eo pipefail

source "evaluation/utils/version_control.sh"

MODEL_CONFIG=$1
COMMIT_HASH=$2
AGENT=$3
EVAL_LIMIT=1  # Default to 1 for example
NUM_WORKERS=${5:-1}
EVAL_IDS=${6:-"0"}  # Default to first example
RUN_EVALUATION=$7  # Parameter to run evaluation after benchmark
ALLOWED_TOOLS=${8:-"all"}  # Parameter to specify allowed tools, default is "all"

# Function to clean up temporary files
cleanup() {
  if [ -n "$TMP_DIR" ] && [ -d "$TMP_DIR" ]; then
    rm -rf "$TMP_DIR"
    echo "Cleaned up temporary directory: $TMP_DIR"
  fi
}

# Register the cleanup function to be called on exit
trap cleanup EXIT

# No temporary config file creation - we'll use the existing config.toml
CONFIG_FILE_ARG=""

# Special case: if the 7th parameter is "eval", set RUN_EVALUATION to "eval"
if [ "$RUN_EVALUATION" = "eval" ]; then
  echo "Evaluation mode enabled"
fi

# Special case: if any parameter is "eval", set RUN_EVALUATION to "eval"
for param in "$@"; do
  if [ "$param" = "eval" ]; then
    RUN_EVALUATION="eval"
    echo "Evaluation mode enabled"
    break
  fi
done

if [ -z "$NUM_WORKERS" ]; then
  NUM_WORKERS=1
  echo "Number of workers not specified, use default $NUM_WORKERS"
fi
checkout_eval_branch

if [ -z "$AGENT" ]; then
  echo "Agent not specified, use default CodeActAgent"
  AGENT="CodeActAgent"
fi

get_openhands_version

echo "AGENT: $AGENT"
echo "OPENHANDS_VERSION: $OPENHANDS_VERSION"
echo "MODEL_CONFIG: $MODEL_CONFIG"
echo "EVAL_IDS: $EVAL_IDS (Running example)"

EVAL_NOTE="$OPENHANDS_VERSION-example"

COMMAND="export PYTHONPATH=evaluation/benchmarks/aime2024:\$PYTHONPATH && poetry run python evaluation/benchmarks/aime2024/run_infer.py \
  --agent-cls $AGENT \
  --llm-config $MODEL_CONFIG \
  --max-iterations 30 \
  --eval-num-workers $NUM_WORKERS \
  --eval-note $EVAL_NOTE \
  --allowed-tools $ALLOWED_TOOLS \
  --eval-n-limit $EVAL_LIMIT \
  --eval-ids $EVAL_IDS \
  $CONFIG_FILE_ARG"

# Print the allowed tools
echo "ALLOWED_TOOLS: $ALLOWED_TOOLS"

# Run the command
eval $COMMAND

# Get the output directory - first try the default location
OUTPUT_DIR=$(find evaluation/evaluation_outputs -path "*/AIME2024/$AGENT/*" -type d -name "*$EVAL_NOTE*" 2>/dev/null | sort -r | head -n 1)

# If not found, try to find it anywhere under evaluation_outputs
if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR=$(find . -path "*/evaluation_outputs/*" -path "*/AIME2024/$AGENT/*" -type d -name "*$EVAL_NOTE*" 2>/dev/null | sort -r | head -n 1)
fi

# If still not found, try to find any output.jsonl file
if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_FILE=$(find . -name "output.jsonl" 2>/dev/null | sort -r | head -n 1)
  if [ -n "$OUTPUT_FILE" ]; then
    OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
  fi
else
  OUTPUT_FILE="$OUTPUT_DIR/output.jsonl"
fi

# Print the output directory and file for debugging
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "Output file: $OUTPUT_FILE"

# Run evaluation if requested
if [ "$RUN_EVALUATION" = "eval" ]; then
  echo ""
  echo "======================================"
  echo "Running evaluation on results..."
  echo "======================================"
  echo ""
  
  if [ -f "$OUTPUT_FILE" ]; then
    echo "Evaluating results in: $OUTPUT_FILE"
    poetry run python evaluation/benchmarks/aime2024/scripts/analyze_results.py "$OUTPUT_FILE" --output-dir "$OUTPUT_DIR/analysis"
    
    echo ""
    echo "Evaluation complete. Results saved to: $OUTPUT_DIR/analysis"
  else
    echo "Error: Output file not found: $OUTPUT_FILE"
    echo "Cannot run evaluation."
  fi
fi