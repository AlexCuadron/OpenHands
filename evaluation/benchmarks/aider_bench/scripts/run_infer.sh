#!/usr/bin/env bash
set -eo pipefail

source "evaluation/utils/version_control.sh"

MODEL_CONFIG=$1
COMMIT_HASH=$2
AGENT=$3
EVAL_LIMIT=$4
NUM_WORKERS=$5
EVAL_IDS=$6
RUN_EVALUATION=$7  # New parameter to run evaluation after benchmark

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

EVAL_NOTE=$OPENHANDS_VERSION

# Default to NOT use unit tests.
if [ -z "$USE_UNIT_TESTS" ]; then
  export USE_UNIT_TESTS=false
fi
echo "USE_UNIT_TESTS: $USE_UNIT_TESTS"
# If use unit tests, set EVAL_NOTE to the commit hash
if [ "$USE_UNIT_TESTS" = true ]; then
  EVAL_NOTE=$EVAL_NOTE-w-test
fi

COMMAND="export PYTHONPATH=evaluation/benchmarks/aider_bench:\$PYTHONPATH && poetry run python evaluation/benchmarks/aider_bench/run_infer.py \
  --agent-cls $AGENT \
  --llm-config $MODEL_CONFIG \
  --max-iterations 30 \
  --eval-num-workers $NUM_WORKERS \
  --eval-note $EVAL_NOTE"

if [ -n "$EVAL_LIMIT" ]; then
  echo "EVAL_LIMIT: $EVAL_LIMIT"
  COMMAND="$COMMAND --eval-n-limit $EVAL_LIMIT"
fi

# Only pass eval-ids if it's not "eval" (which is a special parameter for evaluation mode)
if [ -n "$EVAL_IDS" ] && [ "$EVAL_IDS" != "eval" ]; then
  echo "EVAL_IDS: $EVAL_IDS"
  COMMAND="$COMMAND --eval-ids $EVAL_IDS"
fi

# Run the command
eval $COMMAND

# Get the output directory - first try the default location
OUTPUT_DIR=$(find evaluation/evaluation_outputs -path "*/AiderBench/$AGENT/*" -type d -name "*$EVAL_NOTE*" 2>/dev/null | sort -r | head -n 1)

# If not found, try to find it anywhere under evaluation_outputs
if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR=$(find . -path "*/evaluation_outputs/*" -path "*/AiderBench/$AGENT/*" -type d -name "*$EVAL_NOTE*" 2>/dev/null | sort -r | head -n 1)
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
    poetry run python evaluation/benchmarks/aider_bench/scripts/summarize_results.py "$OUTPUT_FILE"
    
    # Save the evaluation results
    EVAL_RESULTS_FILE="$OUTPUT_DIR/evaluation_results.txt"
    echo "Saving evaluation results to: $EVAL_RESULTS_FILE"
    poetry run python evaluation/benchmarks/aider_bench/scripts/summarize_results.py "$OUTPUT_FILE" > "$EVAL_RESULTS_FILE"
    
    echo ""
    echo "Evaluation complete. Results saved to: $EVAL_RESULTS_FILE"
  else
    echo "Error: Output file not found: $OUTPUT_FILE"
    echo "Cannot run evaluation."
  fi
fi
