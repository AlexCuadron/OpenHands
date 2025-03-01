#!/usr/bin/env bash
set -eo pipefail

# Support both positional and named arguments
# Positional arguments (for compatibility with MATH500 script):
# $1: MODEL_CONFIG - LLM configuration
# $2: COMMIT_HASH - Not used but kept for compatibility
# $3: AGENT - Agent class
# $4: EVAL_LIMIT - Limit the number of examples (default: 1)
# $5: NUM_WORKERS - Number of workers (default: 1)
# $6: EVAL_IDS - Specific example IDs (default: "0")
# $7: RUN_EVALUATION - Whether to run evaluation after benchmark
# $8: ALLOWED_TOOLS - Tools allowed for the agent (default: "all")

# Default values
AGENT_CLS="CodeActAgent"
LLM_CONFIG="claude-3-opus-20240229"
MAX_ITERATIONS=20
EVAL_NOTE="aime2024_example"
EVAL_OUTPUT_DIR="./evaluation/results/aime2024_example"
EVAL_NUM_WORKERS=1
EVAL_N_LIMIT=1
EVAL_IDS="0"  # Just run the first example
ALLOWED_TOOLS="all"
RUN_EVALUATION=""

# Check if positional arguments are provided
if [ -n "$1" ] && [[ "$1" != --* ]]; then
  # Using positional arguments
  LLM_CONFIG=$1
  # COMMIT_HASH=$2 (not used)
  AGENT_CLS=${3:-"CodeActAgent"}
  EVAL_N_LIMIT=${4:-1}
  EVAL_NUM_WORKERS=${5:-1}
  EVAL_IDS=${6:-"0"}
  RUN_EVALUATION=$7
  ALLOWED_TOOLS=${8:-"all"}
  
  # Use current timestamp as eval note
  EVAL_NOTE="aime2024_example_$(date +%Y%m%d_%H%M%S)"
  
  echo "Using positional arguments:"
  echo "LLM_CONFIG: $LLM_CONFIG"
  echo "AGENT_CLS: $AGENT_CLS"
  echo "EVAL_N_LIMIT: $EVAL_N_LIMIT"
  echo "EVAL_NUM_WORKERS: $EVAL_NUM_WORKERS"
  echo "EVAL_IDS: $EVAL_IDS"
  echo "ALLOWED_TOOLS: $ALLOWED_TOOLS"
else
  # Parse named arguments
  while [[ $# -gt 0 ]]; do
    case $1 in
      --agent-cls)
        AGENT_CLS="$2"
        shift 2
        ;;
      --llm-config)
        LLM_CONFIG="$2"
        shift 2
        ;;
      --max-iterations)
        MAX_ITERATIONS="$2"
        shift 2
        ;;
      --eval-note)
        EVAL_NOTE="$2"
        shift 2
        ;;
      --eval-output-dir)
        EVAL_OUTPUT_DIR="$2"
        shift 2
        ;;
      --eval-num-workers)
        EVAL_NUM_WORKERS="$2"
        shift 2
        ;;
      --eval-n-limit)
        EVAL_N_LIMIT="$2"
        shift 2
        ;;
      --eval-ids)
        EVAL_IDS="$2"
        shift 2
        ;;
      --allowed-tools)
        ALLOWED_TOOLS="$2"
        shift 2
        ;;
      *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
  done
fi

# Special case: if any parameter is "eval", set RUN_EVALUATION to "eval"
for param in "$@"; do
  if [ "$param" = "eval" ]; then
    RUN_EVALUATION="eval"
    echo "Evaluation mode enabled"
    break
  fi
done

# Create output directory if it doesn't exist
mkdir -p "$EVAL_OUTPUT_DIR"

# Build the command
COMMAND="python -m evaluation.benchmarks.aime2024.run_infer \
  --agent-cls $AGENT_CLS \
  --llm-config $LLM_CONFIG \
  --max-iterations $MAX_ITERATIONS \
  --eval-note $EVAL_NOTE \
  --eval-output-dir $EVAL_OUTPUT_DIR \
  --eval-num-workers $EVAL_NUM_WORKERS \
  --eval-n-limit $EVAL_N_LIMIT \
  --eval-ids $EVAL_IDS \
  --allowed-tools $ALLOWED_TOOLS"

# Run the command
echo "Running command: $COMMAND"
eval $COMMAND

# Get the output directory
OUTPUT_DIR=$(find "$EVAL_OUTPUT_DIR" -type d -name "*$EVAL_NOTE*" 2>/dev/null | sort -r | head -n 1)
if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="$EVAL_OUTPUT_DIR"
fi
OUTPUT_FILE="$OUTPUT_DIR/output.jsonl"

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
    python evaluation/benchmarks/aime2024/scripts/analyze_results.py --results-file "$OUTPUT_FILE" --output-dir "$OUTPUT_DIR/analysis"
    
    echo ""
    echo "Evaluation complete. Results saved to: $OUTPUT_DIR/analysis"
  else
    echo "Error: Output file not found: $OUTPUT_FILE"
    echo "Cannot run evaluation."
  fi
fi