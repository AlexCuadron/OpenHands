#!/bin/bash

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

# Parse command line arguments
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

# Create output directory if it doesn't exist
mkdir -p "$EVAL_OUTPUT_DIR"

# Run the evaluation
python -m evaluation.benchmarks.aime2024.run_infer \
  --agent-cls "$AGENT_CLS" \
  --llm-config "$LLM_CONFIG" \
  --max-iterations "$MAX_ITERATIONS" \
  --eval-note "$EVAL_NOTE" \
  --eval-output-dir "$EVAL_OUTPUT_DIR" \
  --eval-num-workers "$EVAL_NUM_WORKERS" \
  --eval-n-limit "$EVAL_N_LIMIT" \
  --eval-ids "$EVAL_IDS" \
  --allowed-tools "$ALLOWED_TOOLS"