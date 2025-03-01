#!/bin/bash

# Default values
LLM_CONFIG="openai/gpt-4o"
AGENT_CLS="CodeActAgent"
EVAL_OUTPUT_DIR="eval_outputs/math500"
MAX_ITERATIONS=20
EVAL_N_LIMIT=0  # 0 means no limit
EVAL_NUM_WORKERS=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --llm_config)
      LLM_CONFIG="$2"
      shift 2
      ;;
    --agent_cls)
      AGENT_CLS="$2"
      shift 2
      ;;
    --eval_output_dir)
      EVAL_OUTPUT_DIR="$2"
      shift 2
      ;;
    --max_iterations)
      MAX_ITERATIONS="$2"
      shift 2
      ;;
    --eval_n_limit)
      EVAL_N_LIMIT="$2"
      shift 2
      ;;
    --eval_num_workers)
      EVAL_NUM_WORKERS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Run the benchmark
cd /workspace/OpenHands
python -m evaluation.benchmarks.math500.run_infer \
  --llm_config "$LLM_CONFIG" \
  --agent_cls "$AGENT_CLS" \
  --eval_output_dir "$EVAL_OUTPUT_DIR" \
  --max_iterations "$MAX_ITERATIONS" \
  --eval_n_limit "$EVAL_N_LIMIT" \
  --eval_num_workers "$EVAL_NUM_WORKERS"