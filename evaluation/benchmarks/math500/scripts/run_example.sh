#!/bin/bash

# This script runs the MATH-500 benchmark on a small subset of problems

# Set the output directory
EVAL_OUTPUT_DIR="eval_outputs/math500_example"

# Run the benchmark on 3 problems
cd /workspace/OpenHands
python -m evaluation.benchmarks.math500.run_infer \
  --llm_config "openai/gpt-4o" \
  --agent_cls "CodeActAgent" \
  --eval_output_dir "$EVAL_OUTPUT_DIR" \
  --max_iterations 20 \
  --eval_n_limit 3 \
  --eval_num_workers 1

# Summarize the results
python -m evaluation.benchmarks.math500.scripts.summarize_results \
  --output_file "$EVAL_OUTPUT_DIR/output.jsonl" \
  --csv_output "$EVAL_OUTPUT_DIR/summary.csv"