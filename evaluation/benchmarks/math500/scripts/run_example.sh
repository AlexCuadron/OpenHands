#!/bin/bash

# Example script to run the MATH-500 benchmark with a specific LLM

# Set the LLM configuration
LLM_CONFIG="openai/gpt-4-turbo"

# Set the output directory
OUTPUT_DIR="./eval_results/math500"

# Set the number of iterations
MAX_ITERATIONS=10

# Set the number of workers
NUM_WORKERS=1

# Set the number of examples to evaluate (optional)
# EVAL_N_LIMIT=5

# Run the benchmark
python -m evaluation.benchmarks.math500.run_infer \
  --llm_config $LLM_CONFIG \
  --agent_cls CodeActAgent \
  --max_iterations $MAX_ITERATIONS \
  --eval_output_dir $OUTPUT_DIR \
  --eval_num_workers $NUM_WORKERS \
  ${EVAL_N_LIMIT:+--eval_n_limit $EVAL_N_LIMIT}