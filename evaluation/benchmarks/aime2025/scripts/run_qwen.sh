#!/bin/bash

# Run the AIME2025 benchmark with our custom Qwen provider
cd /workspace/OpenHands
python -m evaluation.benchmarks.aime2025.run_with_qwen \
  --dataset aime2025-I \
  --output_dir evaluation_outputs/aime2025_qwen \
  --agent CodeActAgent \
  --allowed_tools ipython_only \
  --max_iterations 20