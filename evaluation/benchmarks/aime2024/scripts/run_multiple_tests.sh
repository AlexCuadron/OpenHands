#!/bin/bash
# Script to run multiple tests of the AIME2024 benchmark and average the results

# Default values
MODEL_CONFIG=${1:-"togetherDeepseek"}
COMMIT_HASH=${2:-"HEAD"}
AGENT=${3:-"CodeActAgent"}
EVAL_LIMIT=${4:-10}  # Default to 10 examples for testing
NUM_WORKERS=${5:-5}
EVAL_IDS=${6:-""}
ALLOWED_TOOLS=${7:-"ipython_only"}
NUM_RUNS=${8:-3}  # Default to 3 runs

# Create a directory for the multiple runs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./evaluation/evaluation_outputs/AIME2024_multi_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

echo "Starting multiple runs of AIME2024 benchmark"
echo "Model: ${MODEL_CONFIG}"
echo "Agent: ${AGENT}"
echo "Number of examples: ${EVAL_LIMIT}"
echo "Number of runs: ${NUM_RUNS}"
echo "Output directory: ${OUTPUT_DIR}"

# Run the benchmark multiple times
for i in $(seq 1 ${NUM_RUNS}); do
    echo "Starting run ${i}/${NUM_RUNS}..."
    
    # Create a subdirectory for this run
    RUN_DIR="${OUTPUT_DIR}/run_${i}"
    mkdir -p "${RUN_DIR}"
    
    # Run the benchmark
    bash evaluation/benchmarks/aime2024/scripts/run_infer.sh \
        "${MODEL_CONFIG}" \
        "${COMMIT_HASH}" \
        "${AGENT}" \
        "${EVAL_LIMIT}" \
        "${NUM_WORKERS}" \
        "${EVAL_IDS}" \
        "eval" \
        "${ALLOWED_TOOLS}" \
        "${RUN_DIR}"
    
    echo "Completed run ${i}/${NUM_RUNS}"
done

# Analyze the results
echo "Analyzing results from all runs..."

# Create a Python script to average the results
ANALYSIS_SCRIPT="${OUTPUT_DIR}/average_results.py"
cat > "${ANALYSIS_SCRIPT}" << 'EOF'
import json
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    # Get the directory containing all runs
    base_dir = sys.argv[1]
    
    # Find all summary.json files
    summary_files = list(Path(base_dir).glob("run_*/summary.json"))
    
    if not summary_files:
        print("No summary files found!")
        return
    
    # Load all summaries
    summaries = []
    for file in summary_files:
        with open(file, 'r') as f:
            summaries.append(json.load(f))
    
    # Extract accuracy values
    accuracies = [s.get('accuracy', 0) for s in summaries]
    
    # Calculate average and standard deviation
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    # Create a combined summary
    combined_summary = {
        "num_runs": len(summaries),
        "average_accuracy": float(avg_accuracy),
        "std_accuracy": float(std_accuracy),
        "individual_accuracies": accuracies,
        "run_details": summaries
    }
    
    # Save the combined summary
    with open(os.path.join(base_dir, "combined_summary.json"), 'w') as f:
        json.dump(combined_summary, f, indent=2)
    
    print(f"Combined {len(summaries)} runs:")
    print(f"Average accuracy: {avg_accuracy:.2f}% ± {std_accuracy:.2f}%")
    print(f"Individual accuracies: {accuracies}")
    print(f"Results saved to {os.path.join(base_dir, 'combined_summary.json')}")

if __name__ == "__main__":
    main()
EOF

# Make the script executable
chmod +x "${ANALYSIS_SCRIPT}"

# Run the analysis script
python "${ANALYSIS_SCRIPT}" "${OUTPUT_DIR}"

echo "Multiple runs completed and analyzed."
echo "Results are available in ${OUTPUT_DIR}/combined_summary.json"