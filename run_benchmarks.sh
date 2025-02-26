#!/bin/bash

# Script to run OpenHands benchmarks with retry functionality
# This script will run the polyglot_benchmark and aider_bench benchmarks
# and retry them until they succeed or reach the maximum number of attempts.

# Configuration
MAX_ATTEMPTS=10
RETRY_DELAY=30  # seconds
MODEL_CONFIG="togetherDeepseek"
GIT_VERSION="HEAD"
AGENT="CodeActAgent"
EVAL_LIMIT=1000
NUM_WORKERS=30

# Check if Docker is available
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo "WARNING: Docker is not available in this environment."
        echo "The benchmarks require Docker to run properly."
        echo "Continuing anyway, but expect failures if Docker is required."
    fi
}

# Function to run a command and retry until it succeeds
run_with_retry() {
    local cmd="$1"
    local benchmark_name="$2"
    local attempt=1
    local exit_code=1
    
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Running $benchmark_name benchmark"
    echo "Command: $cmd"
    
    while [[ $exit_code -ne 0 && $attempt -le $MAX_ATTEMPTS ]]; do
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Attempt $attempt of $MAX_ATTEMPTS..."
        
        # Run the command
        eval "$cmd"
        exit_code=$?
        
        if [[ $exit_code -ne 0 ]]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Command failed with exit code $exit_code."
            
            if [[ $attempt -lt $MAX_ATTEMPTS ]]; then
                echo "Retrying in $RETRY_DELAY seconds..."
                sleep $RETRY_DELAY
                ((attempt++))
            fi
        fi
    done
    
    if [[ $exit_code -ne 0 ]]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - $benchmark_name benchmark failed after $MAX_ATTEMPTS attempts."
        return 1
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') - $benchmark_name benchmark succeeded on attempt $attempt."
        return 0
    fi
}

# Main execution
echo "====================================================================="
echo "OpenHands Benchmark Runner"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "====================================================================="
echo "Model config: $MODEL_CONFIG"
echo "Git version: $GIT_VERSION"
echo "Agent: $AGENT"
echo "Eval limit: $EVAL_LIMIT"
echo "Number of workers: $NUM_WORKERS"
echo "Maximum retry attempts: $MAX_ATTEMPTS"
echo "Retry delay: $RETRY_DELAY seconds"
echo "====================================================================="

# Check for Docker
check_docker

# Run polyglot_benchmark
echo "====================================================================="
echo "Running polyglot_benchmark"
echo "====================================================================="
run_with_retry "./evaluation/benchmarks/polyglot_benchmark/scripts/run_infer.sh $MODEL_CONFIG $GIT_VERSION $AGENT $EVAL_LIMIT $NUM_WORKERS eval" "polyglot_benchmark"
POLYGLOT_RESULT=$?

# Run aider_bench
echo "====================================================================="
echo "Running aider_bench"
echo "====================================================================="
run_with_retry "./evaluation/benchmarks/aider_bench/scripts/run_infer.sh $MODEL_CONFIG $GIT_VERSION $AGENT $EVAL_LIMIT $NUM_WORKERS \"\" eval" "aider_bench"
AIDER_RESULT=$?

# Summary
echo "====================================================================="
echo "Benchmark Run Summary - Completed at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "====================================================================="
echo "polyglot_benchmark: $([ $POLYGLOT_RESULT -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
echo "aider_bench: $([ $AIDER_RESULT -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
echo "====================================================================="

# Exit with success only if both benchmarks succeeded
if [[ $POLYGLOT_RESULT -eq 0 && $AIDER_RESULT -eq 0 ]]; then
    echo "All benchmarks completed successfully."
    exit 0
else
    echo "One or more benchmarks failed."
    exit 1
fi