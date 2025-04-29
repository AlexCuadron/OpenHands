#!/bin/bash

set -e

# Display usage information
function show_usage {
  echo "Usage: $0 [options]"
  echo ""
  echo "Options:"
  echo "  --help                 Show this help message"
  echo "  --model MODEL          Model configuration (default: eval_gpt4_1106_preview)"
  echo "  --agent AGENT          Agent class (default: CodeActAgent)"
  echo "  --limit LIMIT          Evaluation limit (default: -1 for all)"
  echo "  --workers WORKERS      Number of workers (default: 1)"
  echo "  --ids IDS              Comma-separated list of instance IDs"
  echo "  --languages LANGUAGES  Comma-separated list of languages"
  echo "  --one-per-language     Test one instance per language"
  echo "  --eval                 Run evaluation after benchmark"
  echo ""
  echo "Legacy positional arguments are still supported:"
  echo "  $0 MODEL_CONFIG GIT_VERSION AGENT EVAL_LIMIT EVAL_NUM_WORKERS EVAL_IDS EVAL_LANGUAGES"
  exit 0
}

# Parse named arguments
ONE_PER_LANGUAGE=false
RUN_EVALUATION=false
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --help)
      show_usage
      ;;
    --model)
      MODEL_CONFIG="$2"
      shift 2
      ;;
    --agent)
      AGENT="$2"
      shift 2
      ;;
    --limit)
      EVAL_LIMIT="$2"
      shift 2
      ;;
    --workers)
      EVAL_NUM_WORKERS="$2"
      shift 2
      ;;
    --ids)
      EVAL_IDS="$2"
      shift 2
      ;;
    --languages)
      EVAL_LANGUAGES="$2"
      shift 2
      ;;
    --one-per-language)
      ONE_PER_LANGUAGE=true
      shift
      ;;
    --eval)
      RUN_EVALUATION=true
      shift
      ;;
    eval)
      # Special case for the 'eval' parameter in the positional arguments
      RUN_EVALUATION=true
      shift
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

# Restore positional parameters
set -- "${POSITIONAL_ARGS[@]}"

# Default values (if not set by named arguments)
MODEL_CONFIG=${MODEL_CONFIG:-${1:-"eval_gpt4_1106_preview"}}
GIT_VERSION=${2:-"HEAD"}
AGENT=${AGENT:-${3:-"CodeActAgent"}}
EVAL_LIMIT=${EVAL_LIMIT:-${4:-"-1"}}
EVAL_NUM_WORKERS=${EVAL_NUM_WORKERS:-${5:-"1"}}
EVAL_IDS=${EVAL_IDS:-${6:-""}}
EVAL_LANGUAGES=${EVAL_LANGUAGES:-${7:-""}}

# Set environment variables
export USE_UNIT_TESTS=${USE_UNIT_TESTS:-"true"}
export NO_DOCKER=${NO_DOCKER:-"false"}

# Check if we have a local Docker image env file
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BENCHMARK_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"
DOCKER_ENV_FILE="${BENCHMARK_DIR}/docker_image.env"

# Set BUILD_LOCAL_DOCKER to true by default if not specified
export BUILD_LOCAL_DOCKER=${BUILD_LOCAL_DOCKER:-"true"}

if [ -f "$DOCKER_ENV_FILE" ]; then
  echo "Loading Docker image configuration from $DOCKER_ENV_FILE"
  source "$DOCKER_ENV_FILE"
else
  # If no local image is available, use the default
  export POLYGLOT_DOCKER_IMAGE=${POLYGLOT_DOCKER_IMAGE:-"ghcr.io/opendevin/eval-polyglot:v1.0.0"}
  
  # Try to pull the image first
  echo "Trying to pull Docker image: $POLYGLOT_DOCKER_IMAGE"
  if ! docker pull "$POLYGLOT_DOCKER_IMAGE" 2>/dev/null; then
    echo "Failed to pull Docker image: $POLYGLOT_DOCKER_IMAGE"
    
    # Build a local Docker image if pulling fails and BUILD_LOCAL_DOCKER is true
    if [ "$BUILD_LOCAL_DOCKER" = "true" ]; then
      echo "Building local Docker image..."
      "${SCRIPT_DIR}/build_local_docker.sh"
      source "$DOCKER_ENV_FILE"
    else
      echo "WARNING: Docker image not found and BUILD_LOCAL_DOCKER is not set to true."
      echo "You can build a local Docker image by running:"
      echo "  ${SCRIPT_DIR}/build_local_docker.sh"
      echo "Or set BUILD_LOCAL_DOCKER=true to build it automatically."
    fi
  else
    echo "Successfully pulled Docker image: $POLYGLOT_DOCKER_IMAGE"
  fi
fi

echo "Using Docker image: $POLYGLOT_DOCKER_IMAGE"

# Try to find the polyglot-benchmark repository
if [ -z "$POLYGLOT_BENCHMARK_PATH" ]; then
  # Check common locations
  POSSIBLE_PATHS=(
    "/workspace/polyglot-benchmark"
    "$HOME/polyglot-benchmark"
    "$HOME/thereal/polyglot-benchmark"
    "$(git rev-parse --show-toplevel)/polyglot-benchmark"
    "$(pwd)/polyglot-benchmark"
  )
  
  for path in "${POSSIBLE_PATHS[@]}"; do
    if [ -d "$path" ]; then
      export POLYGLOT_BENCHMARK_PATH="$path"
      echo "Found polyglot-benchmark repository at: $POLYGLOT_BENCHMARK_PATH"
      break
    fi
  done
fi

# If still not found, try to clone it
if [ -z "$POLYGLOT_BENCHMARK_PATH" ] || [ ! -d "$POLYGLOT_BENCHMARK_PATH" ]; then
  echo "Polyglot benchmark repository not found. Attempting to clone it..."
  CLONE_DIR="$(git rev-parse --show-toplevel)/polyglot-benchmark"
  git clone https://github.com/Aider-AI/polyglot-benchmark.git "$CLONE_DIR"
  if [ $? -eq 0 ]; then
    export POLYGLOT_BENCHMARK_PATH="$CLONE_DIR"
    echo "Successfully cloned polyglot-benchmark to $POLYGLOT_BENCHMARK_PATH"
  else
    echo "Failed to clone polyglot-benchmark. Please set POLYGLOT_BENCHMARK_PATH manually."
    exit 1
  fi
fi

# Add additional arguments based on provided parameters
ARGS="--agent-cls ${AGENT} --llm-config ${MODEL_CONFIG} --max-iterations 30 --eval-num-workers ${EVAL_NUM_WORKERS}"

if [ "${EVAL_LIMIT}" != "-1" ]; then
  ARGS="${ARGS} --eval-n-limit ${EVAL_LIMIT}"
fi

# Only pass eval-ids if it's not "eval" (which is a special parameter for evaluation mode)
if [ -n "${EVAL_IDS}" ] && [ "${EVAL_IDS}" != "eval" ]; then
  ARGS="${ARGS} --eval-ids ${EVAL_IDS}"
fi

if [ -n "${EVAL_LANGUAGES}" ]; then
  ARGS="${ARGS} --eval-languages ${EVAL_LANGUAGES}"
fi

# Change to the repository root directory
cd "$(git rev-parse --show-toplevel)"

# If one-per-language mode is enabled
if [ "$ONE_PER_LANGUAGE" = true ]; then
  echo "Running one instance per language mode..."
  
  # Define the languages to test
  LANGUAGES=("python" "javascript" "rust" "go" "cpp" "java")
  
  # Create a temporary directory for results
  RESULTS_DIR="evaluation/evaluation_outputs/one_per_language_test"
  mkdir -p "$RESULTS_DIR"
  
  # Summary file
  SUMMARY_FILE="$RESULTS_DIR/summary.txt"
  echo "POLYGLOT BENCHMARK - ONE INSTANCE PER LANGUAGE TEST" > "$SUMMARY_FILE"
  echo "=================================================" >> "$SUMMARY_FILE"
  echo "Model: $MODEL_CONFIG" >> "$SUMMARY_FILE"
  echo "Agent: $AGENT" >> "$SUMMARY_FILE"
  echo "Date: $(date)" >> "$SUMMARY_FILE"
  echo "=================================================" >> "$SUMMARY_FILE"
  echo "" >> "$SUMMARY_FILE"
  
  # Test each language
  for LANG in "${LANGUAGES[@]}"; do
    echo ""
    echo "===== Testing language: $LANG ====="
    echo ""
    
    # Run with one instance for this language
    LANG_ARGS="--agent-cls ${AGENT} --llm-config ${MODEL_CONFIG} --max-iterations 30 --eval-num-workers 1 --eval-n-limit 1 --eval-languages ${LANG} --eval-note one_per_language_${LANG}"
    
    # Run the evaluation for this language
    if poetry run python -m evaluation.benchmarks.polyglot_benchmark.run_infer ${LANG_ARGS}; then
      RESULT="PASSED"
    else
      RESULT="FAILED"
    fi
    
    # Add to summary
    echo "${LANG}: ${RESULT}" >> "$SUMMARY_FILE"
  done
  
  # Display summary
  echo ""
  echo "===== TEST SUMMARY ====="
  cat "$SUMMARY_FILE"
  echo ""
  echo "Detailed results available in: $RESULTS_DIR"
  
  # Run evaluation if requested
  if [ "$RUN_EVALUATION" = true ]; then
    echo ""
    echo "======================================"
    echo "Running detailed evaluation on results..."
    echo "======================================"
    echo ""
    
    # Evaluate each language's results
    for LANG in "${LANGUAGES[@]}"; do
      # Try to find the output directory for this language
      LANG_OUTPUT_DIR=$(find evaluation/evaluation_outputs -type d -name "*one_per_language_${LANG}*" 2>/dev/null | sort -r | head -n 1)
      
      if [ -z "$LANG_OUTPUT_DIR" ]; then
        LANG_OUTPUT_DIR=$(find . -path "*/evaluation_outputs/*" -type d -name "*one_per_language_${LANG}*" 2>/dev/null | sort -r | head -n 1)
      fi
      
      if [ -z "$LANG_OUTPUT_DIR" ]; then
        LANG_OUTPUT_DIR="evaluation/evaluation_outputs/one_per_language_${LANG}"
      fi
      
      LANG_OUTPUT_FILE="${LANG_OUTPUT_DIR}/output.jsonl"
      
      # Print the language output directory and file for debugging
      echo ""
      echo "Language: $LANG"
      echo "Output directory: $LANG_OUTPUT_DIR"
      echo "Output file: $LANG_OUTPUT_FILE"
      
      if [ -f "$LANG_OUTPUT_FILE" ]; then
        echo ""
        echo "===== Evaluating $LANG results ====="
        echo ""
        echo "Evaluating results in: $LANG_OUTPUT_FILE"
        
        # Save the evaluation results
        EVAL_RESULTS_FILE="${LANG_OUTPUT_DIR}/evaluation_results.txt"
        echo "Saving evaluation results to: $EVAL_RESULTS_FILE"
        poetry run python evaluation/benchmarks/polyglot_benchmark/scripts/summarize_results.py "$LANG_OUTPUT_FILE" > "$EVAL_RESULTS_FILE"
      fi
    done
    
    echo ""
    echo "Detailed evaluation complete."
  fi
else
  # Run the normal evaluation
  poetry run python -m evaluation.benchmarks.polyglot_benchmark.run_infer ${ARGS}
  
  # Run evaluation if requested
  if [ "$RUN_EVALUATION" = true ]; then
    echo ""
    echo "======================================"
    echo "Running evaluation on results..."
    echo "======================================"
    echo ""
    
    # Get the output directory - first try the default location
    OUTPUT_DIR=$(find evaluation/evaluation_outputs -path "*/PolyglotBenchmark/$AGENT/*" -type d -name "*tools_bash+finish+str_replace*" 2>/dev/null | sort -r | head -n 1)
    
    # If not found, try to find it anywhere under evaluation_outputs
    if [ -z "$OUTPUT_DIR" ]; then
      OUTPUT_DIR=$(find . -path "*/evaluation_outputs/*" -path "*/PolyglotBenchmark/$AGENT/*" -type d -name "*tools_bash+finish+str_replace*" 2>/dev/null | sort -r | head -n 1)
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
    
    if [ -f "$OUTPUT_FILE" ]; then
      echo "Evaluating results in: $OUTPUT_FILE"
      poetry run python evaluation/benchmarks/polyglot_benchmark/scripts/summarize_results.py "$OUTPUT_FILE"
      
      # Save the evaluation results
      EVAL_RESULTS_FILE="$OUTPUT_DIR/evaluation_results.txt"
      echo "Saving evaluation results to: $EVAL_RESULTS_FILE"
      poetry run python evaluation/benchmarks/polyglot_benchmark/scripts/summarize_results.py "$OUTPUT_FILE" > "$EVAL_RESULTS_FILE"
      
      echo ""
      echo "Evaluation complete. Results saved to: $EVAL_RESULTS_FILE"
    else
      echo "Error: Output file not found: $OUTPUT_FILE"
      echo "Cannot run evaluation."
    fi
  fi
fi