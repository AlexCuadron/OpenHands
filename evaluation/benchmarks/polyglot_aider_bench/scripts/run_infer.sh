#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.." || exit 1

# Default values
AGENT_CLS="CodeActAgent"
EVAL_NOTE=""
EVAL_OUTPUT_DIR="eval_output"
EVAL_NUM_WORKERS=1
EVAL_N_LIMIT=-1
LLM_CONFIG=""
EVAL_IDS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --agent-cls)
            AGENT_CLS="$2"
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
        --llm-config)
            LLM_CONFIG="$2"
            shift 2
            ;;
        --eval-ids)
            EVAL_IDS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$LLM_CONFIG" ]; then
    echo "Error: --llm-config is required"
    exit 1
fi

# Run the evaluation
python3 run_infer.py \
    --agent-cls "$AGENT_CLS" \
    --eval-note "$EVAL_NOTE" \
    --eval-output-dir "$EVAL_OUTPUT_DIR" \
    --eval-num-workers "$EVAL_NUM_WORKERS" \
    --eval-n-limit "$EVAL_N_LIMIT" \
    --llm-config "$LLM_CONFIG" \
    ${EVAL_IDS:+--eval-ids "$EVAL_IDS"}