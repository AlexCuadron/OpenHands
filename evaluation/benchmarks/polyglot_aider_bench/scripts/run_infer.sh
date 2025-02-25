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

# Check if using positional arguments (old style)
if [[ $# -ge 5 && "$1" != "--"* ]]; then
    # Old style: <model> <commit> <agent> <max_iters> <num_workers>
    MODEL="$1"
    COMMIT="$2"
    AGENT_CLS="$3"
    MAX_ITERS="$4"
    EVAL_NUM_WORKERS="$5"

    # Convert to new style arguments
    LLM_CONFIG="configs/llm/${MODEL}.yaml"
    EVAL_NOTE="${COMMIT}"
    MAX_ITERATIONS="--max-iterations ${MAX_ITERS}"
else
    # Parse named arguments (new style)
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
fi

# Check required arguments
if [ -z "$LLM_CONFIG" ]; then
    echo "Error: LLM config is required"
    echo "Usage:"
    echo "  Old style: $0 <model> <commit> <agent> <max_iters> <num_workers>"
    echo "  New style: $0 --llm-config <config> --agent-cls <agent> [other options]"
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
    ${EVAL_IDS:+--eval-ids "$EVAL_IDS"} \
    ${MAX_ITERATIONS:-}