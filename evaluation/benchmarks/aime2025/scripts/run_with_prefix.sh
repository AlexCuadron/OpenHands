#!/bin/bash
# Run the AIME2025 benchmark with the prefix-based LLM approach

# Set environment variable to indicate we're running AIME2025
export OPENHANDS_BENCHMARK="aime2025"

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Get the path to the original run_infer.sh script
ORIGINAL_SCRIPT="$SCRIPT_DIR/run_infer.sh"

# Check if the original script exists
if [ ! -f "$ORIGINAL_SCRIPT" ]; then
    echo "Error: Original script not found at $ORIGINAL_SCRIPT"
    exit 1
fi

# Import the conditional prefix LLM module before running the original script
PYTHON_SETUP="
import sys
import os
sys.path.insert(0, os.path.join('$(dirname "$SCRIPT_DIR")', '..', '..', '..'))
from openhands.conditional_prefix_llm import patch_llm_creation
original_create_llm = patch_llm_creation()
"

# Run the original script with the same arguments
echo "Running AIME2025 benchmark with prefix-based LLM approach..."
echo "$PYTHON_SETUP" > /tmp/prefix_setup.py
python3 /tmp/prefix_setup.py

# Pass all arguments to the original script
"$ORIGINAL_SCRIPT" "$@"

# Restore the original LLM creation function
PYTHON_CLEANUP="
import sys
import os
sys.path.insert(0, os.path.join('$(dirname "$SCRIPT_DIR")', '..', '..', '..'))
from openhands.conditional_prefix_llm import restore_llm_creation
from openhands.core.main import create_llm
restore_llm_creation(create_llm)
"

echo "$PYTHON_CLEANUP" > /tmp/prefix_cleanup.py
python3 /tmp/prefix_cleanup.py

echo "Finished running AIME2025 benchmark with prefix-based LLM approach."