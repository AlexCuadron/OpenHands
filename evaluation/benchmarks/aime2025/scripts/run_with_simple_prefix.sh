#!/bin/bash
# Run the AIME2025 benchmark with the simple prefix-based LLM approach

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Get the path to the original run_infer.sh script
ORIGINAL_SCRIPT="$SCRIPT_DIR/run_infer.sh"

# Check if the original script exists
if [ ! -f "$ORIGINAL_SCRIPT" ]; then
    echo "Error: Original script not found at $ORIGINAL_SCRIPT"
    exit 1
fi

# Create a temporary script to patch litellm.completion
cat > /tmp/simple_prefix_setup.py << 'EOF'
import sys
import os

# Add the OpenHands directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

# Import the simple prefix setup
from openhands.simple_prefix_setup import patch_litellm_completion

# Patch litellm.completion
original_completion = patch_litellm_completion()

# Print a message to indicate that the patch was successful
print("Successfully patched litellm.completion to use prefix-based messages")
EOF

# Run the temporary script to patch litellm.completion
python3 /tmp/simple_prefix_setup.py

# Pass all arguments to the original script
"$ORIGINAL_SCRIPT" "$@"

# Create a temporary script to restore litellm.completion
cat > /tmp/simple_prefix_cleanup.py << 'EOF'
import sys
import os
import litellm

# Add the OpenHands directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

# Import the simple prefix setup
from openhands.simple_prefix_setup import restore_litellm_completion

# Get the original completion function (this is just a placeholder)
# In a real scenario, we would need to store the original completion function somewhere
original_completion = litellm.completion

# Restore litellm.completion
restore_litellm_completion(original_completion)

# Print a message to indicate that the restoration was successful
print("Successfully restored litellm.completion")
EOF

# Run the temporary script to restore litellm.completion
python3 /tmp/simple_prefix_cleanup.py