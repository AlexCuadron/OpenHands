#!/bin/bash
# Run the AIME2025 benchmark with the direct prefix patch

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
cat > /tmp/direct_prefix_patch.py << 'EOF'
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import litellm
try:
    import litellm
    logger.info("Successfully imported litellm")
except ImportError as e:
    logger.error(f"Failed to import litellm: {e}")
    sys.exit(1)

# Function to transform messages to prefix format
def transform_to_prefix_format(messages):
    """Transform standard messages into prefix-based format."""
    if not messages:
        return []
    
    # Initialize the transformed messages list
    transformed_messages = []
    
    # Extract system messages if any
    system_content = ""
    for msg in messages:
        if msg.get("role") == "system":
            system_content += msg.get("content", "") + "\n\n"
    
    # Find the first user message
    first_user_idx = -1
    for i, msg in enumerate(messages):
        if msg.get("role") == "user":
            first_user_idx = i
            break
    
    if first_user_idx == -1:
        # No user message found, return empty list
        return []
    
    # Add the first user message with system content prepended if any
    first_user_content = messages[first_user_idx].get("content", "")
    if system_content:
        first_user_content = f"{system_content}{first_user_content}"
    
    transformed_messages.append({
        "role": "user",
        "content": first_user_content
    })
    
    # Process the remaining messages to build the assistant's narrative
    assistant_narrative = ""
    
    # Track the current conversation turn
    current_turn = []
    
    for i in range(first_user_idx + 1, len(messages)):
        msg = messages[i]
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role == "assistant":
            # Add to the current turn
            current_turn.append({"role": "assistant", "content": content})
        elif role == "tool" or role == "function":
            # Add observation to the current turn
            current_turn.append({"role": "observation", "content": content})
        elif role == "user":
            # Process the current turn and add to the narrative
            if current_turn:
                for turn_msg in current_turn:
                    if turn_msg["role"] == "assistant":
                        assistant_narrative += turn_msg["content"] + "\n"
                    elif turn_msg["role"] == "observation":
                        assistant_narrative += f"Observation: {turn_msg['content']}\n"
                
                assistant_narrative += "\n"
                current_turn = []
            
            # Add the assistant narrative as a prefix
            if assistant_narrative:
                transformed_messages.append({
                    "role": "assistant",
                    "content": assistant_narrative.strip(),
                    "prefix": True
                })
            
            # Add the new user message
            transformed_messages.append({
                "role": "user",
                "content": content
            })
    
    # Process any remaining turn
    if current_turn:
        for turn_msg in current_turn:
            if turn_msg["role"] == "assistant":
                assistant_narrative += turn_msg["content"] + "\n"
            elif turn_msg["role"] == "observation":
                assistant_narrative += f"Observation: {turn_msg['content']}\n"
    
    # Add any remaining assistant narrative as a prefix
    if assistant_narrative:
        transformed_messages.append({
            "role": "assistant",
            "content": assistant_narrative.strip(),
            "prefix": True
        })
    
    return transformed_messages

# Function to patch litellm.completion to use prefix-based messages
def patch_litellm_completion():
    """Patch litellm.completion to use prefix-based messages."""
    original_completion = litellm.completion
    
    def patched_completion(*args, **kwargs):
        """Patched version of litellm.completion that uses prefix-based messages."""
        # Extract messages from args or kwargs
        messages = None
        if len(args) > 0:
            messages = args[0]
        elif 'messages' in kwargs:
            messages = kwargs['messages']
        
        if messages:
            # Transform messages to prefix format
            transformed_messages = transform_to_prefix_format(messages)
            
            # Log the transformed messages
            logger.debug(f"Original messages: {messages}")
            logger.debug(f"Transformed messages: {transformed_messages}")
            
            # Update args or kwargs with transformed messages
            if len(args) > 0:
                args = (transformed_messages,) + args[1:]
            else:
                kwargs['messages'] = transformed_messages
        
        # Call the original completion function with the transformed messages
        return original_completion(*args, **kwargs)
    
    # Replace the original completion function with our patched version
    litellm.completion = patched_completion
    
    logger.info("Successfully patched litellm.completion to use prefix-based messages")
    
    return original_completion

# Patch litellm.completion
original_completion = patch_litellm_completion()

# Print a message to indicate that the patch was successful
print("Successfully patched litellm.completion to use prefix-based messages")
EOF

# Run the temporary script to patch litellm.completion
python3 /tmp/direct_prefix_patch.py

# Pass all arguments to the original script
"$ORIGINAL_SCRIPT" "$@"