"""Direct patch for LiteLLM to use prefix-based conversations.

This script directly patches the LiteLLM completion function to use prefix-based conversations,
without relying on any complex imports or dependencies.
"""

import copy
import logging
import re
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import litellm
import litellm

# Function to transform messages to prefix format
def transform_to_prefix_format(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Transform standard messages into prefix-based format.
    
    In this format, the assistant's previous responses and observations are 
    combined into a growing narrative that's included as a prefix in subsequent turns.
    
    Args:
        messages: The messages in standard format
    
    Returns:
        The messages in prefix-based format
    """
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
    
    def patched_completion(model: str, messages: List[Dict[str, Any]], **kwargs):
        """Patched version of litellm.completion that uses prefix-based messages."""
        # Transform messages to prefix format
        transformed_messages = transform_to_prefix_format(messages)
        
        # Log the transformed messages
        logger.debug(f"Original messages: {messages}")
        logger.debug(f"Transformed messages: {transformed_messages}")
        
        # Call the original completion function with the transformed messages
        return original_completion(model=model, messages=transformed_messages, **kwargs)
    
    # Replace the original completion function with our patched version
    litellm.completion = patched_completion
    
    logger.info("Successfully patched litellm.completion to use prefix-based messages")
    
    return original_completion

# Function to restore the original litellm.completion
def restore_litellm_completion(original_completion):
    """Restore the original litellm.completion function."""
    litellm.completion = original_completion
    logger.info("Successfully restored litellm.completion")

if __name__ == "__main__":
    # Example usage
    original_completion = patch_litellm_completion()
    
    try:
        # Use litellm.completion with prefix-based messages
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"}
            ]
        )
        print(response)
    finally:
        # Restore the original litellm.completion
        restore_litellm_completion(original_completion)