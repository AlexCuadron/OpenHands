"""Conditional Prefix LLM module.

This module provides a direct way to use the prefix-based LLM approach
when running the AIME2025 benchmark, without requiring the full OpenHands codebase.
"""

import os
import sys
import logging
import importlib
from typing import Optional, Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variable to store the original create_llm function
original_create_llm = None

def is_running_aime2025():
    """Check if we're running the AIME2025 benchmark.
    
    This function checks the command line arguments and environment variables
    to determine if we're running the AIME2025 benchmark.
    
    Returns:
        bool: True if we're running the AIME2025 benchmark, False otherwise.
    """
    # Check command line arguments
    cmd_args = ' '.join(sys.argv)
    if 'aime2025' in cmd_args:
        return True
    
    # Check environment variables
    env_vars = os.environ.get('OPENHANDS_BENCHMARK', '')
    if 'aime2025' in env_vars.lower():
        return True
    
    # Check if the script path contains aime2025
    script_path = os.path.abspath(sys.argv[0])
    if 'aime2025' in script_path:
        return True
    
    return False

def transform_to_prefix_format(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Transform standard OpenHands message format into prefix-based format.
    
    In this format, the assistant's previous responses and observations are 
    combined into a growing narrative that's included as a prefix in subsequent turns.
    
    Args:
        messages: The messages in standard OpenHands format
    
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
        if msg["role"] == "system":
            system_content += msg.get("content", "") + "\n\n"
    
    # Find the first user message
    first_user_idx = -1
    for i, msg in enumerate(messages):
        if msg["role"] == "user":
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
        role = msg["role"]
        content = msg.get("content", "")
        
        if role == "assistant":
            # Add to the current turn
            current_turn.append({"role": "assistant", "content": content})
        elif role == "tool":
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

def patch_litellm_completion():
    """Patch the litellm.completion function to use prefix-based format for AIME2025."""
    try:
        import litellm
        
        # Store the original completion function
        original_completion = litellm.completion
        
        # Define the new completion function
        def prefix_completion(model, messages, **kwargs):
            # Only transform messages for AIME2025 benchmark
            if is_running_aime2025():
                logger.info("Using prefix-based format for AIME2025 benchmark")
                transformed_messages = transform_to_prefix_format(messages)
                return original_completion(model=model, messages=transformed_messages, **kwargs)
            else:
                return original_completion(model=model, messages=messages, **kwargs)
        
        # Replace the original completion function
        litellm.completion = prefix_completion
        logger.info("Patched litellm.completion function")
        
        return original_completion
    except ImportError:
        logger.warning("litellm module not found, skipping patch")
        return None

def patch_llm_creation():
    """Patch the LLM creation function in the main module.
    
    This is a simplified version that doesn't require importing the full OpenHands codebase.
    Instead, it directly patches the litellm.completion function.
    """
    global original_create_llm
    
    # Patch the litellm.completion function
    original_completion = patch_litellm_completion()
    
    logger.info("Patched LLM creation function")
    
    return original_completion

def restore_llm_creation(original_completion):
    """Restore the original LLM creation function."""
    try:
        import litellm
        if original_completion:
            litellm.completion = original_completion
            logger.info("Restored original litellm.completion function")
    except ImportError:
        logger.warning("litellm module not found, skipping restore")
    
    logger.info("Restored original LLM creation function")