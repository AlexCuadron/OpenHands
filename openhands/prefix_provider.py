"""Custom LiteLLM provider that uses the prefix feature for conversations.

This provider transforms standard OpenHands message format into a prefix-based format
where the assistant's previous responses and observations are combined into a growing
narrative that's included as a prefix in subsequent turns.
"""

import copy
import logging
from typing import Dict, List, Any, Optional, Union
import litellm
from litellm.utils import ModelResponse

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def prefix_completion(
    model: str,
    messages: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs
) -> ModelResponse:
    """Custom completion function that uses the prefix feature for conversations.
    
    This function transforms standard OpenHands message format into a prefix-based format
    where the assistant's previous responses and observations are combined into a growing
    narrative that's included as a prefix in subsequent turns.
    
    Args:
        model: The model to use for completion
        messages: The messages in standard OpenHands format
        api_key: The API key to use
        base_url: The base URL for the API
        **kwargs: Additional arguments to pass to the completion function
    
    Returns:
        A ModelResponse object
    """
    # Deep copy the messages to avoid modifying the original
    messages_copy = copy.deepcopy(messages)
    
    # Log the original messages for debugging
    logger.debug(f"Original messages: {messages_copy}")
    
    # Transform the messages into prefix-based format
    transformed_messages = transform_to_prefix_format(messages_copy)
    
    # Log the transformed messages for debugging
    logger.debug(f"Transformed messages: {transformed_messages}")
    
    # Make the API call using LiteLLM's completion function
    response = litellm.completion(
        model=model,
        messages=transformed_messages,
        api_key=api_key,
        base_url=base_url,
        **kwargs
    )
    
    # Log the response for debugging
    logger.debug(f"Response: {response}")
    
    return response

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

# Register our custom provider with LiteLLM
litellm.register_provider("prefix_provider", prefix_completion)