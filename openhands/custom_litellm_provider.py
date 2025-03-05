"""Custom LiteLLM provider for vLLM models with special formatting requirements."""

import copy
import json
import httpx
from typing import Dict, List, Any, Optional, Union
import litellm
from litellm.utils import ModelResponse

# Track if we're in a tool call sequence
_tool_call_in_progress = False
_last_messages = None

def custom_vllm_completion(
    model: str,
    messages: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs
) -> ModelResponse:
    """Custom completion function for vLLM models with special formatting requirements.
    
    This function modifies the request to vLLM to handle tool calls properly.
    """
    global _tool_call_in_progress, _last_messages
    
    # Deep copy the messages to avoid modifying the original
    messages_copy = copy.deepcopy(messages)
    
    # Check if this is a continuation after a tool call
    is_continuation = False
    if _tool_call_in_progress and _last_messages:
        # Compare the current messages with the last messages
        # If they share the same prefix, this is likely a continuation
        if len(messages) > len(_last_messages):
            is_continuation = True
            for i, last_msg in enumerate(_last_messages):
                if i >= len(messages) or messages[i]["role"] != last_msg["role"]:
                    is_continuation = False
                    break
                if messages[i]["role"] == "system" and last_msg["role"] == "system":
                    # Don't compare content for system messages as they might be different
                    continue
                if messages[i].get("content") != last_msg.get("content"):
                    is_continuation = False
                    break
    
    # If this is a continuation, add a special parameter to the request
    if is_continuation:
        # Add a custom parameter to indicate this is a continuation
        kwargs["continue_conversation"] = True
    
    # Store the current messages for future comparison
    _last_messages = copy.deepcopy(messages)
    
    # Check if the last message is a tool response
    if messages and messages[-1]["role"] == "tool":
        _tool_call_in_progress = True
    else:
        # If the last message is from the assistant or user, we're not in a tool call sequence
        _tool_call_in_progress = False
    
    # Make the actual API call using LiteLLM's OpenAI provider
    return litellm.completion(
        model=model,
        messages=messages_copy,
        api_key=api_key,
        base_url=base_url,
        **kwargs
    )

# Register our custom provider with LiteLLM
litellm.register_provider("custom_vllm", custom_vllm_completion)