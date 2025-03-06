"""
This module provides functionality to ensure that system calls are NEVER made in AIME2025.
"""

import functools
from typing import Any, Dict, List, Optional

from openhands.controller.state.state import State
from openhands.core.logger import openhands_logger as logger
from openhands.llm.llm import LLM


def patch_llm_to_never_use_system_calls_ever(state: State) -> None:
    """
    Patch the LLM to ensure that system calls are NEVER made in AIME2025.
    
    Args:
        state: The current state of the agent
    """
    if not hasattr(state, 'agent') or not hasattr(state.agent, 'llm'):
        logger.warning('Cannot patch LLM to never use system calls: agent or llm not found')
        return
    
    logger.info('Patching LLM to NEVER EVER use system calls')
    
    # Get the LLM instance
    llm = state.agent.llm
    
    # Patch all methods that might use system calls
    
    # 1. Patch the completion method
    if hasattr(llm, 'completion'):
        original_completion = llm.completion
        llm.completion = _create_no_system_wrapper(original_completion, 'completion')
    
    # 2. Patch the _completion method
    if hasattr(llm, '_completion'):
        original_inner_completion = llm._completion
        llm._completion = _create_no_system_wrapper(original_inner_completion, '_completion')
    
    # 3. Patch the _completion_unwrapped method
    if hasattr(llm, '_completion_unwrapped'):
        original_unwrapped = llm._completion_unwrapped
        llm._completion_unwrapped = _create_no_system_wrapper(original_unwrapped, '_completion_unwrapped')
    
    # 4. Patch litellm directly
    try:
        import litellm
        original_litellm_completion = litellm.completion
        
        @functools.wraps(original_litellm_completion)
        def patched_litellm_completion(*args, **kwargs):
            # Process messages to remove system roles
            if 'messages' in kwargs:
                kwargs['messages'] = _process_messages_to_remove_system(kwargs['messages'])
            
            # Call the original function
            return original_litellm_completion(*args, **kwargs)
        
        # Replace the original function
        litellm.completion = patched_litellm_completion
        logger.info('Patched litellm.completion to never use system calls')
    except ImportError:
        logger.warning('Could not patch litellm directly: module not found')
    except Exception as e:
        logger.warning(f'Could not patch litellm directly: {e}')
    
    logger.info('LLM patched to NEVER EVER use system calls')


def _create_no_system_wrapper(original_func, func_name):
    """
    Create a wrapper function that ensures no system calls are made.
    
    Args:
        original_func: The original function to wrap
        func_name: The name of the function (for logging)
        
    Returns:
        A wrapped function that ensures no system calls are made
    """
    @functools.wraps(original_func)
    def wrapper(*args, **kwargs):
        # Process messages to remove system roles
        if 'messages' in kwargs:
            kwargs['messages'] = _process_messages_to_remove_system(kwargs['messages'])
        
        # Call the original function
        return original_func(*args, **kwargs)
    
    logger.info(f'Created no-system wrapper for {func_name}')
    return wrapper


def _process_messages_to_remove_system(messages):
    """
    Process messages to remove system roles.
    
    Args:
        messages: The messages to process
        
    Returns:
        The processed messages with no system roles
    """
    if not messages:
        return messages
    
    # Create a new messages list
    new_messages = []
    
    # Find all system and user messages
    system_content = []
    user_messages = []
    
    for msg in messages:
        if isinstance(msg, dict):
            if msg.get('role') == 'system':
                system_content.append(msg.get('content', ''))
            elif msg.get('role') == 'user':
                user_messages.append(msg)
    
    # If there are system messages but no user messages, create a user message
    if system_content and not user_messages:
        combined_system_content = '\n\n'.join(system_content)
        new_messages.append({'role': 'user', 'content': combined_system_content})
    
    # If there are both system and user messages, combine them
    elif system_content and user_messages:
        combined_system_content = '\n\n'.join(system_content)
        
        # Add the system content to the first user message
        first_user_msg = user_messages[0].copy()
        first_user_msg['content'] = combined_system_content + '\n\n' + first_user_msg.get('content', '')
        
        # Add the modified first user message
        new_messages.append(first_user_msg)
        
        # Add the rest of the user messages (if any)
        for msg in user_messages[1:]:
            new_messages.append(msg)
    
    # If there are no system messages, just add all messages except system ones
    else:
        for msg in messages:
            if isinstance(msg, dict) and msg.get('role') != 'system':
                new_messages.append(msg)
            elif not isinstance(msg, dict):
                new_messages.append(msg)
    
    # Add all non-system, non-user messages
    for msg in messages:
        if isinstance(msg, dict) and msg.get('role') != 'system' and msg.get('role') != 'user':
            if msg not in new_messages:  # Avoid duplicates
                new_messages.append(msg)
        elif not isinstance(msg, dict) and msg not in new_messages:
            new_messages.append(msg)
    
    # Ensure there's at least one message
    if not new_messages and messages:
        # If there are no messages left, create a dummy user message
        new_messages.append({'role': 'user', 'content': 'Please help me with this task.'})
    
    return new_messages