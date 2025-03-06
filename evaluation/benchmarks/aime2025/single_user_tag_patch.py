"""
This module provides functionality to ensure that there is always at most ONE user tag.
"""

from typing import Any, Dict, List, Optional

from openhands.controller.state.state import State
from openhands.core.logger import openhands_logger as logger
from openhands.llm.llm import LLM


def patch_llm_for_single_user_tag(state: State) -> None:
    """
    Patch the LLM to ensure that there is always at most ONE user tag.
    
    Args:
        state: The current state of the agent
    """
    if not hasattr(state, 'agent') or not hasattr(state.agent, 'llm'):
        logger.warning('Cannot patch LLM for single user tag: agent or llm not found')
        return
    
    logger.info('Patching LLM to ensure at most ONE user tag')
    
    # Get the LLM instance
    llm = state.agent.llm
    
    # Store the original completion method
    original_completion = llm.completion
    
    # Create a wrapper function that ensures at most ONE user tag
    def completion_with_single_user_tag(*args, **kwargs):
        """
        A wrapper for the completion method that ensures at most ONE user tag.
        
        Args:
            *args: The positional arguments to pass to the original completion method
            **kwargs: The keyword arguments to pass to the original completion method
            
        Returns:
            The result of the original completion method
        """
        # Check if messages are in the kwargs
        if 'messages' in kwargs:
            messages = kwargs['messages']
            
            # Create a new messages list with at most ONE user tag
            new_messages = []
            
            # Find all user messages
            user_messages = []
            for msg in messages:
                if isinstance(msg, dict) and msg.get('role') == 'user':
                    user_messages.append(msg)
            
            # If there are multiple user messages, combine them
            if len(user_messages) > 1:
                # Combine all user messages into one
                combined_user_content = '\n\n'.join(msg.get('content', '') for msg in user_messages)
                
                # Create a new user message with the combined content
                combined_user_message = {'role': 'user', 'content': combined_user_content}
                
                # Replace all user messages with the combined one
                for msg in messages:
                    if isinstance(msg, dict) and msg.get('role') == 'user':
                        if msg == user_messages[0]:  # Only keep the first user message
                            new_messages.append(combined_user_message)
                    else:
                        new_messages.append(msg)
                
                # Replace the messages in the kwargs
                kwargs['messages'] = new_messages
                
                logger.info(f'Combined {len(user_messages)} user messages into ONE user tag')
            
        # Call the original completion method
        return original_completion(*args, **kwargs)
    
    # Replace the completion method with our patched version
    llm.completion = completion_with_single_user_tag
    
    # Also patch the _completion method if it exists
    if hasattr(llm, '_completion'):
        original_inner_completion = llm._completion
        
        # Create a wrapper function for the inner completion method
        def inner_completion_with_single_user_tag(*args, **kwargs):
            """
            A wrapper for the inner completion method that ensures at most ONE user tag.
            
            Args:
                *args: The positional arguments to pass to the original inner completion method
                **kwargs: The keyword arguments to pass to the original inner completion method
                
            Returns:
                The result of the original inner completion method
            """
            # Check if messages are in the kwargs
            if 'messages' in kwargs:
                messages = kwargs['messages']
                
                # Create a new messages list with at most ONE user tag
                new_messages = []
                
                # Find all user messages
                user_messages = []
                for msg in messages:
                    if isinstance(msg, dict) and msg.get('role') == 'user':
                        user_messages.append(msg)
                
                # If there are multiple user messages, combine them
                if len(user_messages) > 1:
                    # Combine all user messages into one
                    combined_user_content = '\n\n'.join(msg.get('content', '') for msg in user_messages)
                    
                    # Create a new user message with the combined content
                    combined_user_message = {'role': 'user', 'content': combined_user_content}
                    
                    # Replace all user messages with the combined one
                    for msg in messages:
                        if isinstance(msg, dict) and msg.get('role') == 'user':
                            if msg == user_messages[0]:  # Only keep the first user message
                                new_messages.append(combined_user_message)
                        else:
                            new_messages.append(msg)
                    
                    # Replace the messages in the kwargs
                    kwargs['messages'] = new_messages
                    
                    logger.info(f'Combined {len(user_messages)} user messages into ONE user tag in inner completion')
                
            # Call the original inner completion method
            return original_inner_completion(*args, **kwargs)
        
        # Replace the inner completion method with our patched version
        llm._completion = inner_completion_with_single_user_tag
    
    logger.info('LLM patched to ensure at most ONE user tag')