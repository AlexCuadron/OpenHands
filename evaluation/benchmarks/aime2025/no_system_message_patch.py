"""
This module provides functionality to ensure that system messages are never used.
"""

from typing import Any, Dict, List, Optional

from openhands.controller.state.state import State
from openhands.core.logger import openhands_logger as logger
from openhands.llm.llm import LLM


def patch_llm_to_never_use_system_messages(state: State) -> None:
    """
    Patch the LLM to ensure that system messages are never used.
    
    Args:
        state: The current state of the agent
    """
    if not hasattr(state, 'agent') or not hasattr(state.agent, 'llm'):
        logger.warning('Cannot patch LLM to never use system messages: agent or llm not found')
        return
    
    logger.info('Patching LLM to never use system messages')
    
    # Get the LLM instance
    llm = state.agent.llm
    
    # Store the original completion method
    original_completion = llm.completion
    
    # Create a wrapper function that ensures no system messages are used
    def completion_without_system_messages(*args, **kwargs):
        """
        A wrapper for the completion method that ensures no system messages are used.
        
        Args:
            *args: The positional arguments to pass to the original completion method
            **kwargs: The keyword arguments to pass to the original completion method
            
        Returns:
            The result of the original completion method
        """
        # Check if messages are in the kwargs
        if 'messages' in kwargs:
            messages = kwargs['messages']
            
            # Create a new messages list without system messages
            new_messages = []
            
            # Find the system message and user message
            system_content = None
            
            # First pass: extract system content
            for msg in messages:
                if isinstance(msg, dict) and msg.get('role') == 'system':
                    if system_content is None:
                        system_content = msg.get('content', '')
                    else:
                        system_content += '\n\n' + msg.get('content', '')
            
            # Second pass: create new messages
            for msg in messages:
                if isinstance(msg, dict):
                    if msg.get('role') == 'system':
                        # Skip system messages
                        continue
                    elif msg.get('role') == 'user' and system_content:
                        # Add system content to the first user message
                        new_msg = msg.copy()
                        new_msg['content'] = system_content + '\n\n' + new_msg.get('content', '')
                        new_messages.append(new_msg)
                        # Clear system content so it's only added to the first user message
                        system_content = None
                    else:
                        # Keep other messages as is
                        new_messages.append(msg)
                else:
                    # Keep non-dict messages as is
                    new_messages.append(msg)
            
            # If we have system content but no user messages, create a user message
            if system_content and not any(isinstance(msg, dict) and msg.get('role') == 'user' for msg in new_messages):
                new_messages.append({'role': 'user', 'content': system_content})
            
            # Replace the messages in the kwargs
            kwargs['messages'] = new_messages
            
            logger.info('Removed system messages from LLM completion')
        
        # Call the original completion method
        return original_completion(*args, **kwargs)
    
    # Replace the completion method with our patched version
    llm.completion = completion_without_system_messages
    
    # Also patch the _completion method if it exists
    if hasattr(llm, '_completion'):
        original_inner_completion = llm._completion
        
        # Create a wrapper function for the inner completion method
        def inner_completion_without_system_messages(*args, **kwargs):
            """
            A wrapper for the inner completion method that ensures no system messages are used.
            
            Args:
                *args: The positional arguments to pass to the original inner completion method
                **kwargs: The keyword arguments to pass to the original inner completion method
                
            Returns:
                The result of the original inner completion method
            """
            # Check if messages are in the kwargs
            if 'messages' in kwargs:
                messages = kwargs['messages']
                
                # Create a new messages list without system messages
                new_messages = []
                
                # Find the system message and user message
                system_content = None
                
                # First pass: extract system content
                for msg in messages:
                    if isinstance(msg, dict) and msg.get('role') == 'system':
                        if system_content is None:
                            system_content = msg.get('content', '')
                        else:
                            system_content += '\n\n' + msg.get('content', '')
                
                # Second pass: create new messages
                for msg in messages:
                    if isinstance(msg, dict):
                        if msg.get('role') == 'system':
                            # Skip system messages
                            continue
                        elif msg.get('role') == 'user' and system_content:
                            # Add system content to the first user message
                            new_msg = msg.copy()
                            new_msg['content'] = system_content + '\n\n' + new_msg.get('content', '')
                            new_messages.append(new_msg)
                            # Clear system content so it's only added to the first user message
                            system_content = None
                        else:
                            # Keep other messages as is
                            new_messages.append(msg)
                    else:
                        # Keep non-dict messages as is
                        new_messages.append(msg)
                
                # If we have system content but no user messages, create a user message
                if system_content and not any(isinstance(msg, dict) and msg.get('role') == 'user' for msg in new_messages):
                    new_messages.append({'role': 'user', 'content': system_content})
                
                # Replace the messages in the kwargs
                kwargs['messages'] = new_messages
                
                logger.info('Removed system messages from inner LLM completion')
            
            # Call the original inner completion method
            return original_inner_completion(*args, **kwargs)
        
        # Replace the inner completion method with our patched version
        llm._completion = inner_completion_without_system_messages
    
    logger.info('LLM patched to never use system messages')