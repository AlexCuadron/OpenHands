"""
Modified LLM module for AIME2025 benchmark.
This ensures:
1. No system messages
2. First message is always a user message
3. All subsequent messages are combined into a single assistant message
"""

import copy
import functools
import os
import time
from typing import Any, Dict, List, Optional

from openhands.core.logger import openhands_logger as logger
from openhands.llm.llm import LLM


def patch_llm_for_aime2025(llm: LLM) -> None:
    """
    Patch the LLM for AIME2025 benchmark.
    
    Args:
        llm: The LLM instance to patch
    """
    logger.info('Patching LLM for AIME2025 benchmark')
    
    # Store the original completion method
    original_completion = llm.completion
    
    # Create a wrapper function that implements the AIME2025 requirements
    @functools.wraps(original_completion)
    def aime2025_completion(*args, **kwargs):
        """
        A wrapper for the completion method that implements the AIME2025 requirements.
        
        Args:
            *args: The positional arguments to pass to the original completion method
            **kwargs: The keyword arguments to pass to the original completion method
            
        Returns:
            The result of the original completion method
        """
        # Process messages to implement AIME2025 requirements
        if 'messages' in kwargs:
            messages = kwargs['messages']
            
            # Ensure we work with a list of messages
            messages = messages if isinstance(messages, list) else [messages]
            
            # Create new messages list
            new_messages = []
            
            # Collect all content by role
            system_content = []
            user_content = []
            assistant_content = []
            
            # First pass: collect content by role
            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get('role')
                    content = msg.get('content', '')
                    
                    if role == 'system':
                        system_content.append(content)
                    elif role == 'user':
                        user_content.append(content)
                    elif role == 'assistant':
                        # Check if this is a prefix message
                        if msg.get('prefix') is True:
                            assistant_content.append(content)
            
            # Create a single user message with system content
            if user_content:
                # Combine all user content
                combined_user_content = '\n\n'.join(user_content)
                
                # Add system content if any
                if system_content:
                    combined_user_content = '\n\n'.join(system_content) + '\n\n' + combined_user_content
                
                # Add the user message
                new_messages.append({'role': 'user', 'content': combined_user_content})
            elif system_content:
                # If there's no user content but there is system content, create a user message with system content
                combined_system_content = '\n\n'.join(system_content)
                new_messages.append({'role': 'user', 'content': combined_system_content})
            
            # Create a single assistant message with all assistant content
            if assistant_content:
                combined_assistant_content = '\n\n'.join(assistant_content)
                new_messages.append({'role': 'assistant', 'content': combined_assistant_content})
            
            # Replace the messages in the kwargs
            kwargs['messages'] = new_messages
            
            logger.info(f'AIME2025: Transformed messages to {len(new_messages)} messages')
            for i, msg in enumerate(new_messages):
                logger.info(f'AIME2025: Message {i+1} - Role: {msg.get("role")}, Content length: {len(msg.get("content", ""))}')
        
        # Call the original completion method
        return original_completion(*args, **kwargs)
    
    # Replace the completion method with our wrapper
    llm.completion = aime2025_completion
    
    # Also patch the _completion method if it exists
    if hasattr(llm, '_completion'):
        original_inner_completion = llm._completion
        
        @functools.wraps(original_inner_completion)
        def aime2025_inner_completion(*args, **kwargs):
            """
            A wrapper for the inner completion method that implements the AIME2025 requirements.
            
            Args:
                *args: The positional arguments to pass to the original inner completion method
                **kwargs: The keyword arguments to pass to the original inner completion method
                
            Returns:
                The result of the original inner completion method
            """
            # Process messages to implement AIME2025 requirements
            if 'messages' in kwargs:
                messages = kwargs['messages']
                
                # Ensure we work with a list of messages
                messages = messages if isinstance(messages, list) else [messages]
                
                # Create new messages list
                new_messages = []
                
                # Collect all content by role
                system_content = []
                user_content = []
                assistant_content = []
                
                # First pass: collect content by role
                for msg in messages:
                    if isinstance(msg, dict):
                        role = msg.get('role')
                        content = msg.get('content', '')
                        
                        if role == 'system':
                            system_content.append(content)
                        elif role == 'user':
                            user_content.append(content)
                        elif role == 'assistant':
                            # Check if this is a prefix message
                            if msg.get('prefix') is True:
                                assistant_content.append(content)
                
                # Create a single user message with system content
                if user_content:
                    # Combine all user content
                    combined_user_content = '\n\n'.join(user_content)
                    
                    # Add system content if any
                    if system_content:
                        combined_user_content = '\n\n'.join(system_content) + '\n\n' + combined_user_content
                    
                    # Add the user message
                    new_messages.append({'role': 'user', 'content': combined_user_content})
                elif system_content:
                    # If there's no user content but there is system content, create a user message with system content
                    combined_system_content = '\n\n'.join(system_content)
                    new_messages.append({'role': 'user', 'content': combined_system_content})
                
                # Create a single assistant message with all assistant content
                if assistant_content:
                    combined_assistant_content = '\n\n'.join(assistant_content)
                    new_messages.append({'role': 'assistant', 'content': combined_assistant_content})
                
                # Replace the messages in the kwargs
                kwargs['messages'] = new_messages
                
                logger.info(f'AIME2025 (inner): Transformed messages to {len(new_messages)} messages')
            
            # Call the original inner completion method
            return original_inner_completion(*args, **kwargs)
        
        # Replace the inner completion method with our wrapper
        llm._completion = aime2025_inner_completion
    
    # Also patch the _completion_unwrapped method if it exists
    if hasattr(llm, '_completion_unwrapped'):
        original_unwrapped = llm._completion_unwrapped
        
        @functools.wraps(original_unwrapped)
        def aime2025_unwrapped(*args, **kwargs):
            """
            A wrapper for the unwrapped completion method that implements the AIME2025 requirements.
            
            Args:
                *args: The positional arguments to pass to the original unwrapped method
                **kwargs: The keyword arguments to pass to the original unwrapped method
                
            Returns:
                The result of the original unwrapped method
            """
            # Process messages to implement AIME2025 requirements
            if 'messages' in kwargs:
                messages = kwargs['messages']
                
                # Ensure we work with a list of messages
                messages = messages if isinstance(messages, list) else [messages]
                
                # Create new messages list
                new_messages = []
                
                # Collect all content by role
                system_content = []
                user_content = []
                assistant_content = []
                
                # First pass: collect content by role
                for msg in messages:
                    if isinstance(msg, dict):
                        role = msg.get('role')
                        content = msg.get('content', '')
                        
                        if role == 'system':
                            system_content.append(content)
                        elif role == 'user':
                            user_content.append(content)
                        elif role == 'assistant':
                            # Check if this is a prefix message
                            if msg.get('prefix') is True:
                                assistant_content.append(content)
                
                # Create a single user message with system content
                if user_content:
                    # Combine all user content
                    combined_user_content = '\n\n'.join(user_content)
                    
                    # Add system content if any
                    if system_content:
                        combined_user_content = '\n\n'.join(system_content) + '\n\n' + combined_user_content
                    
                    # Add the user message
                    new_messages.append({'role': 'user', 'content': combined_user_content})
                elif system_content:
                    # If there's no user content but there is system content, create a user message with system content
                    combined_system_content = '\n\n'.join(system_content)
                    new_messages.append({'role': 'user', 'content': combined_system_content})
                
                # Create a single assistant message with all assistant content
                if assistant_content:
                    combined_assistant_content = '\n\n'.join(assistant_content)
                    new_messages.append({'role': 'assistant', 'content': combined_assistant_content})
                
                # Replace the messages in the kwargs
                kwargs['messages'] = new_messages
                
                logger.info(f'AIME2025 (unwrapped): Transformed messages to {len(new_messages)} messages')
            
            # Call the original unwrapped method
            return original_unwrapped(*args, **kwargs)
        
        # Replace the unwrapped method with our wrapper
        llm._completion_unwrapped = aime2025_unwrapped
    
    logger.info('LLM patched for AIME2025 benchmark')