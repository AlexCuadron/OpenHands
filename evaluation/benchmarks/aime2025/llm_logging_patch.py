"""
This module provides functionality to patch the LLM logging mechanism to ensure
that only user and assistant roles are used in the logged messages.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

from openhands.controller.state.state import State
from openhands.core.logger import openhands_logger as logger
from openhands.events import EventSource
from openhands.events.action import MessageAction
from openhands.events.event import Event
from openhands.events.observation import Observation
from openhands.llm.llm import LLM


def patch_llm_logging(state: State) -> None:
    """
    Patch the LLM logging mechanism to ensure that only user and assistant roles are used in the logged messages.
    
    Args:
        state: The current state of the agent
    """
    if not hasattr(state, 'agent') or not hasattr(state.agent, 'llm'):
        logger.warning('Cannot patch LLM logging: agent or llm not found')
        return
    
    logger.info('Patching LLM logging')
    
    # Get the LLM instance
    llm = state.agent.llm
    
    # Store the original _post_completion method
    original_post_completion = llm._post_completion
    
    # Create a wrapper function that modifies the logged messages
    def patched_post_completion(resp: Any) -> float:
        """
        A wrapper for the _post_completion method that modifies the logged messages.
        
        Args:
            resp: The response from the LLM
            
        Returns:
            The cost of the completion
        """
        # Call the original method to get the cost
        cost = original_post_completion(resp)
        
        # If log_completions is enabled, modify the logged messages
        if llm.config.log_completions:
            # Get the most recent log file
            log_folder = llm.config.log_completions_folder
            if log_folder and os.path.exists(log_folder):
                log_files = [os.path.join(log_folder, f) for f in os.listdir(log_folder) if f.endswith('.json')]
                if log_files:
                    # Sort by modification time (most recent first)
                    log_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                    most_recent_log = log_files[0]
                    
                    # Read the log file
                    try:
                        with open(most_recent_log, 'r') as f:
                            log_data = json.load(f)
                        
                        # Modify the messages to use only user and assistant roles
                        if 'messages' in log_data:
                            messages = log_data['messages']
                            
                            # Create a new messages list with only user and assistant roles
                            new_messages = []
                            
                            # Find the system message and user message
                            system_message = None
                            user_message = None
                            
                            for msg in messages:
                                if msg.get('role') == 'system':
                                    system_message = msg.get('content', '')
                                elif msg.get('role') == 'user':
                                    user_message = msg.get('content', '')
                            
                            # If we have both system and user messages, combine them
                            if system_message and user_message:
                                # Create a new user message with the system message included
                                new_user_message = {
                                    'role': 'user',
                                    'content': f"{system_message}\n\n{user_message}"
                                }
                                new_messages.append(new_user_message)
                            elif user_message:
                                # Just use the user message
                                new_messages.append({'role': 'user', 'content': user_message})
                            
                            # Add an empty assistant message
                            new_messages.append({'role': 'assistant', 'content': ''})
                            
                            # Replace the messages in the log data
                            log_data['messages'] = new_messages
                            
                            # Write the modified log data back to the file
                            with open(most_recent_log, 'w') as f:
                                json.dump(log_data, f, indent=2)
                            
                            logger.info(f'Modified logged messages in {most_recent_log}')
                    except Exception as e:
                        logger.error(f'Error modifying logged messages: {e}')
        
        return cost
    
    # Replace the _post_completion method with our patched version
    llm._post_completion = patched_post_completion
    
    logger.info('LLM logging patched to use only user and assistant roles')