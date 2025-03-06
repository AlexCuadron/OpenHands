"""
This module provides functionality to ensure all assistant messages are combined into a single growing message.
"""

from typing import Any, Dict, List, Optional

from openhands.controller.state.state import State
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import MessageAction
from openhands.events.event import Event
from openhands.events.observation import Observation


def patch_llm_for_single_assistant_message(state: State) -> None:
    """
    Patch the LLM to ensure all assistant messages are combined into a single growing message.
    
    Args:
        state: The current state of the agent
    """
    if not hasattr(state, 'agent') or not hasattr(state.agent, 'llm'):
        logger.warning('Cannot patch LLM for single assistant message: agent or llm not found')
        return
    
    logger.info('Patching LLM for single assistant message')
    
    # Store the original completion method
    original_completion = state.agent.llm.completion
    
    # Create a wrapper function that combines all assistant messages
    def completion_with_single_assistant_message(*args, **kwargs):
        # Get the messages from kwargs
        if 'messages' in kwargs:
            kwargs_messages = kwargs['messages']
            
            # Find the first user message in the history
            first_user_message = None
            for event in state.history:
                if (
                    hasattr(event, 'role')
                    and event.role == 'user'
                    and hasattr(event, 'message')
                ):
                    first_user_message = event.message
                    break
            
            # Collect all assistant messages and tool observations
            assistant_content = []
            for i, event in enumerate(state.history):
                # Add assistant messages
                if (
                    hasattr(event, 'role')
                    and event.role == 'assistant'
                    and hasattr(event, 'message')
                ):
                    assistant_content.append(event.message)
                
                # Add tool observations (results)
                elif isinstance(event, Observation):
                    # Only add observations that follow assistant messages
                    if i > 0 and hasattr(state.history[i-1], 'role') and state.history[i-1].role == 'assistant':
                        # Format the observation as a tool result
                        observation_content = f"\n\nTOOL RESULT:\n{event.content}\n"
                        assistant_content.append(observation_content)
            
            # Combine all assistant content into a single message
            combined_assistant_message = "\n".join(assistant_content)
            
            # If we have the first user message and assistant content
            if first_user_message is not None:
                # Create a new messages list with only the first user message
                new_messages = [{'role': 'user', 'content': first_user_message}]
                
                # Add the combined assistant message as a prefix if it's not empty
                if combined_assistant_message:
                    new_messages.append({
                        'role': 'assistant',
                        'content': combined_assistant_message,
                        'prefix': True
                    })
                
                # Replace the original messages with our new messages
                kwargs['messages'] = new_messages
                
                # Log the transformation
                logger.info(f'Transformed messages: First user message + Single combined assistant message')
        
        # Call the original completion method
        return original_completion(*args, **kwargs)
    
    # Replace the agent's completion method with our wrapper
    state.agent.llm.completion = completion_with_single_assistant_message
    
    logger.info('Single assistant message functionality enabled')


def get_combined_assistant_message(state: State) -> str:
    """
    Get the combined assistant message from the state history.
    
    Args:
        state: The current state of the agent
        
    Returns:
        The combined assistant message
    """
    # Collect all assistant messages and tool observations
    assistant_content = []
    for i, event in enumerate(state.history):
        # Add assistant messages
        if (
            hasattr(event, 'role')
            and event.role == 'assistant'
            and hasattr(event, 'message')
        ):
            assistant_content.append(event.message)
        
        # Add tool observations (results)
        elif isinstance(event, Observation):
            # Only add observations that follow assistant messages
            if i > 0 and hasattr(state.history[i-1], 'role') and state.history[i-1].role == 'assistant':
                # Format the observation as a tool result
                observation_content = f"\n\nTOOL RESULT:\n{event.content}\n"
                assistant_content.append(observation_content)
    
    # Combine all assistant content into a single message
    return "\n".join(assistant_content)