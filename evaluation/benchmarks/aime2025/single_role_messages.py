"""
This module provides functionality to ensure there is only one user message and one assistant message.
"""

from typing import Any, Dict, List, Optional

from openhands.controller.state.state import State
from openhands.core.logger import openhands_logger as logger
from openhands.events import EventSource
from openhands.events.action import MessageAction
from openhands.events.event import Event
from openhands.events.observation import Observation


def patch_llm_for_single_role_messages(state: State) -> None:
    """
    Patch the LLM to ensure there is only one user message and one assistant message.
    
    Args:
        state: The current state of the agent
    """
    if not hasattr(state, 'agent') or not hasattr(state.agent, 'llm'):
        logger.warning('Cannot patch LLM for single role messages: agent or llm not found')
        return
    
    logger.info('Patching LLM for single role messages')
    
    # Store the original completion method
    original_completion = state.agent.llm.completion
    
    # Create a wrapper function that ensures single role messages
    def completion_with_single_role_messages(*args, **kwargs):
        # Get the messages from kwargs
        if 'messages' in kwargs:
            # Find the first user message in the history
            first_user_message = None
            for event in state.history:
                if isinstance(event, MessageAction) and event.source == EventSource.USER:
                    first_user_message = event.content
                    break
            
            # Collect all assistant messages and tool observations
            assistant_content = []
            for i, event in enumerate(state.history):
                # Add assistant messages
                if isinstance(event, MessageAction) and event.source == EventSource.AGENT:
                    assistant_content.append(event.content)
                
                # Add tool observations (results)
                elif isinstance(event, Observation):
                    # Only add observations that follow assistant messages
                    if i > 0 and isinstance(state.history[i-1], MessageAction) and state.history[i-1].source == EventSource.AGENT:
                        # Format the observation as a tool result
                        observation_content = f"\n\nTOOL RESULT:\n{event.content}\n"
                        assistant_content.append(observation_content)
            
            # Combine all assistant content into a single message
            combined_assistant_message = "\n".join(assistant_content)
            
            # If we have the first user message
            if first_user_message is not None:
                # Create a new messages list with only user and assistant messages
                new_messages = []
                
                # Add system instructions to the user message
                system_instructions = 'You are OpenHands agent, a helpful AI assistant that can interact with a computer to solve tasks.'
                user_content = f"{system_instructions}\n\n{first_user_message}"
                
                # Create the user message
                user_message = {
                    'role': 'user',
                    'content': user_content
                }
                new_messages.append(user_message)
                
                # Add an empty assistant message with prefix=true
                assistant_message = {
                    'role': 'assistant',
                    'content': combined_assistant_message if combined_assistant_message else "",
                    'prefix': True
                }
                new_messages.append(assistant_message)
                
                # Replace the original messages with our new messages
                kwargs['messages'] = new_messages
                
                # Log the transformation
                logger.info(f'Transformed messages: Single user message + Single assistant message with prefix=true')
        
        # Call the original completion method
        return original_completion(*args, **kwargs)
    
    # Replace the agent's completion method with our wrapper
    state.agent.llm.completion = completion_with_single_role_messages
    
    logger.info('Single role messages functionality enabled')