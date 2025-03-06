"""
This module provides functionality to ensure all messages are combined into a single system and user message.
"""

from typing import Any, Dict, List, Optional

from openhands.controller.state.state import State
from openhands.core.logger import openhands_logger as logger
from openhands.events import EventSource
from openhands.events.action import MessageAction
from openhands.events.event import Event
from openhands.events.observation import Observation


def patch_llm_for_single_message_roles(state: State) -> None:
    """
    Patch the LLM to ensure all messages are combined into a single system and user message.
    
    Args:
        state: The current state of the agent
    """
    if not hasattr(state, 'agent') or not hasattr(state.agent, 'llm'):
        logger.warning('Cannot patch LLM for single message roles: agent or llm not found')
        return
    
    logger.info('Patching LLM for single message roles')
    
    # Store the original completion method
    original_completion = state.agent.llm.completion
    
    # Create a wrapper function that combines all messages
    def completion_with_single_message_roles(*args, **kwargs):
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
                # Create a new messages list with only system and user messages
                new_messages = []
                
                # Add a system message with instructions
                system_message = {
                    'role': 'system',
                    'content': 'You are OpenHands agent, a helpful AI assistant that can interact with a computer to solve tasks.'
                }
                new_messages.append(system_message)
                
                # Create a user message that includes both the original user message and the assistant's responses
                user_content = first_user_message
                if combined_assistant_message:
                    user_content += f"\n\nPrevious assistant responses:\n{combined_assistant_message}"
                
                user_message = {
                    'role': 'user',
                    'content': user_content
                }
                new_messages.append(user_message)
                
                # Replace the original messages with our new messages
                kwargs['messages'] = new_messages
                
                # Log the transformation
                logger.info(f'Transformed messages: Single system message + Single user message (including assistant responses)')
        
        # Call the original completion method
        return original_completion(*args, **kwargs)
    
    # Replace the agent's completion method with our wrapper
    state.agent.llm.completion = completion_with_single_message_roles
    
    logger.info('Single message roles functionality enabled')