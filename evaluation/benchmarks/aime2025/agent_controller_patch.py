"""
This module provides a patched version of the AgentController class
that ensures all instructions are sent in a single user message.
"""

from openhands.controller.agent_controller import AgentController
from openhands.core.logger import openhands_logger as logger
from openhands.events import EventSource
from openhands.events.action import MessageAction
from openhands.events.event import Event


# Patch the AgentController class to ensure single message
def patch_agent_controller():
    """
    Patch the AgentController class to ensure all instructions are sent in a single user message.
    """
    # Store the original _on_event method
    original_on_event = AgentController._on_event

    # Define a new _on_event method that ensures single message
    async def patched_on_event(self, event: Event):
        """
        Process an event, ensuring all instructions are sent in a single user message.
        """
        # If this is a MessageAction from the user, ensure it's the only one
        if isinstance(event, MessageAction) and event.source == EventSource.USER:
            # Get the initial user message (the problem statement)
            initial_user_message = None
            for e in self.state.history:
                if isinstance(e, MessageAction) and e.source == EventSource.USER:
                    initial_user_message = e
                    break
            
            # If we have an initial user message and this is a different message
            if initial_user_message and initial_user_message.content != event.content:
                # This is a follow-up message (e.g., a reminder to use Python)
                logger.info(f'Intercepted follow-up user message: "{event.content[:50]}..."')
                logger.info('Ignoring follow-up user message to maintain single message approach')
                return
            
            # Clear any existing user messages from the history
            self.state.history = [
                e
                for e in self.state.history
                if not (isinstance(e, MessageAction) and e.source == EventSource.USER)
            ]
            
            # Log that we're using a single user message
            logger.info('Using single user message approach for AIME2025 benchmark')
        
        # Call the original _on_event method
        return await original_on_event(self, event)
    
    # Replace the original _on_event method with our patched version
    AgentController._on_event = patched_on_event
    
    logger.info('AgentController patched to ensure single user message')