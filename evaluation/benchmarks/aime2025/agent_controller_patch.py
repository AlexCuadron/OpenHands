"""
This module provides a patched version of the AgentController class
that ensures all instructions are sent in a single user message.
"""

from openhands.controller.agent_controller import AgentController
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import MessageAction
from openhands.events.event import Event


# Patch the AgentController class to ensure single message
def patch_agent_controller():
    """
    Patch the AgentController class to ensure all instructions are sent in a single user message.
    """
    # Store the original process_event method
    original_process_event = AgentController.process_event

    # Define a new process_event method that ensures single message
    async def patched_process_event(self, event: Event):
        """
        Process an event, ensuring all instructions are sent in a single user message.
        """
        # If this is a MessageAction from the user, ensure it's the only one
        if isinstance(event, MessageAction) and event.role == 'user':
            # Clear any existing user messages from the history
            self.state.history = [
                e
                for e in self.state.history
                if not (isinstance(e, MessageAction) and e.role == 'user')
            ]

            # Log that we're using a single user message
            logger.info('Using single user message approach for AIME2025 benchmark')

        # Call the original process_event method
        return await original_process_event(self, event)

    # Replace the original process_event method with our patched version
    AgentController.process_event = patched_process_event
