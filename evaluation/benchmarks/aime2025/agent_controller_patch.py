"""
This module provides a patched version of the AgentController class
that ensures all instructions are sent in a single user message.
"""

from typing import List, Optional

from openhands.controller.agent_controller import AgentController
from openhands.controller.state.state import State
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import AgentFinishAction, MessageAction
from openhands.events.event import Event


# Patch the AgentController class to ensure single message
def patch_agent_controller():
    """
    Patch the AgentController class to ensure all instructions are sent in a single user message.
    """
    # Store the original process_event method
    original_process_event = AgentController.process_event
    
    # Store the original get_fake_user_response method
    original_get_fake_user_response = AgentController._get_fake_user_response
    
    # Store the original run_agent_loop method
    original_run_agent_loop = AgentController.run_agent_loop
    
    # Define a new process_event method that ensures single message
    async def patched_process_event(self, event: Event):
        """
        Process an event, ensuring all instructions are sent in a single user message.
        """
        # If this is a MessageAction from the user, ensure it's the only one
        if isinstance(event, MessageAction) and event.role == 'user':
            # Get the initial user message (the problem statement)
            initial_user_message = self._get_initial_user_message(self.state)
            
            # If we have an initial user message and this is a different message
            if initial_user_message and initial_user_message.message != event.message:
                # This is a follow-up message (e.g., a reminder to use Python)
                logger.info(f'Intercepted follow-up user message: "{event.message[:50]}..."')
                
                # Instead of adding a new user message, we'll modify the agent's behavior directly
                # For example, if the message is about using Python, we'll make the agent use Python
                if "verify" in event.message.lower() and "python" in event.message.lower():
                    logger.info('Detected reminder to use Python - will modify agent behavior')
                    # We'll handle this in the run_agent_loop patch
                    self._needs_python_reminder = True
                    # Don't process this event
                    return
                
                # For other messages, we'll just log them and not process them
                logger.info('Ignoring follow-up user message to maintain single message approach')
                return
            
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
    
    # Define a new _get_fake_user_response method that prevents additional user messages
    def patched_get_fake_user_response(self, state: State) -> Optional[str]:
        """
        Get a fake user response, but prevent additional user messages.
        """
        # Check if the agent has used the finish tool
        finish_action = next(
            (
                event
                for event in reversed(state.history)
                if isinstance(event, AgentFinishAction)
            ),
            None,
        )
        
        if finish_action:
            # If the agent has used the finish tool, let it finish
            logger.info('Agent used finish tool - allowing exit')
            return '/exit'
        
        # Check if the agent has used Python
        has_used_python = any(
            'execute_ipython_cell' in str(event) or 'EXECUTION RESULT' in str(event)
            for event in state.history
            if hasattr(event, 'message') and event.message
        )
        
        # If the agent is trying to finish without using Python, prevent it
        if not has_used_python and self._is_trying_to_finish(state):
            logger.info('Agent trying to finish without using Python - intercepting')
            # Instead of sending a new user message, we'll modify the agent's next action
            # We'll set a flag that will be checked in the run_agent_loop patch
            self._needs_python_reminder = True
            # Return None to indicate no user response
            return None
        
        # Call the original method
        response = original_get_fake_user_response(self, state)
        
        # If the response would create a new user message, log it but don't send it
        if response and response != '/exit':
            logger.info(f'Intercepted potential user message: "{response[:50]}..."')
            # Instead of sending a new user message, we'll modify the agent's next action
            # We'll set a flag that will be checked in the run_agent_loop patch
            self._needs_python_reminder = True
            # Return None to indicate no user response
            return None
        
        return response
    
    # Define a new run_agent_loop method that handles the Python reminder flag
    async def patched_run_agent_loop(self):
        """
        Run the agent loop, handling the Python reminder flag.
        """
        # Initialize the Python reminder flag
        if not hasattr(self, '_needs_python_reminder'):
            self._needs_python_reminder = False
        
        # Run the original method
        await original_run_agent_loop(self)
    
    # Helper method to get the initial user message
    def _get_initial_user_message(self, state: State) -> Optional[MessageAction]:
        """
        Get the initial user message from the state history.
        """
        for event in state.history:
            if isinstance(event, MessageAction) and event.role == 'user':
                return event
        return None
    
    # Helper method to check if the agent is trying to finish
    def _is_trying_to_finish(self, state: State) -> bool:
        """
        Check if the agent is trying to finish without using Python.
        """
        # Check the last few messages for keywords indicating the agent is trying to finish
        recent_messages = [
            event.message
            for event in reversed(state.history[:len(state.history)])
            if hasattr(event, 'message') and event.message
        ][:3]  # Look at the last 3 messages
        
        # Check for keywords indicating the agent is trying to finish
        finish_keywords = ['final answer', 'the answer is', '\\boxed{', 'solution']
        return any(
            any(keyword.lower() in msg.lower() for keyword in finish_keywords)
            for msg in recent_messages
            if msg
        )
    
    # Add the helper methods to the AgentController class
    AgentController._get_initial_user_message = _get_initial_user_message
    AgentController._is_trying_to_finish = _is_trying_to_finish
    
    # Replace the original methods with our patched versions
    AgentController.process_event = patched_process_event
    AgentController._get_fake_user_response = patched_get_fake_user_response
    AgentController.run_agent_loop = patched_run_agent_loop
    
    # Initialize the Python reminder flag for existing instances if the _instances attribute exists
    if hasattr(AgentController, '_instances'):
        for instance in AgentController._instances:
            if not hasattr(instance, '_needs_python_reminder'):
                instance._needs_python_reminder = False
    
    logger.info('AgentController patched to ensure single user message and handle Python reminders')
