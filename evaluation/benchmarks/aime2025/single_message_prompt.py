"""
This module provides a modified version of the run_controller function
that ensures all instructions are sent in a single user message.
"""

import asyncio
from typing import Callable, Optional

from openhands.controller.agent_controller import AgentController
from openhands.controller.state.state import State
from openhands.core.config import AppConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.schema import AgentState
from openhands.events import EventSource, EventStreamSubscriber
from openhands.events.action import MessageAction
from openhands.events.event import Event
from openhands.events.observation import AgentStateChangedObservation
from openhands.runtime.base import Runtime


async def run_controller_single_message(
    config: AppConfig,
    initial_user_action: MessageAction,
    runtime: Runtime,
    fake_user_response_fn: Optional[Callable[[State], str]] = None,
) -> State:
    """
    Run the agent controller with a single user message.
    
    This function is a modified version of run_controller that ensures
    all instructions are sent in a single user message.
    
    Args:
        config: The application configuration
        initial_user_action: The initial user action
        runtime: The runtime to use
        fake_user_response_fn: Optional function to generate fake user responses
        
    Returns:
        The final state of the agent controller
    """
    # Create the agent controller
    controller = AgentController(
        config=config,
        runtime=runtime,
        # We'll override the fake_user_response_fn to prevent additional user messages
        fake_user_response_fn=single_message_fake_user_response,
    )
    
    # Initialize the controller
    await controller.initialize()
    
    # Process the initial user action
    await controller.process_event(initial_user_action)
    
    # Set up event handling to prevent additional user messages
    event_stream = runtime.event_stream
    
    def on_event(event: Event):
        if isinstance(event, AgentStateChangedObservation):
            if event.agent_state == AgentState.AWAITING_USER_INPUT:
                # If the agent is waiting for user input, we'll exit instead
                logger.info('Agent is waiting for user input, but we are in single message mode. Exiting.')
                asyncio.create_task(controller.set_agent_state_to(AgentState.FINISHED))
    
    # Subscribe to events
    event_stream.subscribe(EventStreamSubscriber.MAIN, on_event, event_stream.sid)
    
    # Define the end states
    end_states = [
        AgentState.FINISHED,
        AgentState.REJECTED,
        AgentState.ERROR,
        AgentState.PAUSED,
        AgentState.STOPPED,
    ]
    
    # Run the agent until it reaches an end state
    while controller.state.agent_state not in end_states:
        await asyncio.sleep(1)
    
    # Return the final state
    return controller.state


def single_message_fake_user_response(state: State) -> str:
    """
    A fake user response function that always returns '/exit' to prevent additional user messages.
    
    Args:
        state: The current state of the agent
        
    Returns:
        The string '/exit' to indicate that the agent should exit
    """
    logger.info('Preventing additional user message in single message mode')
    return '/exit'