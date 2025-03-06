"""
This module provides a modified version of the run_controller function
that ensures all instructions are sent in a single user message.
"""

import asyncio
import threading
import time
from typing import Callable, Optional, Tuple

from openhands.controller.agent_controller import AgentController
from openhands.controller.state.state import State
from openhands.core.config import AppConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.loop import run_agent_until_done
from openhands.core.schema import AgentState
from openhands.core.setup import create_agent, create_controller
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
    # Create the agent
    agent = create_agent(config)
    
    # Create the controller using the create_controller function
    controller, initial_state = create_controller(
        agent=agent,
        runtime=runtime,
        config=config,
        headless_mode=True,
    )
    
    # Set the fake user response function
    if fake_user_response_fn:
        runtime.event_stream.fake_user_response_fn = single_message_fake_user_response
    
    # Add the initial user action to the event stream directly
    runtime.event_stream.add_event(initial_user_action, EventSource.USER)
    
    # Define the end states
    end_states = [
        AgentState.FINISHED,
        AgentState.REJECTED,
        AgentState.ERROR,
        AgentState.PAUSED,
        AgentState.STOPPED,
    ]
    
    # Run the agent until it reaches an end state
    # We'll use a timeout to prevent it from running indefinitely
    timeout = 300  # 5 minutes
    start_time = time.time()
    
    while controller.state.agent_state not in end_states:
        # Check if we've exceeded the timeout
        if time.time() - start_time > timeout:
            logger.warning(f'Timeout exceeded ({timeout} seconds). Exiting.')
            break
        
        # Check if the agent is waiting for user input
        if controller.state.agent_state == AgentState.AWAITING_USER_INPUT:
            logger.info('Agent is waiting for user input, but we are in single message mode. Exiting.')
            # Instead of trying to set the agent state directly, we'll just break out of the loop
            break
        
        # Sleep for a short time to avoid busy waiting
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