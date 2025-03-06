"""
This module provides a modified version of the run_controller function
that ensures all instructions are sent in a single user message.
"""

from typing import Callable, Optional

from openhands.controller.agent_controller import AgentController
from openhands.controller.state.state import State
from openhands.core.config import AppConfig
from openhands.events.action import MessageAction
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
        fake_user_response_fn=fake_user_response_fn,
    )

    # Initialize the controller
    await controller.initialize()

    # Process the initial user action
    await controller.process_event(initial_user_action)

    # Run the agent loop
    await controller.run_agent_loop()

    # Return the final state
    return controller.state
