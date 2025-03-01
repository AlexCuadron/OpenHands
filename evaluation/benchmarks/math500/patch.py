"""Patch functions for the MATH-500 benchmark."""

from evaluation.benchmarks.math500.function_calling import (
    get_tools,
    response_to_actions,
)
from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent


def patch_codeact_agent():
    """Patch the CodeActAgent to use our custom finish tool.

    This function modifies the CodeActAgent class to use our custom finish tool
    without creating a new agent class.
    """
    # Save the original __init__ method
    original_init = CodeActAgent.__init__

    # Define a new __init__ method that uses our custom tools
    def patched_init(self, *args, **kwargs):
        # Call the original __init__ method
        original_init(self, *args, **kwargs)

        # Override the tools with our custom tools
        self.tools = get_tools(
            codeact_enable_browsing=self.config.codeact_enable_browsing,
            codeact_enable_jupyter=self.config.codeact_enable_jupyter,
            codeact_enable_llm_editor=self.config.codeact_enable_llm_editor,
        )

    # We're replacing the step method completely, no need to save the original

    # Define a new step method that uses our custom response_to_actions function
    def patched_step(self, state):
        # Continue with pending actions if any
        if self.pending_actions:
            return self.pending_actions.popleft()

        # if we're done, go back
        latest_user_message = state.get_last_user_message()
        if latest_user_message and latest_user_message.content.strip() == '/exit':
            from openhands.events.action import AgentFinishAction

            return AgentFinishAction()

        # prepare what we want to send to the LLM
        messages = self._get_messages(state)
        params: dict = {
            'messages': self.llm.format_messages_for_llm(messages),
        }
        params['tools'] = self.tools
        response = self.llm.completion(**params)

        # Use our custom response_to_actions function
        actions = response_to_actions(response)
        for action in actions:
            self.pending_actions.append(action)
            
        # If no actions were added to the pending_actions, create a default message action
        if not self.pending_actions:
            from openhands.events.action import MessageAction
            content = "I'm thinking about how to solve this problem. Let me try using the Python interpreter."
            return MessageAction(content=content)
            
        return self.pending_actions.popleft()

    # Replace the original methods with our patched versions
    CodeActAgent.__init__ = patched_init
    CodeActAgent.step = patched_step
