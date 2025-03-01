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
        
        # Check if the model supports function calling
        model_name = self.llm.model.lower() if hasattr(self.llm, 'model') else ""
        if any(name in model_name for name in ['deepseek', 'together', 'llama', 'mistral']):
            # For models that don't support function calling, we'll use a text-only approach
            # Add a special instruction to the last message
            last_message = params['messages'][-1]
            if isinstance(last_message, dict) and last_message.get('role') == 'user':
                # Add instructions for using tools in text format
                tool_instructions = "\n\nTo use tools, respond with:\n```\nTOOL: tool_name\nARGUMENTS: {\"arg1\": \"value1\", \"arg2\": \"value2\"}\n```\n\nAvailable tools:\n"
                for tool in self.tools:
                    if tool['type'] == 'function':
                        tool_instructions += f"- {tool['function']['name']}: {tool['function']['description']}\n"
                
                last_message['content'] += tool_instructions
                
            # Don't include tools in the params
            response = self.llm.completion(**params)
            
            # Parse the response to extract tool calls
            if hasattr(response, 'choices') and len(response.choices) > 0:
                choice = response.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    content = choice.message.content
                    # Look for tool calls in the format: TOOL: tool_name\nARGUMENTS: {...}
                    import re
                    import json
                    tool_calls = []
                    tool_pattern = r'TOOL:\s*(\w+)\s*\nARGUMENTS:\s*({.*?})'
                    matches = re.findall(tool_pattern, content, re.DOTALL)
                    for tool_name, args_str in matches:
                        try:
                            args = json.loads(args_str)
                            tool_calls.append({
                                'id': f'call_{len(tool_calls)}',
                                'type': 'function',
                                'function': {
                                    'name': tool_name,
                                    'arguments': json.dumps(args)
                                }
                            })
                        except json.JSONDecodeError:
                            pass
                    
                    # Add tool_calls to the message
                    if tool_calls:
                        choice.message.tool_calls = tool_calls
        else:
            # For models that support function calling, use the normal approach
            params['tools'] = self.tools
            response = self.llm.completion(**params)

        # Use our custom response_to_actions function
        actions = response_to_actions(response)
        for action in actions:
            self.pending_actions.append(action)
            
        # If no actions were added to the pending_actions, create a default finish action
        if not self.pending_actions:
            from openhands.events.action import AgentFinishAction
            return AgentFinishAction()
            
        return self.pending_actions.popleft()

    # Replace the original methods with our patched versions
    CodeActAgent.__init__ = patched_init
    CodeActAgent.step = patched_step
