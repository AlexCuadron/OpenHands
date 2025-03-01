"""Custom function calling implementation for the MATH-500 benchmark."""

import json
from typing import List

from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
    ModelResponse,
)

from openhands.agenthub.codeact_agent.function_calling import (
    BrowserTool,
    CmdRunTool,
    IPythonTool,
    StrReplaceEditorTool,
    WebReadTool,
)
from openhands.agenthub.codeact_agent.function_calling import (
    response_to_actions as original_response_to_actions,
)
from openhands.events.action import (
    Action,
    AgentFinishAction,
    BrowserAction,
    CmdRunAction,
    IPythonCellAction,
    StrReplaceEditorAction,
    WebReadAction,
)

# Custom finish tool description that accepts an optional answer parameter
_FINISH_DESCRIPTION = """Finish the interaction when the task is complete OR if the assistant cannot proceed further with the task.

Parameters:
- answer: (Optional) Your final answer to the math problem. Make sure it matches the expected format in the problem statement.
"""

# Custom finish tool that accepts an optional answer parameter
FinishWithAnswerTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='finish',
        description=_FINISH_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'answer': {
                    'type': 'string',
                    'description': 'Your final answer to the math problem.',
                },
            },
            'required': [],  # Make answer optional
        },
    ),
)


# Override the get_tools function to use our custom finish tool
def get_tools(
    codeact_enable_browsing: bool = False,
    codeact_enable_llm_editor: bool = False,
    codeact_enable_jupyter: bool = False,
) -> List[ChatCompletionToolParam]:
    """Get the tools for the CodeActAgent.

    This is a custom version of the get_tools function that uses our custom finish tool.
    """
    tools = [CmdRunTool, FinishWithAnswerTool]
    if codeact_enable_browsing:
        tools.append(WebReadTool)
        tools.append(BrowserTool)
    if codeact_enable_jupyter:
        tools.append(IPythonTool)
    if codeact_enable_llm_editor:
        tools.append(StrReplaceEditorTool)
    return tools


# Override the response_to_actions function to handle our custom finish tool
def response_to_actions(response: ModelResponse) -> List[Action]:
    """Convert a model response to a list of actions.

    This is a custom version of the response_to_actions function that handles our custom finish tool.
    """
    actions: List[Action] = []
    assert len(response.choices) == 1, 'Only one choice is supported for now'
    choice = response.choices[0]
    assistant_msg = choice.message
    if hasattr(assistant_msg, 'tool_calls') and assistant_msg.tool_calls:
        # Check if there's assistant_msg.content. If so, add it to the thought
        thought = ''
        if isinstance(assistant_msg.content, str):
            thought = assistant_msg.content
        elif isinstance(assistant_msg.content, list):
            for msg in assistant_msg.content:
                if msg['type'] == 'text':
                    thought += msg['text']

        # Process each tool call to OpenHands action
        for i, tool_call in enumerate(assistant_msg.tool_calls):
            action: Action
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.decoder.JSONDecodeError as e:
                raise RuntimeError(
                    f'Failed to parse tool call arguments: {tool_call.function.arguments}'
                ) from e

            # Handle our custom finish tool
            if tool_call.function.name == 'finish':
                answer = arguments.get('answer', '')
                # Use both the solution parameter and the outputs dictionary for backward compatibility
                action = AgentFinishAction(outputs={'answer': answer}, solution=answer)
            else:
                # Instead of creating a ModelResponse, directly call the original function
                # with the current tool_call
                try:
                    # Create a simple action based on the tool call
                    if tool_call.function.name == 'execute_bash':
                        command = arguments.get('command', '')
                        is_input = arguments.get('is_input', 'false') == 'true'
                        action = CmdRunAction(command=command, is_input=is_input)
                    elif tool_call.function.name == 'execute_ipython_cell':
                        code = arguments.get('code', '')
                        action = IPythonCellAction(code=code)
                    elif tool_call.function.name == 'web_read':
                        url = arguments.get('url', '')
                        action = WebReadAction(url=url)
                    elif tool_call.function.name == 'browser':
                        code = arguments.get('code', '')
                        action = BrowserAction(code=code)
                    elif tool_call.function.name == 'str_replace_editor':
                        command = arguments.get('command', '')
                        path = arguments.get('path', '')
                        file_text = arguments.get('file_text', '')
                        old_str = arguments.get('old_str', '')
                        new_str = arguments.get('new_str', '')
                        insert_line = arguments.get('insert_line', 0)
                        view_range = arguments.get('view_range', None)
                        action = StrReplaceEditorAction(
                            command=command,
                            path=path,
                            file_text=file_text,
                            old_str=old_str,
                            new_str=new_str,
                            insert_line=insert_line,
                            view_range=view_range
                        )
                    else:
                        # Skip unknown tools
                        continue
                except Exception as e:
                    # If there's an error, skip this tool call
                    print(f"Error processing tool call: {e}")
                    continue

            # Add thought to the action
            if thought and hasattr(action, 'thought'):
                action.thought = thought

            actions.append(action)
    else:
        # If no tool calls, create a message action
        content = assistant_msg.content
        if isinstance(content, list):
            content = ''.join(msg['text'] for msg in content if msg['type'] == 'text')
        actions.append(Action.from_content(content))

    return actions
