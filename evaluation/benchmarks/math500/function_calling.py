"""Custom function calling implementation for the MATH-500 benchmark."""

import json
from typing import Any, Dict, List

from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
    ModelResponse,
)

from openhands.agenthub.codeact_agent.function_calling import (
    CmdRunTool,
    IPythonTool,
    StrReplaceEditorTool,
    WebReadTool,
    BrowserTool,
    response_to_actions as original_response_to_actions,
)
from openhands.events.action import Action, AgentFinishAction

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
                action = AgentFinishAction(outputs={'answer': answer})
            else:
                # Use the original response_to_actions function for other tools
                # This is a hack, but it should work for our purposes
                temp_response = ModelResponse(
                    id=response.id,
                    choices=[
                        type('Choice', (), {
                            'message': type('Message', (), {
                                'tool_calls': [tool_call],
                                'content': assistant_msg.content,
                            }),
                        })
                    ],
                )
                temp_actions = original_response_to_actions(temp_response)
                if temp_actions:
                    action = temp_actions[0]
                else:
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