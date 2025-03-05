"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

import json

from litellm import (
    ChatCompletionToolParam,
    ModelResponse,
)

from openhands.agenthub.codeact_agent.tools import (
    BrowserTool,
    CmdRunTool,
    FinishTool,
    IPythonTool,
    LLMBasedFileEditTool,
    StrReplaceEditorTool,
    ThinkTool,
    WebReadTool,
)
from openhands.core.exceptions import (
    FunctionCallNotExistsError,
    FunctionCallValidationError,
)
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import (
    Action,
    AgentDelegateAction,
    AgentFinishAction,
    AgentThinkAction,
    BrowseInteractiveAction,
    BrowseURLAction,
    CmdRunAction,
    FileEditAction,
    FileReadAction,
    IPythonRunCellAction,
    MessageAction,
)
from openhands.events.event import FileEditSource, FileReadSource
from openhands.events.tool import ToolCallMetadata


def combine_thought(action: Action, thought: str) -> Action:
    if not hasattr(action, 'thought'):
        return action
    if thought and action.thought:
        action.thought = f'{thought}\n{action.thought}'
    elif thought:
        action.thought = thought
    return action


def response_to_actions(response: ModelResponse, agent=None) -> list[Action]:
    actions: list[Action] = []
    
    # First, check if we can directly extract an execute_ipython_cell call from the content
    # This is a common operation and we want to handle it directly before any complex parsing
    if hasattr(response, 'choices') and len(response.choices) > 0:
        assistant_msg = response.choices[0].message
        if hasattr(assistant_msg, 'content') and assistant_msg.content:
            content = assistant_msg.content
            import re
            
            # Direct pattern match for IPython cell execution
            ipython_pattern = r'<function=execute_ipython_cell>.*?<parameter=code>(.*?)</parameter>.*?</function>'
            ipython_match = re.search(ipython_pattern, content, re.DOTALL)
            
            if ipython_match:
                # We found a direct match for execute_ipython_cell
                code = ipython_match.group(1)
                logger.info('Directly extracted IPython code from content')
                actions.append(IPythonRunCellAction(code=code))
                return actions
            
            # Fallback: Check if there's a code parameter anywhere in the content
            # This is a strong indicator of an execute_ipython_cell call even if malformed
            if '<parameter=code>' in content:
                code_match = re.search(r'<parameter=code>(.*?)</parameter>', content, re.DOTALL)
                if code_match:
                    code = code_match.group(1)
                    logger.info('Found code parameter in content, creating IPythonRunCellAction')
                    actions.append(IPythonRunCellAction(code=code))
                    return actions
    
    # If we get here and there's an error about missing parameters, try to recover
    if hasattr(response, 'error') and 'Missing required parameters for function' in str(response.error):
        logger.warning(f'Detected error in function call: {response.error}')
        # Try to extract the actual function call from the content
        if hasattr(response, 'choices') and len(response.choices) > 0:
            assistant_msg = response.choices[0].message
            if hasattr(assistant_msg, 'content') and assistant_msg.content:
                import re
                
                # Check for function name and code parameter
                function_match = re.search(r'<function=([^>]+)>', assistant_msg.content)
                code_match = re.search(r'<parameter=code>(.*?)</parameter>', assistant_msg.content, re.DOTALL)
                
                if function_match and function_match.group(1) == 'execute_ipython_cell' and code_match:
                    code = code_match.group(1)
                    logger.info('Recovered IPython code from error state')
                    actions.append(IPythonRunCellAction(code=code))
                    return actions

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

            # ================================================
            # CmdRunTool (Bash)
            # ================================================

            if tool_call.function.name == CmdRunTool['function']['name']:
                if 'command' not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "command" in tool call {tool_call.function.name}'
                    )
                # convert is_input to boolean
                is_input = arguments.get('is_input', 'false') == 'true'
                action = CmdRunAction(command=arguments['command'], is_input=is_input)

            # ================================================
            # IPythonTool (Jupyter)
            # ================================================
            elif tool_call.function.name == IPythonTool['function']['name']:
                if 'code' not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "code" in tool call {tool_call.function.name}'
                    )
                action = IPythonRunCellAction(code=arguments['code'])
            elif tool_call.function.name == 'delegate_to_browsing_agent':
                action = AgentDelegateAction(
                    agent='BrowsingAgent',
                    inputs=arguments,
                )

            # ================================================
            # AgentFinishAction
            # ================================================
            elif tool_call.function.name == FinishTool['function']['name']:
                # Validate required parameters for finish function
                if 'message' not in arguments:
                    logger.warning(
                        "Missing required parameter 'message' for finish function"
                    )
                    # Instead of raising an error, provide a default value
                    arguments['message'] = 'Task completed.'

                if 'task_completed' not in arguments:
                    logger.warning(
                        "Missing required parameter 'task_completed' for finish function"
                    )
                    # Instead of raising an error, provide a default value
                    arguments['task_completed'] = 'true'

                # Check if Python has been used (if agent is provided)
                if agent and hasattr(agent, 'python_used') and not agent.python_used:
                    # Python hasn't been used, create a message action instead
                    error_message = 'I need to use Python to solve this problem. Let me try using Python first.'
                    logger.warning(
                        "Blocked finish action because Python hasn't been used yet"
                    )
                    action = MessageAction(
                        content=error_message,
                        wait_for_response=False,
                    )
                # Check if this is the first time the agent is trying to finish
                elif (
                    agent
                    and hasattr(agent, 'has_tried_finish')
                    and not agent.has_tried_finish
                ):
                    # First time trying to finish, ask for verification
                    agent.has_tried_finish = True
                    agent.saved_finish_args = arguments  # Save the arguments for later
                    verification_message = 'Have you verified your solution with code? Please run one final verification to confirm your answer is correct.'
                    logger.info(
                        'Asking for verification before accepting finish action'
                    )
                    action = MessageAction(
                        content=verification_message,
                        wait_for_response=False,
                    )
                else:
                    # Python has been used and either verification was done or agent not provided, proceed with finish
                    action = AgentFinishAction(
                        final_thought=arguments.get('message', ''),
                        task_completed=arguments.get('task_completed', None),
                        solution=arguments.get('solution', ''),
                    )

            # ================================================
            # LLMBasedFileEditTool (LLM-based file editor, deprecated)
            # ================================================
            elif tool_call.function.name == LLMBasedFileEditTool['function']['name']:
                if 'path' not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "path" in tool call {tool_call.function.name}'
                    )
                if 'content' not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "content" in tool call {tool_call.function.name}'
                    )
                action = FileEditAction(
                    path=arguments['path'],
                    content=arguments['content'],
                    start=arguments.get('start', 1),
                    end=arguments.get('end', -1),
                )
            elif tool_call.function.name == StrReplaceEditorTool['function']['name']:
                if 'command' not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "command" in tool call {tool_call.function.name}'
                    )
                if 'path' not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "path" in tool call {tool_call.function.name}'
                    )
                path = arguments['path']
                command = arguments['command']
                other_kwargs = {
                    k: v for k, v in arguments.items() if k not in ['command', 'path']
                }

                if command == 'view':
                    action = FileReadAction(
                        path=path,
                        impl_source=FileReadSource.OH_ACI,
                        view_range=other_kwargs.get('view_range', None),
                    )
                else:
                    if 'view_range' in other_kwargs:
                        # Remove view_range from other_kwargs since it is not needed for FileEditAction
                        other_kwargs.pop('view_range')
                    action = FileEditAction(
                        path=path,
                        command=command,
                        impl_source=FileEditSource.OH_ACI,
                        **other_kwargs,
                    )
            # ================================================
            # AgentThinkAction
            # ================================================
            elif tool_call.function.name == ThinkTool['function']['name']:
                action = AgentThinkAction(thought=arguments.get('thought', ''))

            # ================================================
            # BrowserTool
            # ================================================
            elif tool_call.function.name == BrowserTool['function']['name']:
                if 'code' not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "code" in tool call {tool_call.function.name}'
                    )
                action = BrowseInteractiveAction(browser_actions=arguments['code'])

            # ================================================
            # WebReadTool (simplified browsing)
            # ================================================
            elif tool_call.function.name == WebReadTool['function']['name']:
                if 'url' not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "url" in tool call {tool_call.function.name}'
                    )
                action = BrowseURLAction(url=arguments['url'])
            else:
                raise FunctionCallNotExistsError(
                    f'Tool {tool_call.function.name} is not registered. (arguments: {arguments}). Please check the tool name and retry with an existing tool.'
                )

            # We only add thought to the first action
            if i == 0:
                action = combine_thought(action, thought)
            # Add metadata for tool calling
            action.tool_call_metadata = ToolCallMetadata(
                tool_call_id=tool_call.id,
                function_name=tool_call.function.name,
                model_response=response,
                total_calls_in_response=len(assistant_msg.tool_calls),
            )
            actions.append(action)
    else:
        actions.append(
            MessageAction(
                content=str(assistant_msg.content) if assistant_msg.content else '',
                wait_for_response=True,
            )
        )

    assert len(actions) >= 1
    return actions


def get_tools(
    codeact_enable_browsing: bool = False,
    codeact_enable_llm_editor: bool = False,
    codeact_enable_jupyter: bool = False,
) -> list[ChatCompletionToolParam]:
    # Default behavior
    tools = [CmdRunTool, FinishTool]
    if codeact_enable_browsing:
        tools.append(WebReadTool)
        tools.append(BrowserTool)
    if codeact_enable_jupyter:
        tools.append(IPythonTool)
    if codeact_enable_llm_editor:
        tools.append(LLMBasedFileEditTool)
    else:
        tools.append(StrReplaceEditorTool)
    return tools
