"""Convert tool calling messages to non-tool calling messages and vice versa.

This will inject prompts so that models that doesn't support tool calling
can still be used with tool calling agents.

We follow format from: https://docs.litellm.ai/docs/completion/function_call
"""

import copy
import json
import re
from typing import Iterable

from litellm import ChatCompletionToolParam

from openhands.core.exceptions import (
    FunctionCallConversionError,
    FunctionCallValidationError,
)

# Inspired by: https://docs.together.ai/docs/llama-3-function-calling#function-calling-w-llama-31-70b
SYSTEM_PROMPT_SUFFIX_TEMPLATE = """
You have access to the following tools:

{description}

If you choose to call a tool ONLY reply in the following format with NO suffix:

<tool=example_tool_name>
<example_parameter_1>value_1</example_parameter_1>
<example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</example_parameter_2>
</tool>

<IMPORTANT>
Reminder:
- Tool calls MUST follow the specified format, start with <tool= and end with </tool>
- Required parameters MUST be specified
- Only call one tool at a time
- You may provide optional reasoning for your tool call in natural language BEFORE the tool call, but NOT after.
- If there is no tool call available, answer the question like normal with your current knowledge and do not tell the user about tool calls
</IMPORTANT>
"""

STOP_WORDS = ['</tool']

# NOTE: we need to make sure this example is always in-sync with the tool interface designed in openhands/agenthub/codeact_agent/function_calling.py
IN_CONTEXT_LEARNING_EXAMPLE_PREFIX = """
Here's a running example of how to perform a task with the provided tools.

--------------------- START OF EXAMPLE ---------------------

USER: Problem: Convert the point $(0,3)$ in rectangular coordinates to polar coordinates. Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$

ASSISTANT: I need to convert the rectangular coordinates $(0,3)$ to polar coordinates $(r,\\theta)$.

For the conversion, I'll use these formulas:
- $r = \\sqrt{x^2 + y^2}$
- $\\theta = \\arctan(\\frac{y}{x})$

Given the point $(0,3)$:
- $x = 0$
- $y = 3$

First, I'll calculate $r$:
$r = \\sqrt{0^2 + 3^2} = \\sqrt{9} = 3$

Now for $\\theta$, I need to calculate $\\arctan(\\frac{y}{x}) = \\arctan(\\frac{3}{0})$

Wait, this is a problem because division by zero is undefined. Since $x = 0$ and $y > 0$, the point lies on the positive y-axis, which corresponds to an angle of $\\theta = \\frac{\\pi}{2}$ or 90 degrees.

Let me verify my reasoning with Python:

<tool=execute_ipython_cell>
<code>
import numpy as np
import math

# Given point
x, y = 0, 3

# Calculate r
r = np.sqrt(x**2 + y**2)
print(f"r = √({x}² + {y}²) = {r}")

# Try to calculate θ using the standard arctan formula
print("Attempting standard arctan formula:")
try:
    theta_standard = np.arctan(y/x)
    print(f"θ = arctan({y}/{x}) = {theta_standard} radians")
except ZeroDivisionError:
    print(f"Error: Division by zero when calculating arctan({y}/{x})")
</code>
</tool>

USER: EXECUTION RESULT of [execute_ipython_cell]:
r = √(0² + 3²) = 3.0
Attempting standard arctan formula:
Error: Division by zero when calculating arctan(3/0)

ASSISTANT: I see that my calculation for $r = 3$ is correct, but as I suspected, the standard arctangent formula leads to a division by zero error when $x = 0$.

For points on the y-axis, I need to use a different approach. Let me try using the `arctan2` function, which is specifically designed to handle all quadrants and special cases:

<tool=execute_ipython_cell>
<code>
import numpy as np

# Given point
x, y = 0, 3

# Calculate θ using arctan2 which handles special cases
theta = np.arctan2(y, x)
print(f"θ = arctan2({y}, {x}) = {theta} radians")
print(f"θ = {np.degrees(theta)} degrees")

# Check if θ is in the range [0, 2π)
if theta < 0:
    theta += 2 * np.pi
    print(f"Adjusted θ to be in [0, 2π) range: {theta} radians")
else:
    print(f"θ is already in [0, 2π) range: {theta} radians")
</code>
</tool>

USER: EXECUTION RESULT of [execute_ipython_cell]:
θ = arctan2(3, 0) = 1.5707963267948966 radians
θ = 90.0 degrees
θ is already in [0, 2π) range: 1.5707963267948966 radians

ASSISTANT: Great! The `arctan2` function confirms that for the point $(0,3)$:
- $\\theta = 1.5707963267948966$ radians
- $\\theta = 90$ degrees
- This angle is already in the required range $[0, 2\\pi)$

This matches my reasoning that when a point is on the positive y-axis, the angle is $\\frac{\\pi}{2}$ or 90 degrees.

Let me double-check that this is the correct representation in polar coordinates by converting back to rectangular coordinates:

<tool=execute_ipython_cell>
<code>
import numpy as np

# Polar coordinates
r = 3.0
theta = 1.5707963267948966  # π/2 radians (90 degrees)

# Convert back to rectangular coordinates
x_back = r * np.cos(theta)
y_back = r * np.sin(theta)

print(f"Converting from polar (r={r}, θ={theta} radians) back to rectangular:")
print(f"x = r·cos(θ) = {r} · cos({theta}) = {x_back}")
print(f"y = r·sin(θ) = {r} · sin({theta}) = {y_back}")

# Check if we get back to our original point (0,3)
print(f"\nOriginal point: (0, 3)")
print(f"Converted back point: ({x_back:.10f}, {y_back:.10f})")

# Check for numerical precision issues
if abs(x_back) < 1e-10:
    print("x is effectively zero (within numerical precision)")
</code>
</tool>

USER: EXECUTION RESULT of [execute_ipython_cell]:
Converting from polar (r=3.0, θ=1.5707963267948966 radians) back to rectangular:
x = r·cos(θ) = 3.0 · cos(1.5707963267948966) = 1.8369701987210297e-16
y = r·sin(θ) = 3.0 · sin(1.5707963267948966) = 3.0

Original point: (0, 3)
Converted back point: (0.0000000000, 3.0000000000)
x is effectively zero (within numerical precision)

ASSISTANT: Perfect! The conversion back to rectangular coordinates confirms our polar representation is correct:
- When converting $(r=3, \\theta=\\frac{\\pi}{2})$ back to rectangular coordinates, we get $(0, 3)$
- The x-value is not exactly zero but $1.8369701987210297 \\times 10^{-16}$, which is effectively zero due to floating-point precision limitations

Let me verify one more thing - I want to make sure I understand the range requirement correctly. The problem asks for $0 \\le \\theta < 2\\pi$:

<tool=execute_ipython_cell>
<code>
import numpy as np

# Check if our theta is in the required range
theta = 1.5707963267948966  # π/2 radians
pi = np.pi

print(f"θ = {theta} radians")
print(f"π/2 = {pi/2} radians")
print(f"2π = {2*pi} radians")

# Check range requirement
in_range = 0 <= theta < 2*pi
print(f"Is θ in the range [0, 2π)? {in_range}")

# Express in terms of π for mathematical notation
print(f"θ = {theta/pi}π radians")
</code>
</tool>

USER: EXECUTION RESULT of [execute_ipython_cell]:
θ = 1.5707963267948966 radians
π/2 = 1.5707963267948966 radians
2π = 6.283185307179586 radians
Is θ in the range [0, 2π)? True
θ = 0.5π radians

ASSISTANT: This confirms that our answer is correct and meets all the requirements:
1. We have $r = 3$, which is positive as required
2. We have $\\theta = \\frac{\\pi}{2} = 1.5707963267948966$ radians, which is in the range $[0, 2\\pi)$
3. When converted back to rectangular coordinates, these polar coordinates correctly represent the original point $(0,3)$

Therefore, the polar coordinates of the point $(0,3)$ are $(3, \\frac{\\pi}{2})$.

<tool=finish>
<solution>\\boxed{(3, \\frac{\\pi}{2})}</solution>
</tool>

--------------------- END OF EXAMPLE ---------------------

Do NOT assume the environment is the same as in the example above.

--------------------- NEW TASK DESCRIPTION ---------------------
""".lstrip()

IN_CONTEXT_LEARNING_EXAMPLE_SUFFIX = """
--------------------- END OF NEW TASK DESCRIPTION ---------------------

PLEASE follow the format strictly! PLEASE EMIT ONE AND ONLY ONE FUNCTION CALL PER MESSAGE.
"""

# Regex patterns for tool call parsing
FN_REGEX_PATTERN = r'<tool=([^>]+)>\n(.*?)</tool>'
FN_PARAM_REGEX_PATTERN = r'<(?!tool=)([^>]+)>(.*?)</\1>'

# Add new regex pattern for tool execution results
TOOL_RESULT_REGEX_PATTERN = r'EXECUTION RESULT of \[(.*?)\]:\n(.*)'


def convert_tool_call_to_string(tool_call: dict) -> str:
    """Convert tool call to content in string format."""
    if 'function' not in tool_call:
        raise FunctionCallConversionError("Tool call must contain 'function' key.")
    if 'id' not in tool_call:
        raise FunctionCallConversionError("Tool call must contain 'id' key.")
    if 'type' not in tool_call:
        raise FunctionCallConversionError("Tool call must contain 'type' key.")
    if tool_call['type'] != 'function':
        raise FunctionCallConversionError("Tool call type must be 'function'.")

    ret = f"<tool={tool_call['function']['name']}>\n"
    try:
        args = json.loads(tool_call['function']['arguments'])
    except json.JSONDecodeError as e:
        raise FunctionCallConversionError(
            f"Failed to parse arguments as JSON. Arguments: {tool_call['function']['arguments']}"
        ) from e
    for param_name, param_value in args.items():
        is_multiline = isinstance(param_value, str) and '\n' in param_value
        ret += f'<{param_name}>'
        if is_multiline:
            ret += '\n'
        ret += f'{param_value}'
        if is_multiline:
            ret += '\n'
        ret += f'</{param_name}>\n'
    ret += '</tool>'
    return ret


def convert_tools_to_description(tools: list[dict]) -> str:
    ret = ''
    for i, tool in enumerate(tools):
        assert tool['type'] == 'function'
        fn = tool['function']
        if i > 0:
            ret += '\n'
        ret += f"---- BEGIN FUNCTION #{i+1}: {fn['name']} ----\n"
        ret += f"Description: {fn['description']}\n"

        if 'parameters' in fn:
            ret += 'Parameters:\n'
            properties = fn['parameters'].get('properties', {})
            required_params = set(fn['parameters'].get('required', []))

            for j, (param_name, param_info) in enumerate(properties.items()):
                # Indicate required/optional in parentheses with type
                is_required = param_name in required_params
                param_status = 'required' if is_required else 'optional'
                param_type = param_info.get('type', 'string')

                # Get parameter description
                desc = param_info.get('description', 'No description provided')

                # Handle enum values if present
                if 'enum' in param_info:
                    enum_values = ', '.join(f'`{v}`' for v in param_info['enum'])
                    desc += f'\nAllowed values: [{enum_values}]'

                ret += (
                    f'  ({j+1}) {param_name} ({param_type}, {param_status}): {desc}\n'
                )
        else:
            ret += 'No parameters are required for this function.\n'

        ret += f'---- END FUNCTION #{i+1} ----\n'
    return ret


def convert_tool_messages_to_non_tool_messages(
    messages: list[dict],
    tools: list[ChatCompletionToolParam],
    add_in_context_learning_example: bool = True,
) -> list[dict]:
    """Convert tool calling messages to non-tool calling messages."""
    messages = copy.deepcopy(messages)

    formatted_tools = convert_tools_to_description(tools)
    system_prompt_suffix = SYSTEM_PROMPT_SUFFIX_TEMPLATE.format(
        description=formatted_tools
    )

    converted_messages = []
    first_user_message_encountered = False
    for message in messages:
        role = message['role']
        content = message['content']

        # 1. SYSTEM MESSAGES
        # append system prompt suffix to content
        if role == 'system':
            if isinstance(content, str):
                content += system_prompt_suffix
            elif isinstance(content, list):
                if content and content[-1]['type'] == 'text':
                    content[-1]['text'] += system_prompt_suffix
                else:
                    content.append({'type': 'text', 'text': system_prompt_suffix})
            else:
                raise FunctionCallConversionError(
                    f'Unexpected content type {type(content)}. Expected str or list. Content: {content}'
                )
            converted_messages.append({'role': 'system', 'content': content})

        # 2. USER MESSAGES (no change)
        elif role == 'user':
            # Add in-context learning example for the first user message
            if not first_user_message_encountered and add_in_context_learning_example:
                first_user_message_encountered = True
                # Check tools - need either execute_bash or execute_ipython_cell, and finish
                if not (
                    tools
                    and len(tools) > 0
                    and (
                        # Either bash tool is available
                        any(
                            (
                                tool['type'] == 'function'
                                and tool['function']['name'] == 'execute_bash'
                                and 'parameters' in tool['function']
                                and 'properties' in tool['function']['parameters']
                                and 'command' in tool['function']['parameters']['properties']
                            )
                            for tool in tools
                        )
                        or
                        # Or IPython tool is available
                        any(
                            (
                                tool['type'] == 'function'
                                and tool['function']['name'] == 'execute_ipython_cell'
                                and 'parameters' in tool['function']
                                and 'properties' in tool['function']['parameters']
                                and 'code' in tool['function']['parameters']['properties']
                            )
                            for tool in tools
                        )
                    )
                    and any(
                        (
                            tool['type'] == 'function'
                            and tool['function']['name'] == 'finish'
                        )
                        for tool in tools
                    )
                ):
                    raise FunctionCallConversionError(
                        'The currently provided tool set are NOT compatible with the in-context learning example for FnCall to Non-FnCall conversion. '
                        'Please update your tool set OR the in-context learning example in openhands/llm/fn_call_converter.py'
                    )

                # add in-context learning example
                if isinstance(content, str):
                    content = (
                        IN_CONTEXT_LEARNING_EXAMPLE_PREFIX
                        + content
                        + IN_CONTEXT_LEARNING_EXAMPLE_SUFFIX
                    )
                elif isinstance(content, list):
                    if content and content[0]['type'] == 'text':
                        content[0]['text'] = (
                            IN_CONTEXT_LEARNING_EXAMPLE_PREFIX
                            + content[0]['text']
                            + IN_CONTEXT_LEARNING_EXAMPLE_SUFFIX
                        )
                    else:
                        content = (
                            [
                                {
                                    'type': 'text',
                                    'text': IN_CONTEXT_LEARNING_EXAMPLE_PREFIX,
                                }
                            ]
                            + content
                            + [
                                {
                                    'type': 'text',
                                    'text': IN_CONTEXT_LEARNING_EXAMPLE_SUFFIX,
                                }
                            ]
                        )
                else:
                    raise FunctionCallConversionError(
                        f'Unexpected content type {type(content)}. Expected str or list. Content: {content}'
                    )
            converted_messages.append(
                {
                    'role': 'user',
                    'content': content,
                }
            )

        # 3. ASSISTANT MESSAGES
        # - 3.1 no change if no function call
        # - 3.2 change if function call
        elif role == 'assistant':
            if 'tool_calls' in message and message['tool_calls'] is not None:
                if len(message['tool_calls']) != 1:
                    raise FunctionCallConversionError(
                        f'Expected exactly one tool call in the message. More than one tool call is not supported. But got {len(message["tool_calls"])} tool calls. Content: {content}'
                    )
                try:
                    tool_content = convert_tool_call_to_string(message['tool_calls'][0])
                except FunctionCallConversionError as e:
                    raise FunctionCallConversionError(
                        f'Failed to convert tool call to string.\nCurrent tool call: {message["tool_calls"][0]}.\nRaw messages: {json.dumps(messages, indent=2)}'
                    ) from e
                if isinstance(content, str):
                    content += '\n\n' + tool_content
                    content = content.lstrip()
                elif isinstance(content, list):
                    if content and content[-1]['type'] == 'text':
                        content[-1]['text'] += '\n\n' + tool_content
                        content[-1]['text'] = content[-1]['text'].lstrip()
                    else:
                        content.append({'type': 'text', 'text': tool_content})
                else:
                    raise FunctionCallConversionError(
                        f'Unexpected content type {type(content)}. Expected str or list. Content: {content}'
                    )
            converted_messages.append({'role': 'assistant', 'content': content})

        # 4. TOOL MESSAGES (tool outputs)
        elif role == 'tool':
            # Convert tool result as user message
            tool_name = message.get('name', 'function')
            prefix = f'EXECUTION RESULT of [{tool_name}]:\n'
            # and omit "tool_call_id" AND "name"
            if isinstance(content, str):
                content = prefix + content
            elif isinstance(content, list):
                if content and content[-1]['type'] == 'text':
                    content[-1]['text'] = prefix + content[-1]['text']
                else:
                    content = [{'type': 'text', 'text': prefix}] + content
            else:
                raise FunctionCallConversionError(
                    f'Unexpected content type {type(content)}. Expected str or list. Content: {content}'
                )
            converted_messages.append({'role': 'user', 'content': content})
        else:
            raise FunctionCallConversionError(
                f'Unexpected role {role}. Expected system, user, assistant or tool.'
            )
    return converted_messages


def _extract_and_validate_params(
    matching_tool: dict, param_matches: Iterable[re.Match], fn_name: str
) -> dict:
    params = {}
    # Parse and validate parameters
    required_params = set()
    if 'parameters' in matching_tool and 'required' in matching_tool['parameters']:
        required_params = set(matching_tool['parameters'].get('required', []))

    allowed_params = set()
    if 'parameters' in matching_tool and 'properties' in matching_tool['parameters']:
        allowed_params = set(matching_tool['parameters']['properties'].keys())

    param_name_to_type = {}
    if 'parameters' in matching_tool and 'properties' in matching_tool['parameters']:
        param_name_to_type = {
            name: val.get('type', 'string')
            for name, val in matching_tool['parameters']['properties'].items()
        }

    # Collect parameters
    found_params = set()
    for param_match in param_matches:
        param_name = param_match.group(1)
        param_value = param_match.group(2).strip()

        # Validate parameter is allowed
        if allowed_params and param_name not in allowed_params:
            raise FunctionCallValidationError(
                f"Parameter '{param_name}' is not allowed for function '{fn_name}'. "
                f'Allowed parameters: {allowed_params}'
            )

        # Validate and convert parameter type
        # supported: string, integer, array
        if param_name in param_name_to_type:
            if param_name_to_type[param_name] == 'integer':
                try:
                    param_value = int(param_value)
                except ValueError:
                    raise FunctionCallValidationError(
                        f"Parameter '{param_name}' is expected to be an integer."
                    )
            elif param_name_to_type[param_name] == 'array':
                try:
                    param_value = json.loads(param_value)
                except json.JSONDecodeError:
                    raise FunctionCallValidationError(
                        f"Parameter '{param_name}' is expected to be an array."
                    )
            else:
                # string
                pass

        # Enum check
        if ('parameters' in matching_tool and 
            'properties' in matching_tool['parameters'] and 
            param_name in matching_tool['parameters']['properties'] and
            'enum' in matching_tool['parameters']['properties'][param_name]):
            if (
                param_value
                not in matching_tool['parameters']['properties'][param_name]['enum']
            ):
                raise FunctionCallValidationError(
                    f"Parameter '{param_name}' is expected to be one of {matching_tool['parameters']['properties'][param_name]['enum']}."
                )

        params[param_name] = param_value
        found_params.add(param_name)

    # Check all required parameters are present
    missing_params = required_params - found_params
    if missing_params:
        raise FunctionCallValidationError(
            f"Missing required parameters for function '{fn_name}': {missing_params}"
        )
    return params


def _fix_stopword(content: str) -> str:
    """Fix the issue when some LLM would NOT return the stopword."""
    if '<tool=' in content and content.count('<tool=') == 1:
        if content.endswith('</'):
            content = content.rstrip() + 'tool>'
        else:
            content = content + '\n</tool>'
    return content


def convert_non_tool_messages_to_tool_messages(
    messages: list[dict],
    tools: list[ChatCompletionToolParam],
) -> list[dict]:
    """Convert non-tool calling messages back to tool calling messages."""
    messages = copy.deepcopy(messages)
    formatted_tools = convert_tools_to_description(tools)
    system_prompt_suffix = SYSTEM_PROMPT_SUFFIX_TEMPLATE.format(
        description=formatted_tools
    )

    converted_messages = []
    tool_call_counter = 1  # Counter for tool calls

    first_user_message_encountered = False
    for message in messages:
        role, content = message['role'], message['content']
        content = content or ''  # handle cases where content is None
        # For system messages, remove the added suffix
        if role == 'system':
            if isinstance(content, str):
                # Remove the suffix if present
                content = content.split(system_prompt_suffix)[0]
            elif isinstance(content, list):
                if content and content[-1]['type'] == 'text':
                    # Remove the suffix from the last text item
                    content[-1]['text'] = content[-1]['text'].split(
                        system_prompt_suffix
                    )[0]
            converted_messages.append({'role': 'system', 'content': content})
        # Skip user messages (no conversion needed)
        elif role == 'user':
            # Check & replace in-context learning example
            if not first_user_message_encountered:
                first_user_message_encountered = True
                if isinstance(content, str):
                    content = content.replace(IN_CONTEXT_LEARNING_EXAMPLE_PREFIX, '')
                    content = content.replace(IN_CONTEXT_LEARNING_EXAMPLE_SUFFIX, '')
                elif isinstance(content, list):
                    for item in content:
                        if item['type'] == 'text':
                            item['text'] = item['text'].replace(
                                IN_CONTEXT_LEARNING_EXAMPLE_PREFIX, ''
                            )
                            item['text'] = item['text'].replace(
                                IN_CONTEXT_LEARNING_EXAMPLE_SUFFIX, ''
                            )
                else:
                    raise FunctionCallConversionError(
                        f'Unexpected content type {type(content)}. Expected str or list. Content: {content}'
                    )

            # Check for tool execution result pattern
            if isinstance(content, str):
                tool_result_match = re.search(
                    TOOL_RESULT_REGEX_PATTERN, content, re.DOTALL
                )
            elif isinstance(content, list):
                tool_result_match = next(
                    (
                        _match
                        for item in content
                        if item.get('type') == 'text'
                        and (
                            _match := re.search(
                                TOOL_RESULT_REGEX_PATTERN, item['text'], re.DOTALL
                            )
                        )
                    ),
                    None,
                )
            else:
                raise FunctionCallConversionError(
                    f'Unexpected content type {type(content)}. Expected str or list. Content: {content}'
                )

            if tool_result_match:
                if not (
                    isinstance(content, str)
                    or (
                        isinstance(content, list)
                        and len(content) == 1
                        and content[0].get('type') == 'text'
                    )
                ):
                    raise FunctionCallConversionError(
                        f'Expected str or list with one text item when tool result is present in the message. Content: {content}'
                    )
                tool_name = tool_result_match.group(1)
                tool_result = tool_result_match.group(2).strip()

                # Convert to tool message format
                converted_messages.append(
                    {
                        'role': 'tool',
                        'name': tool_name,
                        'content': [{'type': 'text', 'text': tool_result}]
                        if isinstance(content, list)
                        else tool_result,
                        'tool_call_id': f'toolu_{tool_call_counter-1:02d}',  # Use last generated ID
                    }
                )
            else:
                converted_messages.append({'role': 'user', 'content': content})

        # Handle assistant messages
        elif role == 'assistant':
            if isinstance(content, str):
                content = _fix_stopword(content)
                tool_match = re.search(FN_REGEX_PATTERN, content, re.DOTALL)
            elif isinstance(content, list):
                if content and content[-1]['type'] == 'text':
                    content[-1]['text'] = _fix_stopword(content[-1]['text'])
                    tool_match = re.search(
                        FN_REGEX_PATTERN, content[-1]['text'], re.DOTALL
                    )
                else:
                    tool_match = None
                tool_match_exists = any(
                    item.get('type') == 'text'
                    and re.search(FN_REGEX_PATTERN, item['text'], re.DOTALL)
                    for item in content
                )
                if tool_match_exists and not tool_match:
                    raise FunctionCallConversionError(
                        f'Expecting tool call in the LAST index of content list. But got content={content}'
                    )
            else:
                raise FunctionCallConversionError(
                    f'Unexpected content type {type(content)}. Expected str or list. Content: {content}'
                )

            if tool_match:
                tool_name = tool_match.group(1)
                tool_body = tool_match.group(2)
                matching_tool = next(
                    (
                        tool['function']
                        for tool in tools
                        if tool['type'] == 'function'
                        and tool['function']['name'] == tool_name
                    ),
                    None,
                )
                # Validate function exists in tools
                if not matching_tool:
                    raise FunctionCallValidationError(
                        f"Tool '{tool_name}' not found in available tools: {[tool['function']['name'] for tool in tools if tool['type'] == 'function']}"
                    )

                # Parse parameters
                param_matches = re.finditer(FN_PARAM_REGEX_PATTERN, tool_body, re.DOTALL)
                params = _extract_and_validate_params(
                    matching_tool, param_matches, tool_name
                )

                # Create tool call with unique ID
                tool_call_id = f'toolu_{tool_call_counter:02d}'
                tool_call = {
                    'index': 1,  # always 1 because we only support **one tool call per message**
                    'id': tool_call_id,
                    'type': 'function',
                    'function': {'name': tool_name, 'arguments': json.dumps(params)},
                }
                tool_call_counter += 1  # Increment counter

                # Remove the function call part from content
                if isinstance(content, list):
                    assert content and content[-1]['type'] == 'text'
                    content[-1]['text'] = (
                        content[-1]['text'].split('<tool=')[0].strip()
                    )
                elif isinstance(content, str):
                    content = content.split('<tool=')[0].strip()
                else:
                    raise FunctionCallConversionError(
                        f'Unexpected content type {type(content)}. Expected str or list. Content: {content}'
                    )

                converted_messages.append(
                    {'role': 'assistant', 'content': content, 'tool_calls': [tool_call]}
                )
            else:
                # No function call, keep message as is
                converted_messages.append(message)

        else:
            raise FunctionCallConversionError(
                f'Unexpected role {role}. Expected system, user, or assistant in non-tool calling messages.'
            )
    return converted_messages


def convert_from_multiple_tool_calls_to_single_tool_call_messages(
    messages: list[dict],
    ignore_final_tool_result: bool = False,
) -> list[dict]:
    """Break one message with multiple tool calls into multiple messages."""
    converted_messages = []

    pending_tool_calls: dict[str, dict] = {}
    for message in messages:
        role, content = message['role'], message['content']
        if role == 'assistant':
            if message.get('tool_calls') and len(message['tool_calls']) > 1:
                # handle multiple tool calls by breaking them into multiple messages
                for i, tool_call in enumerate(message['tool_calls']):
                    pending_tool_calls[tool_call['id']] = {
                        'role': 'assistant',
                        'content': content if i == 0 else '',
                        'tool_calls': [tool_call],
                    }
            else:
                converted_messages.append(message)
        elif role == 'tool':
            if message['tool_call_id'] in pending_tool_calls:
                # remove the tool call from the pending list
                _tool_call_message = pending_tool_calls.pop(message['tool_call_id'])
                converted_messages.append(_tool_call_message)
                # add the tool result
                converted_messages.append(message)
            else:
                assert (
                    len(pending_tool_calls) == 0
                ), f'Found pending tool calls but not found in pending list: {pending_tool_calls=}'
                converted_messages.append(message)
        else:
            assert (
                len(pending_tool_calls) == 0
            ), f'Found pending tool calls but not expect to handle it with role {role}: {pending_tool_calls=}, {message=}'
            converted_messages.append(message)

    if not ignore_final_tool_result and len(pending_tool_calls) > 0:
        raise FunctionCallConversionError(
            f'Found pending tool calls but no tool result: {pending_tool_calls=}'
        )
    return converted_messages
