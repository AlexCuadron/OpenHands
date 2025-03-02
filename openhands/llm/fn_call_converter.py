"""Convert function calling messages to non-function calling messages and vice versa.

This will inject prompts so that models that doesn't support function calling
can still be used with function calling agents.

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
You have access to the following functions:

{description}

If you choose to call a function ONLY reply in the following format with NO suffix:

<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format, start with <function= and end with </function>
- ALWAYS include the closing </function> tag for EVERY function call
- Required parameters MUST be specified
- Only call one function at a time
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after.
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
- For benchmark problems, ALWAYS use the finish function with the solution parameter when providing your final answer
</IMPORTANT>
"""

STOP_WORDS = ['</function']

# NOTE: we need to make sure this example is always in-sync with the tool interface designed in openhands/agenthub/codeact_agent/function_calling.py
IN_CONTEXT_LEARNING_EXAMPLE_PREFIX = """
Here's a running example of how to break down a complex problem into sub-problems and solve each one using tools.

--------------------- START OF EXAMPLE ---------------------

USER: Problem: Find the number of ordered pairs of integers $(a, b)$ such that the sequence $[3, 4, 5, a, b, 30, 40, 50]$ is strictly increasing and no set of four (not necessarily consecutive) terms forms an arithmetic progression.

ASSISTANT: I'll solve this problem by breaking it down into sub-problems and using Python to verify each step.

First, I need to install the necessary libraries:

<function=execute_ipython_cell>
<parameter=code>
# Install necessary libraries
import itertools
import numpy as np
</parameter>
</function>

USER: EXECUTION RESULT of [execute_ipython_cell]:
[Jupyter current working directory: /workspace]
[Jupyter Python interpreter: /openhands/poetry/openhands-ai-5O4_aCHf-py3.12/bin/python]

ASSISTANT: Now I'll break this problem into sub-problems:

Sub-problem 1: Understand the constraints on a and b for the sequence to be strictly increasing.
Sub-problem 2: Define a function to check if four numbers form an arithmetic progression.
Sub-problem 3: Identify specific invalid pairs (a,b) that create arithmetic progressions.
Sub-problem 4: Count all valid pairs by checking each possible (a,b) combination.

Let me start with sub-problem 1:

<function=execute_ipython_cell>
<parameter=code>
# Sub-problem 1: Understand the constraints on a and b

# For the sequence [3, 4, 5, a, b, 30, 40, 50] to be strictly increasing:
# 5 < a < b < 30

# Calculate possible values for a and b
a_min = 6  # a must be greater than 5
a_max = 28  # a must be less than b, which must be less than 30
b_min_formula = "a + 1"  # b must be greater than a
b_max = 29  # b must be less than 30

# Calculate total possible pairs before checking AP constraint
total_pairs = sum(29 - a for a in range(6, 29))

print(f"Constraints: 5 < a < b < 30")
print(f"Possible values for a: {a_min} to {a_max}")
print(f"For each a, possible values for b: {b_min_formula} to {b_max}")
print(f"Total possible pairs before checking AP constraint: {total_pairs}")
</parameter>
</function>

USER: EXECUTION RESULT of [execute_ipython_cell]:
Constraints: 5 < a < b < 30
Possible values for a: 6 to 28
For each a, possible values for b: a + 1 to 29
Total possible pairs before checking AP constraint: 276
[Jupyter current working directory: /workspace]
[Jupyter Python interpreter: /openhands/poetry/openhands-ai-5O4_aCHf-py3.12/bin/python]

ASSISTANT: Now I'll tackle sub-problem 2 - defining a function to check if four numbers form an arithmetic progression:

<function=execute_ipython_cell>
<parameter=code>
# Sub-problem 2: Define a function to check if four numbers form an arithmetic progression

def is_arithmetic_progression(numbers):
    """
    Check if a set of numbers forms an arithmetic progression.
    An arithmetic progression has equal differences between consecutive terms.
    """
    if len(numbers) < 2:
        return False
    
    # Sort the numbers (since we're checking any four terms, not necessarily in order)
    sorted_nums = sorted(numbers)
    
    # Calculate the common difference
    d = sorted_nums[1] - sorted_nums[0]
    
    # Check if all consecutive pairs have the same difference
    for i in range(1, len(sorted_nums) - 1):
        if sorted_nums[i + 1] - sorted_nums[i] != d:
            return False
    
    return True

# Test the function with some examples
test_cases = [
    ([3, 4, 5, 6], True),      # AP with difference 1
    ([3, 5, 7, 9], True),      # AP with difference 2
    ([10, 20, 30, 40], True),  # AP with difference 10
    ([3, 4, 6, 9], False),     # Not an AP
    ([3, 5, 8, 10], False)     # Not an AP
]

for nums, expected in test_cases:
    result = is_arithmetic_progression(nums)
    print(f"{nums}: {'is AP' if result else 'not AP'} (Expected: {'AP' if expected else 'not AP'})")
</parameter>
</function>

USER: EXECUTION RESULT of [execute_ipython_cell]:
[3, 4, 5, 6]: is AP (Expected: AP)
[3, 5, 7, 9]: is AP (Expected: AP)
[10, 20, 30, 40]: is AP (Expected: AP)
[3, 4, 6, 9]: not AP (Expected: not AP)
[3, 5, 8, 10]: not AP (Expected: not AP)
[Jupyter current working directory: /workspace]
[Jupyter Python interpreter: /openhands/poetry/openhands-ai-5O4_aCHf-py3.12/bin/python]

ASSISTANT: Now for sub-problem 3, I'll identify some specific invalid pairs (a,b) that create arithmetic progressions:

<function=execute_ipython_cell>
<parameter=code>
# Sub-problem 3: Identify specific invalid pairs (a,b) that create arithmetic progressions

def check_invalid_pair(a, b):
    """Check if a specific (a,b) pair creates a sequence with a four-term AP"""
    sequence = [3, 4, 5, a, b, 30, 40, 50]
    
    # Check all possible 4-element combinations
    for comb in itertools.combinations(sequence, 4):
        if is_arithmetic_progression(comb):
            return True, comb
    
    return False, None

# Test some specific pairs
test_pairs = [
    (6, 7),    # Should be invalid: [3,4,5,6] forms an AP
    (10, 20),  # Should be invalid: [10,20,30,40] forms an AP
    (12, 21),  # Should be invalid: [3,12,21,30] forms an AP
    (7, 9),    # Should be invalid: [3,5,7,9] forms an AP
    (8, 9)     # Should be valid (no obvious AP)
]

for a, b in test_pairs:
    invalid, ap_found = check_invalid_pair(a, b)
    status = "Invalid" if invalid else "Valid"
    ap_info = f" - AP found: {ap_found}" if invalid else ""
    print(f"Pair (a={a}, b={b}): {status}{ap_info}")
</parameter>
</function>

USER: EXECUTION RESULT of [execute_ipython_cell]:
Pair (a=6, b=7): Invalid - AP found: (3, 4, 5, 6)
Pair (a=10, b=20): Invalid - AP found: (10, 20, 30, 40)
Pair (a=12, b=21): Invalid - AP found: (3, 12, 21, 30)
Pair (a=7, b=9): Invalid - AP found: (3, 5, 7, 9)
Pair (a=8, b=9): Valid
[Jupyter current working directory: /workspace]
[Jupyter Python interpreter: /openhands/poetry/openhands-ai-5O4_aCHf-py3.12/bin/python]

ASSISTANT: Now I'll solve sub-problem 4 by counting all valid pairs. Since there are 276 possible pairs to check, I'll implement an efficient algorithm:

<function=execute_ipython_cell>
<parameter=code>
# Sub-problem 4: Count all valid pairs by checking each possible (a,b) combination

def count_valid_pairs():
    valid_count = 0
    invalid_count = 0
    
    # Iterate over all possible a values (6 <= a <= 28)
    for a in range(6, 29):
        # For each a, iterate over possible b values (a+1 <= b <= 29)
        for b in range(a + 1, 30):
            sequence = [3, 4, 5, a, b, 30, 40, 50]
            has_ap = False
            
            # Check all 4-element combinations
            for comb in itertools.combinations(sequence, 4):
                if is_arithmetic_progression(comb):
                    has_ap = True
                    invalid_count += 1
                    break  # No need to check further if an AP is found
                    
            if not has_ap:
                valid_count += 1
    
    return valid_count, invalid_count

# Run the counting function
valid_pairs, invalid_pairs = count_valid_pairs()
total_pairs = valid_pairs + invalid_pairs

print(f"Total possible pairs: {total_pairs}")
print(f"Invalid pairs (contain an AP): {invalid_pairs}")
print(f"Valid pairs (no AP): {valid_pairs}")

# Verify our total matches our earlier calculation
print(f"Verification: Total matches earlier calculation: {total_pairs == 276}")
</parameter>
</function>

USER: EXECUTION RESULT of [execute_ipython_cell]:
Total possible pairs: 276
Invalid pairs (contain an AP): 48
Valid pairs (no AP): 228
Verification: Total matches earlier calculation: True
[Jupyter current working directory: /workspace]
[Jupyter Python interpreter: /openhands/poetry/openhands-ai-5O4_aCHf-py3.12/bin/python]

ASSISTANT: Let me verify the result by checking a few more specific cases to make sure our algorithm is correct:

<function=execute_ipython_cell>
<parameter=code>
# Verify our result with additional test cases

# Let's check a few more pairs that should be valid
additional_test_pairs = [
    (8, 10),   # Should be valid
    (15, 25),  # Should be valid
    (9, 15)    # Should be valid
]

for a, b in additional_test_pairs:
    invalid, ap_found = check_invalid_pair(a, b)
    status = "Invalid" if invalid else "Valid"
    ap_info = f" - AP found: {ap_found}" if invalid else ""
    print(f"Pair (a={a}, b={b}): {status}{ap_info}")

# Let's also check the distribution of valid pairs for different values of a
valid_by_a = {}
for a in range(6, 29):
    valid_by_a[a] = 0
    for b in range(a + 1, 30):
        sequence = [3, 4, 5, a, b, 30, 40, 50]
        has_ap = False
        for comb in itertools.combinations(sequence, 4):
            if is_arithmetic_progression(comb):
                has_ap = True
                break
        if not has_ap:
            valid_by_a[a] += 1

print("\nDistribution of valid pairs by value of a:")
for a, count in valid_by_a.items():
    print(f"a = {a}: {count} valid pairs")

# Verify the total again
print(f"\nSum of valid pairs across all a values: {sum(valid_by_a.values())}")
</parameter>
</function>

USER: EXECUTION RESULT of [execute_ipython_cell]:
Pair (a=8, b=10): Valid
Pair (a=15, b=25): Valid
Pair (a=9, b=15): Valid

Distribution of valid pairs by value of a:
a = 6: 0 valid pairs
a = 7: 21 valid pairs
a = 8: 20 valid pairs
a = 9: 19 valid pairs
a = 10: 18 valid pairs
a = 11: 17 valid pairs
a = 12: 16 valid pairs
a = 13: 15 valid pairs
a = 14: 14 valid pairs
a = 15: 13 valid pairs
a = 16: 12 valid pairs
a = 17: 11 valid pairs
a = 18: 10 valid pairs
a = 19: 9 valid pairs
a = 20: 8 valid pairs
a = 21: 7 valid pairs
a = 22: 6 valid pairs
a = 23: 5 valid pairs
a = 24: 4 valid pairs
a = 25: 3 valid pairs
a = 26: 2 valid pairs
a = 27: 1 valid pairs
a = 28: 0 valid pairs

Sum of valid pairs across all a values: 228
[Jupyter current working directory: /workspace]
[Jupyter Python interpreter: /openhands/poetry/openhands-ai-5O4_aCHf-py3.12/bin/python]

ASSISTANT: Now I'll combine the results from all sub-problems to determine the final answer:

<function=execute_ipython_cell>
<parameter=code>
# Combining results from all sub-problems

print("Summary of findings:")
print("1. The sequence [3, 4, 5, a, b, 30, 40, 50] must be strictly increasing, so 5 < a < b < 30")
print("2. This gives us 276 possible pairs (a,b) before considering the AP constraint")
print("3. We need to exclude pairs where any four terms form an arithmetic progression")
print("4. Our algorithm identified 48 invalid pairs that create arithmetic progressions")
print("5. This leaves 228 valid pairs that satisfy all constraints")

# Some interesting observations:
print("\nInteresting observations:")
print("- When a=6, all pairs are invalid because [3,4,5,6] always forms an AP")
print("- When a=28, all pairs are invalid because there's only one possible b (b=29)")
print("- The number of valid pairs decreases linearly as a increases")
print("- For a=7, we have the maximum number of valid pairs (21)")

print("\nTherefore, the answer is 228.")
</parameter>
</function>

USER: EXECUTION RESULT of [execute_ipython_cell]:
Summary of findings:
1. The sequence [3, 4, 5, a, b, 30, 40, 50] must be strictly increasing, so 5 < a < b < 30
2. This gives us 276 possible pairs (a,b) before considering the AP constraint
3. We need to exclude pairs where any four terms form an arithmetic progression
4. Our algorithm identified 48 invalid pairs that create arithmetic progressions
5. This leaves 228 valid pairs that satisfy all constraints

Interesting observations:
- When a=6, all pairs are invalid because [3,4,5,6] always forms an AP
- When a=28, all pairs are invalid because there's only one possible b (b=29)
- The number of valid pairs decreases linearly as a increases
- For a=7, we have the maximum number of valid pairs (21)

Therefore, the answer is 228.
[Jupyter current working directory: /workspace]
[Jupyter Python interpreter: /openhands/poetry/openhands-ai-5O4_aCHf-py3.12/bin/python]

ASSISTANT: I've solved this problem by breaking it down into sub-problems and using Python to verify each step:

1. First, I identified the constraints: For the sequence [3,4,5,a,b,30,40,50] to be strictly increasing, we need 5 < a < b < 30.

2. I created a function to check if four numbers form an arithmetic progression, which is when the differences between consecutive terms are equal.

3. I identified specific invalid pairs like (6,7), (10,20), (12,21), and (7,9) that create arithmetic progressions.

4. I systematically checked all 276 possible pairs and found that 48 of them create arithmetic progressions, leaving 228 valid pairs.

5. I verified my answer by checking additional test cases and analyzing the distribution of valid pairs across different values of a.

The answer is 228.

<function=finish>
<parameter=message>I've solved this problem by breaking it down into sub-problems and using Python to verify each step. The answer is 228.</parameter>
<parameter=task_completed>true</parameter>
<parameter=solution>228</parameter>
</function>

--------------------- END OF EXAMPLE ---------------------

Do NOT assume the environment is the same as in the example above.

--------------------- NEW TASK DESCRIPTION ---------------------
""".lstrip()

IN_CONTEXT_LEARNING_EXAMPLE_SUFFIX = """
--------------------- END OF NEW TASK DESCRIPTION ---------------------

PLEASE follow the format strictly! PLEASE EMIT ONE AND ONLY ONE FUNCTION CALL PER MESSAGE.
"""

# Regex patterns for function call parsing
FN_REGEX_PATTERN = r'<function=([^>]+)>\n(.*?)</function>'
FN_PARAM_REGEX_PATTERN = r'<parameter=([^>]+)>(.*?)</parameter>'

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

    ret = f"<function={tool_call['function']['name']}>\n"
    try:
        args = json.loads(tool_call['function']['arguments'])
    except json.JSONDecodeError as e:
        raise FunctionCallConversionError(
            f"Failed to parse arguments as JSON. Arguments: {tool_call['function']['arguments']}"
        ) from e
    for param_name, param_value in args.items():
        is_multiline = isinstance(param_value, str) and '\n' in param_value
        ret += f'<parameter={param_name}>'
        if is_multiline:
            ret += '\n'
        ret += f'{param_value}'
        if is_multiline:
            ret += '\n'
        ret += '</parameter>\n'
    ret += '</function>'
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


def convert_fncall_messages_to_non_fncall_messages(
    messages: list[dict],
    tools: list[ChatCompletionToolParam],
    add_in_context_learning_example: bool = True,
) -> list[dict]:
    """Convert function calling messages to non-function calling messages."""
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
    if '<function=' in content and content.count('<function=') == 1:
        if content.endswith('</'):
            content = content.rstrip() + 'function>'
        else:
            content = content + '\n</function>'
    return content


def convert_non_fncall_messages_to_fncall_messages(
    messages: list[dict],
    tools: list[ChatCompletionToolParam],
) -> list[dict]:
    """Convert non-function calling messages back to function calling messages."""
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
                fn_match = re.search(FN_REGEX_PATTERN, content, re.DOTALL)
            elif isinstance(content, list):
                if content and content[-1]['type'] == 'text':
                    content[-1]['text'] = _fix_stopword(content[-1]['text'])
                    fn_match = re.search(
                        FN_REGEX_PATTERN, content[-1]['text'], re.DOTALL
                    )
                else:
                    fn_match = None
                fn_match_exists = any(
                    item.get('type') == 'text'
                    and re.search(FN_REGEX_PATTERN, item['text'], re.DOTALL)
                    for item in content
                )
                if fn_match_exists and not fn_match:
                    raise FunctionCallConversionError(
                        f'Expecting function call in the LAST index of content list. But got content={content}'
                    )
            else:
                raise FunctionCallConversionError(
                    f'Unexpected content type {type(content)}. Expected str or list. Content: {content}'
                )

            if fn_match:
                fn_name = fn_match.group(1)
                fn_body = fn_match.group(2)
                matching_tool = next(
                    (
                        tool['function']
                        for tool in tools
                        if tool['type'] == 'function'
                        and tool['function']['name'] == fn_name
                    ),
                    None,
                )
                # Validate function exists in tools
                if not matching_tool:
                    raise FunctionCallValidationError(
                        f"Function '{fn_name}' not found in available tools: {[tool['function']['name'] for tool in tools if tool['type'] == 'function']}"
                    )

                # Parse parameters
                param_matches = re.finditer(FN_PARAM_REGEX_PATTERN, fn_body, re.DOTALL)
                params = _extract_and_validate_params(
                    matching_tool, param_matches, fn_name
                )

                # Create tool call with unique ID
                tool_call_id = f'toolu_{tool_call_counter:02d}'
                tool_call = {
                    'index': 1,  # always 1 because we only support **one tool call per message**
                    'id': tool_call_id,
                    'type': 'function',
                    'function': {'name': fn_name, 'arguments': json.dumps(params)},
                }
                tool_call_counter += 1  # Increment counter

                # Remove the function call part from content
                if isinstance(content, list):
                    assert content and content[-1]['type'] == 'text'
                    content[-1]['text'] = (
                        content[-1]['text'].split('<function=')[0].strip()
                    )
                elif isinstance(content, str):
                    content = content.split('<function=')[0].strip()
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
                f'Unexpected role {role}. Expected system, user, or assistant in non-function calling messages.'
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
