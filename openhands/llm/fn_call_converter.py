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
<code>
This is the code or content for the tool
that can span
multiple lines
</code>
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
Here's a running example of how to break down a complex problem into sub-problems and solve each one using tools.

--------------------- START OF EXAMPLE ---------------------

USER: Problem: Find the number of ordered pairs of integers $(a, b)$ such that the sequence $[3, 4, 5, a, b, 30, 40, 50]$ is strictly increasing and no set of four (not necessarily consecutive) terms forms an arithmetic progression.

ASSISTANT: I'll solve this problem by breaking it down into sub-problems and using Python to verify each step.

First, I need to install the necessary libraries:

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
[Jupyter current working directory: /workspace]
[Jupyter Python interpreter: /openhands/poetry/openhands-ai-5O4_aCHf-py3.12/bin/python]

ASSISTANT: Now I'll break this problem into sub-problems:

Sub-problem 1: Understand the constraints on a and b for the sequence to be strictly increasing.
Sub-problem 2: Define a function to check if four numbers form an arithmetic progression.
Sub-problem 3: Identify specific invalid pairs (a,b) that create arithmetic progressions.
Sub-problem 4: Count all valid pairs by checking each possible (a,b) combination.

Let me start with sub-problem 1:

<tool=execute_ipython_cell>
<code>
import numpy as np

# For the sequence [3, 4, 5, a, b, 30, 40, 50] to be strictly increasing:
# 5 < a < b < 30

# Calculate possible values for a and b
a_min = 6  # a must be greater than 5
a_max = 28  # a must be less than b, which must be less than 30
b_min_formula = "a + 1"  # b must be greater than a
b_max = 29  # b must be less than 30

# Check if θ is in the range [0, 2π)
if theta < 0:
    theta += 2 * np.pi
    print(f"Adjusted θ to be in [0, 2π) range: {theta} radians")
else:
    print(f"θ is already in [0, 2π) range: {theta} radians")
</code>
</tool>

USER: EXECUTION RESULT of [execute_ipython_cell]:
Constraints: 5 < a < b < 30
Possible values for a: 6 to 28
For each a, possible values for b: a + 1 to 29
Total possible pairs before checking AP constraint: 276
[Jupyter current working directory: /workspace]
[Jupyter Python interpreter: /openhands/poetry/openhands-ai-5O4_aCHf-py3.12/bin/python]

ASSISTANT: Now I'll tackle sub-problem 2 - defining a function to check if four numbers form an arithmetic progression:

<tool=execute_ipython_cell>
<code>
import numpy as np

def is_arithmetic_progression(numbers):
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

    # Check for numerical precision issues
    if abs(x_back) < 1e-10:
        print("x is effectively zero (within numerical precision)")
</code>
</tool>

USER: EXECUTION RESULT of [execute_ipython_cell]:
[3, 4, 5, 6]: is AP (Expected: AP)
[3, 5, 7, 9]: is AP (Expected: AP)
[10, 20, 30, 40]: is AP (Expected: AP)
[3, 4, 6, 9]: not AP (Expected: not AP)
[3, 5, 8, 10]: not AP (Expected: not AP)
[Jupyter current working directory: /workspace]
[Jupyter Python interpreter: /openhands/poetry/openhands-ai-5O4_aCHf-py3.12/bin/python]

ASSISTANT: Now for sub-problem 3, I'll identify some specific invalid pairs (a,b) that create arithmetic progressions:

<tool=execute_ipython_cell>
<code>
import numpy as np

def check_invalid_pair(a, b):
    sequence = [3, 4, 5, a, b, 30, 40, 50]

    # Check all possible 4-element combinations
    for comb in itertools.combinations(sequence, 4):
        if is_arithmetic_progression(comb):
            return True, comb

    return False, None

# Express in terms of π for mathematical notation
print(f"θ = {theta/pi}π radians")
</code>
</tool>

USER: EXECUTION RESULT of [execute_ipython_cell]:
Pair (a=6, b=7): Invalid - AP found: (3, 4, 5, 6)
Pair (a=10, b=20): Invalid - AP found: (10, 20, 30, 40)
Pair (a=12, b=21): Invalid - AP found: (3, 12, 21, 30)
Pair (a=7, b=9): Invalid - AP found: (3, 5, 7, 9)
Pair (a=8, b=9): Valid
[Jupyter current working directory: /workspace]
[Jupyter Python interpreter: /openhands/poetry/openhands-ai-5O4_aCHf-py3.12/bin/python]

ASSISTANT: Now I'll solve sub-problem 4 by counting all valid pairs. Since there are 276 possible pairs to check, I'll implement an efficient algorithm:

<tool=execute_ipython_cell>
<code>
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
</code>
</tool>

USER: EXECUTION RESULT of [execute_ipython_cell]:
Total possible pairs: 276
Invalid pairs (contain an AP): 48
Valid pairs (no AP): 228
Verification: Total matches earlier calculation: True
[Jupyter current working directory: /workspace]
[Jupyter Python interpreter: /openhands/poetry/openhands-ai-5O4_aCHf-py3.12/bin/python]

ASSISTANT: Let me verify the result by checking a few more specific cases to make sure our algorithm is correct:

<tool=execute_ipython_cell>
<code>
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
</code>
</tool>

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

<tool=execute_ipython_cell>
<code>
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
</code>
</tool>

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

<tool=finish>
<solution>\\boxed{(3, \\frac{\\pi}{2})}</solution>
</tool>

--------------------- END OF EXAMPLE ---------------------

Do NOT assume the environment is the same as in the example above.

--------------------- NEW TASK DESCRIPTION ---------------------
""".lstrip()

IN_CONTEXT_LEARNING_EXAMPLE_SUFFIX = """
--------------------- END OF TASK DESCRIPTION ---------------------

I'll solve this step-by-step using the available tools.
"""

# Regex patterns for extracting function calls
FN_CALL_REGEX_PATTERN = r'<tool=([^>]+)>(.*?)</tool>'
FN_PARAM_REGEX_PATTERN = r'<(?!tool=)([^>]+)>(.*?)</\1>'


def _extract_and_validate_params(
    matching_tool: dict, param_matches: Iterable, tool_name: str
) -> dict:
    """Extract and validate parameters from a function call."""
    params = {}
    required_params = [
        param['name']
        for param in matching_tool['function']['parameters']['properties'].values()
        if param.get('required', False)
    ]
    for match in param_matches:
        param_name = match.group(1)
        param_value = match.group(2).strip()
        params[param_name] = param_value

    # Check for missing required parameters
    missing_params = [param for param in required_params if param not in params]
    if missing_params:
        raise FunctionCallValidationError(
            f"Missing required parameters for tool '{tool_name}': {missing_params}"
        )

    return params


def convert_to_tool_calling_messages(
    messages: list[dict], tools: list[ChatCompletionToolParam]
) -> list[dict]:
    """Convert non-tool calling messages to tool calling messages.

    This is used when the model doesn't support tool calling, but we want to
    use it with a tool calling agent.
    """
    # TODO: implement this
    return messages


def convert_from_tool_calling_messages(
    messages: list[dict], tools: list[ChatCompletionToolParam]
) -> list[dict]:
    """Convert tool calling messages to non-tool calling messages.

    This is used when the model supports tool calling, but we want to
    use it with a non-tool calling agent.
    """
    converted_messages = []
    tool_call_counter = 0

    for message in messages:
        role = message['role']
        content = message.get('content', '')

        if role == 'system':
            # Add tool descriptions to system message
            if tools:
                tool_descriptions = []
                for tool in tools:
                    if tool['type'] == 'function':
                        fn = tool['function']
                        tool_descriptions.append(
                            f"Tool: {fn['name']}\nDescription: {fn['description']}\n"
                        )
                tool_description_str = '\n'.join(tool_descriptions)
                if content:
                    content += '\n\n' + SYSTEM_PROMPT_SUFFIX_TEMPLATE.format(
                        description=tool_description_str
                    )
                else:
                    content = SYSTEM_PROMPT_SUFFIX_TEMPLATE.format(
                        description=tool_description_str
                    )

            converted_messages.append({'role': 'system', 'content': content})

        elif role == 'user':
            converted_messages.append({'role': 'user', 'content': content})

        elif role == 'assistant':
            # Check if this is a tool call
            if 'tool_calls' in message and message['tool_calls']:
                # Only handle the first tool call for now
                tool_call = message['tool_calls'][0]
                if tool_call['type'] == 'function':
                    fn_name = tool_call['function']['name']
                    fn_args = json.loads(tool_call['function']['arguments'])
                    # Format as a tool call
                    tool_call_str = f"<tool={fn_name}>\n"
                    for arg_name, arg_value in fn_args.items():
                        tool_call_str += f"<{arg_name}>{arg_value}</{arg_name}>\n"
                    tool_call_str += "</tool>"

                    # Combine with content
                    if content:
                        content = f"{content}\n\n{tool_call_str}"
                    else:
                        content = tool_call_str

                converted_messages.append({'role': 'assistant', 'content': content})
            else:
                converted_messages.append({'role': 'assistant', 'content': content})

        elif role == 'tool':
            # Format as a user message with execution result
            tool_call_id = message['tool_call_id']
            content = message['content']
            # Find the corresponding tool call
            for i, msg in enumerate(converted_messages):
                if (
                    msg['role'] == 'assistant'
                    and 'tool_calls' in messages[i]
                    and messages[i]['tool_calls']
                    and any(tc['id'] == tool_call_id for tc in messages[i]['tool_calls'])
                ):
                    # Found the tool call
                    tool_call = next(
                        tc
                        for tc in messages[i]['tool_calls']
                        if tc['id'] == tool_call_id
                    )
                    fn_name = tool_call['function']['name']
                    break
            else:
                fn_name = "unknown_tool"

            user_content = f"EXECUTION RESULT of [{fn_name}]:\n{content}"
            converted_messages.append({'role': 'user', 'content': user_content})

        else:
            raise FunctionCallConversionError(
                f'Unexpected role {role}. Expected system, user, assistant, or tool.'
            )

    return converted_messages


def extract_tool_calls_from_content(
    content: str | list, tools: list[ChatCompletionToolParam]
) -> tuple[str | list, list[dict]]:
    """Extract tool calls from content.

    Args:
        content: The content to extract tool calls from.
        tools: The available tools.

    Returns:
        A tuple of (content without tool calls, list of tool calls).
    """
    if isinstance(content, list):
        # Handle content as a list of parts
        text_parts = []
        for part in content:
            if part['type'] == 'text':
                text_parts.append(part['text'])
        content_str = '\n'.join(text_parts)
    else:
        content_str = content

    # Extract tool calls
    tool_calls = []
    matches = re.finditer(FN_CALL_REGEX_PATTERN, content_str, re.DOTALL)
    for match in matches:
        tool_name = match.group(1)
        tool_body = match.group(2)

        # Find the matching tool
        matching_tool = next(
            (
                tool
                for tool in tools
                if tool['type'] == 'function'
                and tool['function']['name'] == tool_name
            ),
            None,
        )
        if not matching_tool:
            raise FunctionCallValidationError(
                f"Tool '{tool_name}' not found in available tools: {[tool['function']['name'] for tool in tools if tool['type'] == 'function']}"
            )

        # Parse parameters
        param_matches = re.finditer(FN_PARAM_REGEX_PATTERN, tool_body, re.DOTALL)
        params = _extract_and_validate_params(matching_tool, param_matches, tool_name)

        # Create tool call
        tool_call = {
            'id': f'call_{len(tool_calls)}',
            'type': 'function',
            'function': {
                'name': tool_name,
                'arguments': json.dumps(params),
            },
        }
        tool_calls.append(tool_call)

    # Remove tool calls from content
    if tool_calls:
        if isinstance(content, list):
            # Handle content as a list of parts
            new_content = copy.deepcopy(content)
            for i, part in enumerate(new_content):
                if part['type'] == 'text':
                    # Remove all tool calls from text
                    part['text'] = re.sub(
                        FN_CALL_REGEX_PATTERN, '', part['text'], flags=re.DOTALL
                    ).strip()
            return new_content, tool_calls
        else:
            # Handle content as a string
            new_content = re.sub(
                FN_CALL_REGEX_PATTERN, '', content_str, flags=re.DOTALL
            ).strip()
            return new_content, tool_calls
    else:
        return content, []


def convert_from_text_to_tool_calling_messages(
    messages: list[dict], tools: list[ChatCompletionToolParam]
) -> list[dict]:
    """Convert text messages to tool calling messages.

    This is used when the model doesn't support tool calling, but we want to
    extract tool calls from the text.
    """
    converted_messages = []
    tool_call_counter = 0

    for message in messages:
        role, content = message['role'], message.get('content', '')

        if role == 'system' or role == 'user':
            # Keep system and user messages as is
            converted_messages.append(message)
        elif role == 'assistant':
            # Check if there's a function call in the content
            matches = list(re.finditer(FN_CALL_REGEX_PATTERN, content, re.DOTALL))
            if matches:
                # Extract the tool call
                tool_match = matches[0]  # Only handle the first tool call for now
                tool_name = tool_match.group(1)
                tool_body = tool_match.group(2)

                # Find the matching tool
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

# Aliases for backward compatibility
convert_tool_messages_to_non_tool_messages = convert_from_tool_calling_messages
convert_non_tool_messages_to_tool_messages = convert_to_tool_calling_messages