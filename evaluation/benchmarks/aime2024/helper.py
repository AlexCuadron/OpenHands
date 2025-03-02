from evaluation.utils.shared import codeact_user_response

INSTRUCTIONS_ADDENDUM = """
Please solve this problem using a programmatic approach with Python to verify your work.

PROBLEM-SOLVING APPROACH:
1. ANALYZE: First, carefully analyze the problem and understand what's being asked
2. PLAN: Develop a programmatic approach using Python to solve the problem
3. IMPLEMENT: Write Python code to implement your solution
4. VERIFY: Test your solution with examples and edge cases

IMPORTANT GUIDELINES:
- Start by installing any libraries you need: `%pip install sympy numpy scipy matplotlib`
- Use Python's mathematical libraries (sympy, numpy, etc.) to solve the problem efficiently
- Implement your solution step-by-step, explaining your approach
- Verify your solution with test cases or examples
- If code execution reveals errors in your reasoning, acknowledge the mistake and correct your approach
- Use tools to discover information that might contradict your initial assumptions
- AIME problems typically have integer answers, so make sure your final answer is an integer
- When you have the final answer, use the finish tool with your solution as the parameter

EXAMPLE STRUCTURE:
```
Problem Analysis:
[Brief analysis of the problem]

Solution Approach:
[Explanation of your programmatic approach]

Implementation:
[Python code implementing your solution]

Verification:
[Python code testing your solution]

Final answer: [Answer]
```

When you have the final answer, use the finish tool with your solution as the parameter.
"""


def aime2024_user_response(state, **kwargs):
    """Custom response function for AIME2024 benchmark."""
    # First check if the agent has already provided a solution
    # Check if the agent used the finish tool
    finish_action = next(
        (
            event
            for event in reversed(state.history)
            if hasattr(event, 'action') and event.action == 'finish'
        ),
        None,
    )
    
    if finish_action:
        # If the agent has used the finish tool, let it finish
        return '/exit'
    
    # Also check for "The answer is" in the last message (for backward compatibility)
    last_message = next(
        (
            event.message
            for event in reversed(state.history)
            if hasattr(event, 'message') and event.message
        ),
        None,
    )

    if last_message and ('The answer is' in last_message):
        # If the agent has provided a solution in text, let it finish
        return '/exit'

    # Check if there was a ModuleNotFoundError in recent messages
    recent_messages = [
        event.message
        for event in reversed(state.history[: len(state.history)])
        if hasattr(event, 'message') and event.message
    ][:3]  # Look at the last 3 messages

    module_error = any(
        'ModuleNotFoundError' in msg or 'No module named' in msg
        for msg in recent_messages
        if msg
    )

    has_used_python = any(
        'execute_ipython_cell' in msg or 'EXECUTION RESULT' in msg
        for msg in recent_messages
        if msg
    )

    # Check if the agent is using a programmatic approach
    has_programmatic_approach = any(
        (
            'Solution Approach' in msg
            or 'Implementation' in msg
            or 'Verification' in msg
            or 'programmatic' in msg
            or 'algorithm' in msg
        )
        for msg in recent_messages
        if msg
    )

    if module_error:
        # If there was a module error, prompt to install the missing library
        return 'It looks like you need to install some Python libraries. Use %pip install to install the libraries you need (e.g., %pip install sympy numpy scipy matplotlib).'
    elif not has_programmatic_approach and len(recent_messages) >= 1:
        # If the agent isn't using a programmatic approach, encourage it to do so
        return 'Please develop a programmatic approach to solve this problem. Analyze the problem, plan your solution, implement it in Python, and verify your results with test cases.'
    elif not has_used_python and recent_messages:
        # If the agent hasn't used Python in recent messages, encourage it to do so
        return "Please use Python to implement your solution. Mathematical libraries like sympy and numpy can help you solve this problem efficiently. Don't rely solely on your own thinking - use code to verify your approach."

    # Otherwise, use the standard CodeActAgent response
    return codeact_user_response(state)


FAKE_RESPONSES = {
    'CodeActAgent': aime2024_user_response,
}

INST_SUFFIXES: dict[str, str] = {
    'CodeActAgent': (
        'IMPORTANT: Develop a programmatic approach to solve this problem using Python. '
        'First, analyze the problem and understand what is being asked. '
        'Then, plan your solution and implement it step-by-step in Python. '
        'Install any libraries you need using %pip install (e.g., %pip install sympy numpy scipy matplotlib). '
        'Use mathematical libraries like sympy and numpy to solve the problem efficiently. '
        'Verify your solution with test cases or examples. '
        'Do not trust your own reasoning without verification through code execution. '
        'If code execution reveals errors in your thinking, acknowledge them and correct your approach. '
        'When you have the final answer (verified with code), use the "finish" tool with your solution as the parameter.\n'
        'For example: finish(solution="42")\n'
    )
}
