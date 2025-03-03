from evaluation.utils.shared import codeact_user_response

INSTRUCTIONS_ADDENDUM = """
Please solve this problem by reasoning through each step and immediately verifying with Python code.

PROBLEM-SOLVING APPROACH:
1. INSTALL: Start by installing necessary libraries: `%pip install sympy numpy scipy matplotlib`
2. REASON & VERIFY: For each step in your reasoning:
   - First, briefly explain your approach
   - Immediately write Python code to verify your thinking
   - Let the code execution results guide your next step
3. ITERATE: Refine your approach based on code execution results
4. CONFIRM: Verify your final answer with code before submitting

IMPORTANT GUIDELINES:
- Verify EVERY step of your reasoning with Python code - don't rely on mental calculations
- Use powerful libraries like sympy, numpy, and scipy to handle the mathematical heavy lifting
- Be extremely careful with floating-point calculations and rounding errors:
  * Use the Fraction class or sympy.Rational for exact arithmetic when possible
  * Avoid floating-point comparisons for equality
  * When using floats, check results with sufficient precision
- Write code early and often - don't wait until you've fully solved the problem
- Use print statements liberally to see intermediate results
- If code execution contradicts your reasoning, trust the code and adjust your approach
- If your code produces errors, fix them immediately before proceeding
- When you have the final answer, put it in a \\boxed{} notation AND use the finish tool with your solution as the parameter

EXAMPLE STRUCTURE:
```
Step 1: Initial approach
[Brief explanation of your first step]
[Python code to verify this step]

Step 2: Refining the approach
[Brief explanation based on previous results]
[Python code to implement and verify this step]

Step 3: Final solution
[Brief explanation of your solution]
[Python code to verify the final answer]

The final answer is \\boxed{42}
```

Remember: Verify each step with code as you go. Don't trust your reasoning without code verification.
When you have the final answer, put it in a \\boxed{} notation AND use the finish tool with your solution as the parameter. You'll be asked to run a final verification before your solution is accepted.
"""


def math500_user_response(state, **kwargs):
    """Custom response function for MATH-500 benchmark."""
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
    
    # Also check for "The answer is" or "boxed{" in the last message (for backward compatibility)
    last_message = next(
        (
            event.message
            for event in reversed(state.history)
            if hasattr(event, 'message') and event.message
        ),
        None,
    )

    if last_message and ('boxed{' in last_message or '\\boxed{' in last_message or 'The answer is' in last_message):
        # If the agent has provided a solution in text, let it finish
        return '/exit'

    # Check if the agent has used Python code execution in the last few messages
    recent_messages = [
        event.message
        for event in reversed(state.history[: len(state.history)])
        if hasattr(event, 'message') and event.message
    ][:3]  # Look at the last 3 messages

    has_used_python = any(
        'execute_ipython_cell' in msg or 'EXECUTION RESULT' in msg
        for msg in recent_messages
        if msg
    )

    # Check if there was a ModuleNotFoundError in recent messages
    module_error = any(
        'ModuleNotFoundError' in msg or 'No module named' in msg
        for msg in recent_messages
        if msg
    )

    # Check if the agent is verifying with code
    has_verified_with_code = any(
        (
            'execute_ipython_cell' in msg
            or 'EXECUTION RESULT' in msg
        )
        for msg in recent_messages
        if msg
    )

    if module_error:
        # If there was a module error, prompt to install the missing library
        return 'It looks like you need to install some Python libraries. Use %pip install to install the libraries you need (e.g., %pip install sympy numpy scipy matplotlib).'
    elif not has_verified_with_code and len(recent_messages) >= 1:
        # If the agent hasn't verified with code, strongly encourage it
        return 'Please verify your reasoning with Python code. Write code to check each step of your thinking - don\'t rely on mental calculations. Install libraries and write verification code for the steps you\'ve already taken.'
    elif not has_used_python and recent_messages:
        # If the agent hasn't used Python in recent messages, strongly encourage it
        return "You need to verify each step with Python code. Don't proceed with your reasoning until you've confirmed your current step with code execution. Use sympy and numpy to verify your mathematical reasoning."
    elif any(('float' in msg or 'decimal' in msg or '0.' in msg) for msg in recent_messages if msg):
        # If the agent is using floating-point calculations, remind about rounding errors
        return "Be careful with floating-point calculations and rounding errors. Use the Fraction class or sympy.Rational for exact arithmetic when possible. Avoid floating-point comparisons for equality, and when using floats, check results with sufficient precision."

    # Otherwise, use the standard CodeActAgent response
    return codeact_user_response(state)


FAKE_RESPONSES = {
    'CodeActAgent': math500_user_response,
}

INST_SUFFIXES: dict[str, str] = {
    'CodeActAgent': (
        'IMPORTANT: Verify EVERY step of your reasoning with Python code as you go. '
        'First, install necessary libraries: %pip install sympy numpy scipy matplotlib '
        'For each step in your solution process: '
        '1. Briefly explain your approach for that step '
        '2. IMMEDIATELY write Python code to verify your thinking '
        '3. Use the code execution results to guide your next step '
        'Use mathematical libraries like sympy and numpy to verify calculations. '
        'Be extremely careful with floating-point calculations and rounding errors: '
        '- Use the Fraction class or sympy.Rational for exact arithmetic '
        '- Avoid floating-point comparisons for equality '
        '- When using floats, check results with sufficient precision '
        'Do not proceed to the next step until you\'ve verified your current step with code. '
        'If code execution contradicts your reasoning, trust the code and adjust your approach. '
        'When you have the final answer (verified with code), put it in a \\boxed{} notation AND use the "finish" tool with your solution as the parameter.\n'
        'You\'ll be asked to run a final verification before your solution is accepted.\n'
        'For example: The final answer is \\boxed{42} and then finish(solution="42")\n'
        'Remember: Don\'t trust your reasoning without code verification!\n'
    )
}
