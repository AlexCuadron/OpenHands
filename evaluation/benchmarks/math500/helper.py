from evaluation.utils.shared import codeact_user_response

INSTRUCTIONS_ADDENDUM = """
Please solve this math problem step by step. Show your work and explain your reasoning clearly.
When you have the final answer, please provide it in the format: "The answer is [your answer]".
You can also use LaTeX notation with \\boxed{} to highlight your final answer.

For example, if the answer is 42, you can write: "The answer is \\boxed{42}".
"""

def math500_user_response(state, **kwargs):
    """Custom response function for MATH-500 benchmark."""
    # First check if the agent has already provided a solution
    last_message = next(
        (event.message for event in reversed(state.history) 
         if hasattr(event, 'message') and event.message),
        None
    )
    
    if last_message and ('boxed{' in last_message or 'The answer is' in last_message):
        # If the agent has provided a solution, let it finish
        return '/exit'
    
    # Otherwise, use the standard CodeActAgent response
    return codeact_user_response(state)

FAKE_RESPONSES = {
    'CodeActAgent': math500_user_response,
}

INST_SUFFIXES: dict[str, str] = {
    'CodeActAgent': (
        'IMPORTANT: You should solve this problem step by step. When you have the final answer, '
        'use the "finish" tool with your solution as the parameter.\n'
        'For example: finish(solution="\\boxed{42}")\n'
    )
}