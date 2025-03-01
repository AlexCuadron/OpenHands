from evaluation.utils.shared import codeact_user_response

INSTRUCTIONS_ADDENDUM = """
Please solve this math problem by using Python to verify each step of your reasoning. 

IMPORTANT:
- Use Python code execution to verify your calculations and reasoning at each step
- Do NOT rely solely on your own mathematical reasoning - verify everything with code
- If your code execution reveals errors in your reasoning, acknowledge the mistake and correct your approach
- The following libraries are pre-installed and ready to use:
  * sympy - for symbolic mathematics (already imported as sp)
  * numpy - for numerical computations (already imported as np)
  * scipy - for scientific computing
  * matplotlib - for plotting (plt is already imported)
- Common sympy functions and symbols are pre-imported (symbols, solve, Eq, simplify, etc.)
- Break down complex calculations into smaller parts that can be verified with code
- When you have the final answer, please provide it in the format: "The answer is [your answer]"
- You can also use LaTeX notation with \\boxed{} to highlight your final answer

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
    
    # Check if the agent has used Python code execution in the last few messages
    recent_messages = [
        event.message for event in reversed(state.history[:len(state.history)])
        if hasattr(event, 'message') and event.message
    ][:3]  # Look at the last 3 messages
    
    has_used_python = any(
        'execute_ipython_cell' in msg or 'EXECUTION RESULT' in msg
        for msg in recent_messages if msg
    )
    
    if not has_used_python and recent_messages:
        # If the agent hasn't used Python in recent messages, encourage it to do so
        return "Please use Python code execution to verify your calculations and reasoning. Don't rely solely on your own mathematical reasoning."
    
    # Otherwise, use the standard CodeActAgent response
    return codeact_user_response(state)

FAKE_RESPONSES = {
    'CodeActAgent': math500_user_response,
}

INST_SUFFIXES: dict[str, str] = {
    'CodeActAgent': (
        'IMPORTANT: You MUST use Python code execution to verify your mathematical reasoning at EACH step. '
        'Do not trust your own calculations without verification. '
        'If Python execution reveals errors in your reasoning, acknowledge them and correct your approach. '
        'Remember that sympy, numpy, scipy, and matplotlib are pre-installed with common imports already set up. '
        'When you have the final answer (verified with code), use the "finish" tool with your solution as the parameter.\n'
        'For example: finish(solution="\\boxed{42}")\n'
    )
}