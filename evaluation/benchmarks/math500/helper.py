from evaluation.utils.shared import codeact_user_response

INSTRUCTIONS_ADDENDUM = """
Please solve this problem by using tools to verify each step of your reasoning. 

IMPORTANT:
- Use Python code execution to verify your thinking at EACH step
- Do NOT rely solely on your own reasoning - verify everything with tools
- If tool execution reveals errors in your thinking, acknowledge the mistake and correct your approach
- Use tools to discover new information that might not be obvious from initial reasoning
- Break down complex problems into smaller parts that can be verified with tools
- You should first install any libraries you need using %pip install:
  * For mathematical problems, install sympy, numpy, scipy: `%pip install sympy numpy scipy matplotlib`
  * Always verify that imports work before proceeding with your solution
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
    
    # Check if there was a ModuleNotFoundError in recent messages
    module_error = any(
        'ModuleNotFoundError' in msg or 'No module named' in msg
        for msg in recent_messages if msg
    )
    
    if module_error:
        # If there was a module error, prompt to install the missing library
        return "It looks like you need to install some Python libraries. Use %pip install to install the libraries you need (e.g., %pip install sympy numpy scipy matplotlib)."
    elif not has_used_python and recent_messages:
        # If the agent hasn't used Python in recent messages, encourage it to do so
        return "Please use tools to verify your reasoning. Don't rely solely on your own thinking - use tools to discover information that might contradict your initial assumptions."
    
    # Otherwise, use the standard CodeActAgent response
    return codeact_user_response(state)

FAKE_RESPONSES = {
    'CodeActAgent': math500_user_response,
}

INST_SUFFIXES: dict[str, str] = {
    'CodeActAgent': (
        'IMPORTANT: You MUST use tools to verify your reasoning at EACH step. '
        'First, install any libraries you need using %pip install (e.g., %pip install sympy numpy scipy). '
        'Do not trust your own reasoning without verification through tool execution. '
        'If tool execution reveals errors in your thinking, acknowledge them and correct your approach. '
        'The key insight is that by using tools, you will discover information that contradicts your initial reasoning. '
        'When you have the final answer (verified with tools), use the "finish" tool with your solution as the parameter.\n'
        'For example: finish(solution="\\boxed{42}")\n'
    )
}