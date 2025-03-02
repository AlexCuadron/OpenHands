from evaluation.utils.shared import codeact_user_response

INSTRUCTIONS_ADDENDUM = """
Please solve this problem by breaking it down into sub-problems and using tools to verify each step.

PROBLEM-SOLVING APPROACH:
1. ANALYZE: First, carefully analyze the problem and identify 2-4 distinct sub-problems or steps needed to reach the solution
2. PLAN: For each sub-problem, plan how you'll use Python tools to solve it
3. EXECUTE: Solve each sub-problem separately, using Python to verify your work
4. COMBINE: Combine the results from all sub-problems to find the final answer

IMPORTANT GUIDELINES:
- Start by installing any libraries you need: `%pip install sympy numpy scipy matplotlib`
- For EACH sub-problem:
  * State the sub-problem clearly
  * Use Python code to solve it
  * Verify the result
  * Explain what you learned
- If code execution reveals errors in your reasoning, acknowledge the mistake and correct your approach
- Use tools to discover information that might contradict your initial assumptions
- AIME problems typically have integer answers, so make sure your final answer is an integer
- When you have the final answer, provide it in the format: "The answer is [your answer]"

EXAMPLE STRUCTURE:
```
Sub-problem 1: [Description]
[Python code to solve sub-problem 1]
Result: [What you learned]

Sub-problem 2: [Description]
[Python code to solve sub-problem 2]
Result: [What you learned]

...

Combining results:
[Python code to combine results]
Final answer: [Answer]
```

For example, if the answer is 42, you can write: "The answer is 42".
"""

def aime2024_user_response(state, **kwargs):
    """Custom response function for AIME2024 benchmark."""
    # First check if the agent has already provided a solution
    last_message = next(
        (event.message for event in reversed(state.history) 
         if hasattr(event, 'message') and event.message),
        None
    )
    
    if last_message and ('The answer is' in last_message):
        # If the agent has provided a solution, let it finish
        return '/exit'
    
    # Check if there was a ModuleNotFoundError in recent messages
    recent_messages = [
        event.message for event in reversed(state.history[:len(state.history)])
        if hasattr(event, 'message') and event.message
    ][:3]  # Look at the last 3 messages
    
    module_error = any(
        'ModuleNotFoundError' in msg or 'No module named' in msg
        for msg in recent_messages if msg
    )
    
    has_used_python = any(
        'execute_ipython_cell' in msg or 'EXECUTION RESULT' in msg
        for msg in recent_messages if msg
    )
    
    # Check if the agent is breaking down the problem into sub-problems
    has_sub_problems = any(
        ('Sub-problem' in msg or 'Subproblem' in msg or 'Step ' in msg or 'sub-problem' in msg)
        for msg in recent_messages if msg
    )
    
    if module_error:
        # If there was a module error, prompt to install the missing library
        return "It looks like you need to install some Python libraries. Use %pip install to install the libraries you need (e.g., %pip install sympy numpy scipy matplotlib)."
    elif not has_sub_problems and len(recent_messages) >= 1:
        # If the agent isn't breaking down the problem, encourage it to do so
        return "Please break down this problem into smaller sub-problems. For each sub-problem: (1) State it clearly, (2) Write Python code to solve it, (3) Verify the result, (4) Explain what you learned."
    elif not has_used_python and recent_messages:
        # If the agent hasn't used Python in recent messages, encourage it to do so
        return "Please use Python tools to verify your reasoning for each sub-problem. Don't rely solely on your own thinking - use tools to discover information that might contradict your initial assumptions."
    
    # Otherwise, use the standard CodeActAgent response
    return codeact_user_response(state)

FAKE_RESPONSES = {
    'CodeActAgent': aime2024_user_response,
}

INST_SUFFIXES: dict[str, str] = {
    'CodeActAgent': (
        'IMPORTANT: Break down this problem into 2-4 distinct sub-problems and solve each one separately using Python tools. '
        'For each sub-problem: (1) State it clearly, (2) Write Python code to solve it, (3) Verify the result, (4) Explain what you learned. '
        'First, install any libraries you need using %pip install (e.g., %pip install sympy numpy scipy matplotlib). '
        'Do not trust your own reasoning without verification through tool execution. '
        'If tool execution reveals errors in your thinking, acknowledge them and correct your approach. '
        'After solving all sub-problems, combine the results with Python code to find the final answer. '
        'When you have the final answer (verified with tools), use the "finish" tool with your solution as the parameter.\n'
        'For example: finish(solution="42")\n'
    )
}