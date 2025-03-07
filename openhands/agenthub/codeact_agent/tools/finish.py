from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

_FINISH_DESCRIPTION = """Signals the completion of the current task or conversation.

Use this tool when:
- You have successfully completed the user's requested task
- You cannot proceed further due to technical limitations or missing information

The message should include:
- A clear summary of actions taken and their results
- Any next steps for the user
- Explanation if you're unable to complete the task
- Any follow-up questions if more information is needed

The task_completed field should be set to True if you believed you have completed the task, and False otherwise.

IMPORTANT: 
1. You MUST use Python (execute_ipython_cell) at least once before using this tool. If you haven't used Python yet, you will not be allowed to finish.
2. For benchmark problems (like MATH-500 or AIME), you MUST use the solution parameter to provide your final answer. The solution parameter should contain ONLY the answer value without any explanatory text.

Examples of correct solution parameter usage:
- If your answer is 125: set solution="125"
- If your answer is a fraction: set solution="3/4"
- If your answer is a mathematical expression: set solution="x^2+2x" or use LaTeX format
"""

FinishTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='finish',
        description=_FINISH_DESCRIPTION,
        parameters={
            'type': 'object',
            'required': ['message', 'task_completed'],
            'properties': {
                'message': {
                    'type': 'string',
                    'description': 'Final message to send to the user',
                },
                'task_completed': {
                    'type': 'string',
                    'enum': ['true', 'false', 'partial'],
                    'description': 'Whether you have completed the task.',
                },
                'solution': {
                    'type': 'string',
                    'description': 'REQUIRED for benchmark problems (MATH-500, AIME, etc.). Provide ONLY your final answer as a concise value (e.g., "125", "3/4", "x^2+2x"). Do NOT include explanations or working in this field.',
                },
            },
        },
    ),
)
