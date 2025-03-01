"""Helper functions for the MATH-500 benchmark."""

import re

# Instructions for the agent
INSTRUCTIONS_ADDENDUM = """
You are tasked with solving a math problem. You can use the Python interpreter to help you solve the problem.

To use the Python interpreter, you can use the `execute_ipython_cell` tool. This allows you to run Python code to help with calculations, visualizations, or any other computational tasks.

Work through the problem step by step, showing your reasoning and calculations.

When you have the final answer, use the `finish` tool with your answer as a parameter. For example:
```
finish(answer="42")
```

Make sure your answer matches the expected format in the problem statement. Your answer will be extracted from the `answer` parameter and used for evaluation.
"""

# Different instruction suffixes for different agent types
INST_SUFFIXES = {
    'CodeActAgent': (
        "IMPORTANT: When you're confident in your final answer, use the `finish` tool with your answer as a parameter. "
        'For example: finish(answer="42"). This will complete the task and submit your answer for evaluation.\n'
    ),
    # Add other agent types as needed
}


# Fake response function for different agent types
def codeact_fake_response(state, **kwargs):
    """Fake response function for CodeActAgent."""
    from openhands.events.action import AgentFinishAction

    # Check if the last action is a finish action
    last_action = state.history[-1] if state.history else None
    if isinstance(last_action, AgentFinishAction):
        # If the finish action has an answer in the outputs, acknowledge it
        if 'answer' in last_action.outputs:
            answer = last_action.outputs['answer']
            return (
                f'Thank you for your answer: {answer}. The task is now complete. /exit'
            )
        return '/exit'

    return 'Continue solving the problem. Use the Python interpreter if needed, and when you have the final answer, use the finish tool with your answer as a parameter.'


# Dictionary of fake response functions for different agent types
FAKE_RESPONSES = {
    'CodeActAgent': codeact_fake_response,
    # Add other agent types as needed
}


def normalize_answer(answer: str) -> str:
    """Normalize the answer for comparison.

    Args:
        answer: The answer string to normalize

    Returns:
        The normalized answer string
    """
    # Remove whitespace and convert to lowercase
    normalized = re.sub(r'\s+', '', answer).lower()

    # Remove LaTeX formatting
    normalized = re.sub(r'\\boxed{(.*?)}', r'\1', normalized)
    normalized = re.sub(r'\\left', '', normalized)
    normalized = re.sub(r'\\right', '', normalized)

    return normalized


def compare_answers(predicted: str, reference: str) -> bool:
    """Compare the predicted answer with the reference answer.

    Args:
        predicted: The predicted answer
        reference: The reference answer

    Returns:
        True if the answers match, False otherwise
    """
    return normalize_answer(predicted) == normalize_answer(reference)
