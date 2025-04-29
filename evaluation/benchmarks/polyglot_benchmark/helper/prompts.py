"""Prompts used in the polyglot benchmark."""

INSTRUCTIONS_ADDENDUM = """
I've provided the following files that need to be modified:
{file_list}

Please help me implement the necessary changes to meet the requirements.
You should ONLY modify these files, and NOT create any new files.
"""

TEST_FAILURES = """
The tests failed. Please fix the issues and try again.
Remember to only modify the following files:
{file_list}
"""

# Dictionary mapping agent class names to their specific instruction suffixes
INST_SUFFIXES = {
    'CodeActAgent': (
        'REMEMBER: All edits must be made directly in the files. Do NOT send'
        ' the edited file as output to the user.\n'
    )
}

# Dictionary mapping agent class names to their fake response functions
FAKE_RESPONSES = {
    'CodeActAgent': lambda _: None,  # Will be replaced with codeact_user_response from shared.py
}