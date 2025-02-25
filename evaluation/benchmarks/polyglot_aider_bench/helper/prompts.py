"""Prompts used in the polyglot aider benchmark."""

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