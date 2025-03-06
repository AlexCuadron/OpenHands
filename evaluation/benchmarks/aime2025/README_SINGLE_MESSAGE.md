# AIME2025 Benchmark - Single Message Mode

This document describes the modifications made to the AIME2025 benchmark to ensure all instructions are sent in a single user message.

## Problem

The original implementation of the AIME2025 benchmark sent multiple messages to the LLM, which could result in:
1. Multiple "user" roles in the prompt
2. Fragmented instructions that might confuse the LLM
3. Inconsistent behavior across different LLM providers
4. Additional user messages being sent when the agent tries to finish without using Python

## Solution

We've implemented a "Single Message Mode" that ensures all instructions are sent in a single user message. This is achieved through the following components:

1. `single_message_prompt.py`: Provides a modified version of the `run_controller` function that ensures all instructions are sent in a single user message.
2. `agent_controller_patch.py`: Patches the `AgentController` class to ensure it properly handles our single message approach.
3. Updates to `run_infer.py`: Modified to use our patched components.

## Usage

No changes are needed to the existing scripts. The patch is automatically applied when running the AIME2025 benchmark.

## Testing

You can test the single message mode by running:

```bash
python test_single_message.py
```

This will run the AIME2025 benchmark with a single instance and verify that the single message mode is working correctly.

## Implementation Details

The key changes are:

1. We patch the `AgentController.process_event` method to ensure that only one user message is in the history at any time.
2. We patch the `AgentController._get_fake_user_response` method to prevent additional user messages from being sent.
3. We patch the `AgentController.run_agent_loop` method to handle the Python reminder flag.
4. We use a custom `run_controller_single_message` function that ensures all instructions are sent in a single user message.
5. We apply these patches before running the controller.

### Handling Additional User Messages

The enhanced implementation now handles several scenarios where additional user messages might be sent:

1. **Python Reminders**: When the agent hasn't used Python yet, instead of sending a reminder message, we set a flag that will be checked in the run_agent_loop patch.
2. **Finish Without Python**: When the agent tries to finish without using Python, we intercept this and prevent it from finishing.
3. **Follow-up Messages**: Any follow-up messages from the user are intercepted and not processed, to maintain the single message approach.

These changes ensure that the LLM receives a single, coherent set of instructions in one user message, which should improve the consistency and quality of the responses.
