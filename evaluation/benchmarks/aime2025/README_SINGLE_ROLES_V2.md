# AIME2025 Benchmark - Single Role Messages Mode

This document describes the modifications made to the AIME2025 benchmark to ensure:
1. There is only ONE "user" message (including system instructions)
2. There is only ONE "assistant" message that starts empty and grows with prefix=true

## Problem

The original implementation of the AIME2025 benchmark had several issues:
1. Multiple "user" roles in the prompt
2. Multiple "assistant" roles in the prompt
3. "system" role messages that might not be handled consistently across LLM providers
4. Fragmented instructions that might confuse the LLM
5. Inconsistent behavior across different LLM providers

## Solution

We've implemented a "Single Role Messages Mode" that ensures:
1. There is only ONE "user" message that includes both the system instructions and the original user message
2. There is only ONE "assistant" message that starts empty and grows with prefix=true
3. NO "system" role messages are used

This is achieved through the following components:

1. `single_message_prompt.py`: Provides a modified version of the `run_controller` function that ensures all instructions are sent in a single user message and prevents any additional user messages.
2. `agent_controller_patch.py`: Patches the `AgentController` class to ensure it properly handles our single message approach.
3. `single_role_messages.py`: Provides functionality to ensure there is only one user message and one assistant message.
4. `llm_logging_patch.py`: Patches the LLM logging mechanism to ensure that only user and assistant roles are used in the logged messages.
5. `no_system_message_patch.py`: Patches the LLM to ensure that system messages are never used.
6. `single_user_tag_patch.py`: Patches the LLM to ensure that there is always at most ONE user tag.
7. Updates to `run_infer.py`: Modified to use our patched components.

## Usage

No changes are needed to the existing scripts. The patch is automatically applied when running the AIME2025 benchmark.

## Testing

You can test the single role messages mode by running:

```bash
python test_single_message.py
```

This will run the AIME2025 benchmark with a single instance and verify that the single role messages mode is working correctly.

## Implementation Details

### Single Role Messages

1. **User Message**: We create a single user message that includes both the system instructions and the original user message.

2. **Assistant Message**: We create a single assistant message that starts empty and grows with prefix=true.

3. **No System Role**: We don't use the "system" role at all, which ensures that the LLM only sees two roles: "user" and "assistant".

4. **Tool Results Integration**: Tool results (observations) are included in the assistant message, formatted as "TOOL RESULT: [result]".

5. **LLM Patching**: We patch the LLM's completion method to transform the messages before sending them to the LLM.

6. **Logging Patching**: We patch the LLM's logging mechanism to ensure that only user and assistant roles are used in the logged messages.

7. **No System Messages**: We patch the LLM to ensure that system messages are never used at any level, including the internal LLM calls.

8. **Single User Tag**: We patch the LLM to ensure that there is always at most ONE user tag, combining multiple user messages if necessary.

These changes ensure that the LLM receives a single, coherent set of instructions in one user message and maintains a single, growing assistant message, which should improve the consistency and quality of the responses.