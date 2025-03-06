# AIME2025 Benchmark - Single Message Roles Mode

This document describes the modifications made to the AIME2025 benchmark to ensure:
1. All messages are combined into a single system and user message
2. No additional roles (like "assistant") are used

## Problem

The original implementation of the AIME2025 benchmark had several issues:
1. Multiple "user" roles in the prompt
2. Multiple "assistant" roles in the prompt
3. Fragmented instructions that might confuse the LLM
4. Inconsistent behavior across different LLM providers

## Solution

We've implemented a "Single Message Roles Mode" that ensures:
1. There is only one "system" message
2. There is only one "user" message that includes both the original user message and the assistant's responses

This is achieved through the following components:

1. `single_message_prompt.py`: Provides a modified version of the `run_controller` function that ensures all instructions are sent in a single user message and prevents any additional user messages.
2. `agent_controller_patch.py`: Patches the `AgentController` class to ensure it properly handles our single message approach.
3. `single_message_roles.py`: Provides functionality to combine all messages into a single system and user message.
4. Updates to `run_infer.py`: Modified to use our patched components.

## Usage

No changes are needed to the existing scripts. The patch is automatically applied when running the AIME2025 benchmark.

## Testing

You can test the single message roles mode by running:

```bash
python test_single_message.py
```

This will run the AIME2025 benchmark with a single instance and verify that the single message roles mode is working correctly.

## Implementation Details

### Single Message Roles

1. **System Message**: We create a single system message with the standard OpenHands agent instructions.

2. **User Message**: We create a single user message that includes both the original user message and the assistant's responses.

3. **No Assistant Role**: We don't use the "assistant" role at all, which ensures that the LLM only sees two roles: "system" and "user".

4. **Tool Results Integration**: Tool results (observations) are included in the user message, formatted as "TOOL RESULT: [result]".

5. **LLM Patching**: We patch the LLM's completion method to transform the messages before sending them to the LLM.

These changes ensure that the LLM receives a single, coherent set of instructions in one system message and one user message, which should improve the consistency and quality of the responses.