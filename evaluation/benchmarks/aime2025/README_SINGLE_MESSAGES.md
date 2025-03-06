# AIME2025 Benchmark - Single Message Mode

This document describes the modifications made to the AIME2025 benchmark to ensure:
1. All instructions are sent in a single user message
2. All assistant responses are combined into a single growing assistant message

## Problem

The original implementation of the AIME2025 benchmark had several issues:
1. Multiple "user" roles in the prompt
2. Multiple "assistant" roles in the prompt
3. Fragmented instructions that might confuse the LLM
4. Inconsistent behavior across different LLM providers
5. Additional user messages being sent when the agent tries to finish without using Python

## Solution

We've implemented a "Single Message Mode" that ensures:
1. All instructions are sent in a single user message
2. All assistant responses are combined into a single growing assistant message

This is achieved through the following components:

1. `single_message_prompt.py`: Provides a modified version of the `run_controller` function that ensures all instructions are sent in a single user message and prevents any additional user messages.
2. `agent_controller_patch.py`: Patches the `AgentController` class to ensure it properly handles our single message approach.
3. `single_assistant_message.py`: Provides functionality to combine all assistant messages and tool results into a single growing message.
4. Updates to `run_infer.py`: Modified to use our patched components.

## Usage

No changes are needed to the existing scripts. The patch is automatically applied when running the AIME2025 benchmark.

## Testing

You can test the single message mode by running:

```bash
python test_single_message.py
```

This will run the AIME2025 benchmark with a single instance and verify that the single message mode is working correctly.

## Implementation Details

### Single User Message

1. **Preventing Additional User Messages**: We've completely replaced the fake user response function with one that always returns '/exit' to prevent any additional user messages from being sent.

2. **Event Handling**: We've added event handling to detect when the agent is waiting for user input and automatically finish the interaction instead of sending a new user message.

3. **Process Event Patching**: We've patched the `AgentController.process_event` method to ensure that only one user message is in the history at any time and to intercept any follow-up user messages.

### Single Assistant Message

1. **Message Accumulation**: We've implemented a mechanism to accumulate all assistant messages and tool results into a single growing message.

2. **Prefix Parameter**: We use the 'prefix' parameter to indicate that the assistant message should be treated as a prefix for the next assistant message.

3. **Tool Results Integration**: We include tool results (observations) in the growing assistant message, formatted as "TOOL RESULT: [result]".

4. **LLM Patching**: We patch the LLM's completion method to transform the messages before sending them to the LLM.

These changes ensure that the LLM receives a single, coherent set of instructions in one user message and maintains a single, growing assistant message, which should improve the consistency and quality of the responses.