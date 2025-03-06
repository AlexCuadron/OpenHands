# AIME2025 Benchmark - Simple Message Structure

This document describes a simplified approach to the AIME2025 benchmark that ensures:
1. NO system messages are ever used
2. There is ALWAYS at most ONE user message
3. There is ALWAYS at most ONE assistant message

## Problem

The original implementation of the AIME2025 benchmark had several issues:
1. Multiple "user" roles in the prompt
2. Multiple "assistant" roles in the prompt
3. "system" role messages that might not be handled consistently across LLM providers
4. Fragmented instructions that might confuse the LLM
5. Inconsistent behavior across different LLM providers

## Solution

We've implemented a simple approach that directly modifies the LLM's message processing:

1. `llm_aime2025.py`: A new module that patches the LLM to ensure:
   - NO system messages are ever used
   - There is ALWAYS at most ONE user message
   - There is ALWAYS at most ONE assistant message

2. `aime2025_llm_patch.py`: A simple patch that applies the LLM modification to the AIME2025 benchmark.

3. `run_infer.py`: Modified to use our new patch.

## Implementation Details

The implementation is straightforward:

1. We patch the LLM's completion methods at all levels:
   - `completion`
   - `_completion`
   - `_completion_unwrapped`

2. For each method, we process the messages to:
   - Collect all system content and add it to the user message
   - Combine all user messages into a single user message
   - Combine all assistant messages into a single assistant message

3. The result is a simple message structure:
   - ONE user message (including any system content)
   - ONE assistant message (including all previous assistant content)

This approach ensures that the LLM receives a consistent message structure, which should improve the quality of the responses.

## Benefits

1. **Simplicity**: The implementation is simple and easy to understand.
2. **Consistency**: The message structure is consistent across all LLM providers.
3. **Reliability**: By patching at the lowest level, we ensure that our changes are applied to all LLM calls.
4. **Maintainability**: The code is modular and can be easily updated or extended.