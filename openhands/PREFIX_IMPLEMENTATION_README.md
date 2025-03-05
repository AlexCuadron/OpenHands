# Prefix-Based Conversation Implementation for OpenHands

This implementation enhances OpenHands to support prefix-based conversations, where the assistant's previous responses and observations are combined into a growing narrative that's included as a prefix in subsequent turns. This approach is particularly useful for:

1. Models that support the `prefix` parameter in their API (like DeepSeek)
2. Creating a more coherent conversation flow where the assistant builds on its previous responses
3. Maintaining context across multiple turns, especially with function/tool calls

## Files Created

1. **prefix_provider.py**
   - Implements a custom LiteLLM provider that handles prefix-based conversations
   - Contains the `transform_to_prefix_format` function that converts standard messages to prefix format
   - Registers the provider with LiteLLM

2. **prefix_llm.py**
   - Contains the `PrefixLLM` class that inherits from the original `LLM` class
   - Overrides the `completion` method to transform messages to prefix format
   - Overrides the `format_messages_for_llm` method to handle prefix-based messages

3. **run_with_prefix.py**
   - Script to run OpenHands with the prefix-based LLM implementation
   - Monkey patches the LLM creation function to use our PrefixLLM class
   - Uses a custom configuration file for the model

4. **test_prefix_transformation.py**
   - Test script to demonstrate the transformation of messages
   - Includes various test cases, including the World Cup example

## How It Works

### Message Transformation

The key part of this implementation is the `transform_to_prefix_format` function, which:

1. Extracts system messages and prepends them to the first user message
2. Processes the conversation sequentially, building up the assistant's narrative
3. Combines assistant responses and observations (from tools/functions) into a coherent narrative
4. Uses the `prefix=True` parameter to indicate that the assistant's narrative should be treated as a prefix

### Example Transformation: World Cup Example

Original messages:
```json
[
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "Who won the world cup in 2022?"},
  {"role": "assistant", "content": "Let me check <tool>get_world_cup_winner(2022)</tool>"},
  {"role": "tool", "content": "Argentina"},
  {"role": "user", "content": "What was the score?"}
]
```

Transformed messages:
```json
[
  {
    "role": "user",
    "content": "You are a helpful assistant.\n\nWho won the world cup in 2022?"
  },
  {
    "role": "assistant",
    "content": "Let me check <tool>get_world_cup_winner(2022)</tool>\nObservation: Argentina",
    "prefix": true
  },
  {
    "role": "user",
    "content": "What was the score?"
  }
]
```

Next turn (after assistant responds and function is called):
```json
[
  {
    "role": "user",
    "content": "You are a helpful assistant.\n\nWho won the world cup in 2022?"
  },
  {
    "role": "assistant",
    "content": "Let me check <tool>get_world_cup_winner(2022)</tool>\nObservation: Argentina\nLet me see by how much <tool>get_world_cup_score(2022)</tool>\nObservation: Argentina 3(4) - France 3(2) on penalties",
    "prefix": true
  },
  {
    "role": "user",
    "content": "Who scored for Argentina?"
  }
]
```

## Usage

To use this implementation:

1. Run OpenHands with the prefix-based provider:
   ```
   python openhands/run_with_prefix.py
   ```

2. To test the message transformation:
   ```
   python openhands/test_prefix_transformation.py
   ```

## Configuration

The configuration for the model is defined in `prefix_config.toml`:

```toml
[llm.sft]
model = "hosted_vllm/AlexCuadron/DSR1-Qwen-14B-8a4e8f3a-checkpoint-64"
temperature = 0.0
api_key = "ddd"
max_input_tokens = 4096
max_output_tokens = 4096
base_url = "http://127.0.0.1:8001/v1/"
custom_llm_provider = "prefix_provider"

[core]
workspace_base = "./workspace"
default_agent = "CodeActAgent"

[agent]
codeact_enable_browsing = true
codeact_enable_jupyter = true
enable_history_truncation = true
```

## Benefits of This Approach

1. **Improved Context**: The assistant maintains context across turns by building on its previous responses
2. **Better Function Calling**: Function calls and their responses are incorporated into the assistant's narrative
3. **Compatibility**: Works with models that support the `prefix` parameter (like DeepSeek)
4. **Flexibility**: Can be easily adapted for different message formats and models

## Example Use Case: World Cup Query

In this example, the user asks about the 2022 World Cup:

1. User: "Who won the world cup in 2022?"
2. Assistant: "Let me check <tool>get_world_cup_winner(2022)</tool>"
3. Tool returns: "Argentina"
4. User: "What was the score?"
5. Assistant (with prefix): "Let me check <tool>get_world_cup_winner(2022)</tool>\nObservation: Argentina\nLet me see by how much <tool>get_world_cup_score(2022)</tool>"
6. Tool returns: "Argentina 3(4) - France 3(2) on penalties"
7. User: "Who scored for Argentina?"
8. Assistant (with prefix): "Let me check <tool>get_world_cup_winner(2022)</tool>\nObservation: Argentina\nLet me see by how much <tool>get_world_cup_score(2022)</tool>\nObservation: Argentina 3(4) - France 3(2) on penalties\nLet me find out who scored for Argentina <tool>get_world_cup_scorers(2022, 'Argentina')</tool>"

This approach allows the assistant to build a coherent narrative across multiple turns, incorporating both its own responses and the results of tool calls.