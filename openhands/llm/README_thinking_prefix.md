# Thinking Prefix for Empty Assistant Messages

The LLM class has been modified to automatically add a thinking prefix and tool response when the first assistant message is empty.

## Purpose

This modification makes the model believe that certain tools (like Python libraries) are already installed, by injecting a predefined tool call and its response at the beginning of the conversation.

## How It Works

When the LLM processes messages, it checks if there are any assistant messages and if the first one is empty. If so, it:

1. Inserts a thinking prefix message with a tool call to install Python libraries (sympy, numpy, scipy, matplotlib)
2. Inserts a tool response message showing that the libraries were successfully installed
3. Continues with the normal conversation

This makes the model believe that these libraries are already installed and available for use, without actually having to install them.

## Usage

You don't need to do anything special to use this feature. Just use the LLM class as usual:

```python
from openhands.core.config import LLMConfig
from openhands.llm import LLM

# Create a config
config = LLMConfig(
    model="your-model-name",
    api_key=SecretStr("your-api-key"),
    temperature=0.7,
    max_output_tokens=1000,
)

# Create an instance of LLM
llm = LLM(config)

# Use it as usual
response = llm.completion(messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Solve this geometry problem: Find the perimeter of triangle ABC."}
])
```

## Customization

You can modify the `llm.py` file to change:

- The thinking prefix content
- The tool call (e.g., to install different libraries)
- The tool response

Look for the section in `llm.py` that starts with:

```python
# Check if there are any assistant messages and if the first one is empty
assistant_messages = [msg for msg in messages if msg.get('role') == 'assistant']
if not assistant_messages or not assistant_messages[0].get('content'):
    # ...
```