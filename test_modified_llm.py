"""Test script for the modified LLM class."""

import json
from pydantic import SecretStr

from openhands.core.config import LLMConfig
from openhands.llm import LLM


def main():
    """Test the modified LLM class."""
    # Create a basic LLM config
    config = LLMConfig(
        model="gpt-4o",
        api_key=SecretStr("dummy-key"),
        temperature=0.7,
        max_output_tokens=1000,
    )
    
    # Create an instance of our LLM
    llm = LLM(config)
    
    # Create a simple message list with an empty assistant message
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Solve this geometry problem: Find the perimeter of triangle ABC."},
        {"role": "assistant", "content": ""}  # Empty assistant message
    ]
    
    # Mock the completion function to return a properly structured response
    original_completion = llm._completion_unwrapped
    
    def mock_completion(*args, **kwargs):
        messages = kwargs.get('messages', args[1] if len(args) > 1 else [])
        return {
            "id": "mock-id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "mock-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "This is a mock response"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            },
            "_messages": messages  # Store the messages for our test
        }
    
    llm._completion_unwrapped = mock_completion
    
    # Call the completion function
    result = llm.completion(messages=messages)
    
    # Print the result
    print("Original messages:")
    print(json.dumps(messages, indent=2))
    print("\nModified messages:")
    print(json.dumps(result["_messages"], indent=2))
    
    # Verify that our prefix was added
    modified_messages = result["_messages"]
    has_thinking_prefix = any(
        msg.get("role") == "assistant" and 
        msg.get("content", "").startswith("<think>") and
        "tool_calls" in msg
        for msg in modified_messages
    )
    
    has_tool_response = any(
        msg.get("role") == "tool" and 
        msg.get("tool_call_id") == "toolu_01"
        for msg in modified_messages
    )
    
    print("\nVerification:")
    print(f"Has thinking prefix: {has_thinking_prefix}")
    print(f"Has tool response: {has_tool_response}")
    
    # Restore the original completion function
    llm._completion_unwrapped = original_completion


if __name__ == "__main__":
    main()