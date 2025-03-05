"""Test script to demonstrate the transformation of messages for prefix-based conversations.

This script tests the transform_to_prefix_format function from the prefix_provider module
with various test cases to ensure it correctly transforms messages into the prefix-based format.
"""

import os
import sys
import json
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the transform function from prefix_provider
from prefix_provider import transform_to_prefix_format

def print_messages(title, messages):
    """Print messages in a readable format."""
    print(f"\n{title}:")
    print(json.dumps(messages, indent=2))
    print("-" * 80)

def test_transformation(test_name, messages):
    """Test the transformation of messages and print the results."""
    print(f"\n\n=== Test: {test_name} ===\n")
    
    # Print the original messages
    print_messages("Original Messages", messages)
    
    # Transform the messages
    transformed = transform_to_prefix_format(messages)
    
    # Print the transformed messages
    print_messages("Transformed Messages", transformed)
    
    return transformed

def run_tests():
    """Run various tests to demonstrate the transformation of messages."""
    # Test 1: Simple conversation
    test1_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you for asking! How can I help you today?"},
        {"role": "user", "content": "What's the weather like?"},
        {"role": "assistant", "content": "I don't have real-time weather information. Would you like me to help you find a weather service?"}
    ]
    test_transformation("Simple Conversation", test1_messages)
    
    # Test 2: Conversation with tool calls
    test2_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's 2 + 2?"},
        {"role": "assistant", "content": "To calculate 2 + 2, I'll use a calculator."},
        {"role": "tool", "content": "The result of 2 + 2 is 4."},
        {"role": "assistant", "content": "The answer is 4."},
        {"role": "user", "content": "Now what's 3 * 5?"}
    ]
    test_transformation("Conversation with Tool Calls", test2_messages)
    
    # Test 3: World Cup example
    test3_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world cup in 2022?"},
        {"role": "assistant", "content": "Let me check <function>get_world_cup_winner(2022)</function>"},
        {"role": "function", "content": "Argentina"},
        {"role": "user", "content": "What was the score?"},
        {"role": "assistant", "content": "Let me see by how much <function>get_world_cup_score(2022)</function>"},
        {"role": "function", "content": "Argentina 3(4) - France 3(2) on penalties"},
        {"role": "user", "content": "Who scored for Argentina?"}
    ]
    test_transformation("World Cup Example", test3_messages)
    
    # Test 4: Complex conversation with multiple tool calls
    test4_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a Python function to calculate the factorial of a number."},
        {"role": "assistant", "content": "I'll write a Python function to calculate the factorial of a number."},
        {"role": "tool", "content": "Here's a Python function to calculate the factorial of a number:\n\n```python\ndef factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n-1)\n```"},
        {"role": "assistant", "content": "Here's a recursive Python function to calculate the factorial of a number:"},
        {"role": "assistant", "content": "```python\ndef factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n-1)\n```"},
        {"role": "assistant", "content": "This function uses recursion to calculate the factorial. For example, factorial(5) would calculate 5 * 4 * 3 * 2 * 1 = 120."},
        {"role": "user", "content": "Can you also write an iterative version?"},
        {"role": "assistant", "content": "Sure, I'll write an iterative version of the factorial function."},
        {"role": "tool", "content": "Here's an iterative version of the factorial function:\n\n```python\ndef factorial_iterative(n):\n    result = 1\n    for i in range(1, n + 1):\n        result *= i\n    return result\n```"},
        {"role": "assistant", "content": "Here's an iterative version of the factorial function:"},
        {"role": "assistant", "content": "```python\ndef factorial_iterative(n):\n    result = 1\n    for i in range(1, n + 1):\n        result *= i\n    return result\n```"},
        {"role": "assistant", "content": "This version uses a loop instead of recursion, which can be more efficient for large numbers as it avoids the overhead of recursive function calls."}
    ]
    test_transformation("Complex Conversation with Multiple Tool Calls", test4_messages)
    
    # Test 5: Conversation with only system and user messages
    test5_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, who are you?"}
    ]
    test_transformation("Conversation with Only System and User Messages", test5_messages)
    
    # Test 6: Empty messages
    test6_messages = []
    test_transformation("Empty Messages", test6_messages)

if __name__ == "__main__":
    run_tests()