"""Test script for the prefix-based LiteLLM provider."""

import os
import sys
import logging
import importlib.util
import litellm

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our custom prefix provider
spec = importlib.util.spec_from_file_location(
    "prefix_provider", 
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "prefix_provider.py")
)
prefix_provider = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prefix_provider)

def test_simple_conversation():
    """Test a simple conversation with the prefix provider."""
    try:
        # Configure litellm with debug mode
        litellm.set_verbose = True
        
        # Test messages for a simple conversation
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you for asking! How can I help you today?"},
            {"role": "user", "content": "What's the weather like?"},
            {"role": "assistant", "content": "I don't have real-time weather information. Would you like me to help you find a weather service?"},
            {"role": "user", "content": "No thanks, just tell me about yourself."}
        ]
        
        # Make a completion request using our prefix provider
        response = litellm.completion(
            model="hosted_vllm/AlexCuadron/DSR1-Qwen-14B-8a4e8f3a-checkpoint-64",
            messages=messages,
            api_key="ddd",
            base_url="http://127.0.0.1:8001/v1/",
            custom_llm_provider="prefix_provider",
            temperature=0.0,
            max_tokens=4096
        )
        
        # Print the response
        logger.info("Response received:")
        logger.info(f"Content: {response.choices[0].message.content}")
        logger.info(f"Full response: {response}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing prefix provider: {e}", exc_info=True)
        return False

def test_tool_conversation():
    """Test a conversation with tool calls using the prefix provider."""
    try:
        # Configure litellm with debug mode
        litellm.set_verbose = True
        
        # Test messages for a conversation with tool calls
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's 2 + 2?"},
            {"role": "assistant", "content": "To calculate 2 + 2, I'll use a calculator."},
            {"role": "tool", "content": "The result of 2 + 2 is 4."},
            {"role": "assistant", "content": "The answer is 4."},
            {"role": "user", "content": "Now what's 3 * 5?"},
            {"role": "assistant", "content": "Let me calculate 3 * 5."},
            {"role": "tool", "content": "The result of 3 * 5 is 15."},
            {"role": "assistant", "content": "The answer is 15."}
        ]
        
        # Make a completion request using our prefix provider
        response = litellm.completion(
            model="hosted_vllm/AlexCuadron/DSR1-Qwen-14B-8a4e8f3a-checkpoint-64",
            messages=messages,
            api_key="ddd",
            base_url="http://127.0.0.1:8001/v1/",
            custom_llm_provider="prefix_provider",
            temperature=0.0,
            max_tokens=4096
        )
        
        # Print the response
        logger.info("Response received:")
        logger.info(f"Content: {response.choices[0].message.content}")
        logger.info(f"Full response: {response}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing prefix provider with tools: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    logger.info("Testing simple conversation...")
    success1 = test_simple_conversation()
    
    logger.info("\nTesting tool conversation...")
    success2 = test_tool_conversation()
    
    sys.exit(0 if success1 and success2 else 1)