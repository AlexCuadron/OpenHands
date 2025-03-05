"""Test script for the prefix-based LLM class."""

import os
import sys
import logging
from pydantic import SecretStr

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the prefix LLM class
from openhands.prefix_llm import PrefixLLM
from openhands.core.config import LLMConfig
from openhands.llm.metrics import Metrics

def test_prefix_llm():
    """Test the prefix LLM class with a simple completion."""
    try:
        # Create a configuration for our model
        config = LLMConfig(
            model="hosted_vllm/AlexCuadron/DSR1-Qwen-14B-8a4e8f3a-checkpoint-64",
            temperature=0.0,
            api_key=SecretStr("ddd"),
            max_input_tokens=4096,
            max_output_tokens=4096,
            base_url="http://127.0.0.1:8001/v1/"
        )
        
        # Create a metrics object
        metrics = Metrics(model_name=config.model)
        
        # Create an instance of our prefix LLM class
        llm = PrefixLLM(config=config, metrics=metrics)
        
        # Test messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you for asking! How can I help you today?"},
            {"role": "user", "content": "What's the weather like?"}
        ]
        
        # Make a completion request using our prefix LLM class
        response = llm.completion(messages=messages)
        
        # Print the response
        logger.info("Response received:")
        logger.info(f"Content: {response.choices[0].message.content}")
        logger.info(f"Full response: {response}")
        
        # Test messages with tool calls
        tool_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's 2 + 2?"},
            {"role": "assistant", "content": "To calculate 2 + 2, I'll use a calculator."},
            {"role": "tool", "content": "The result of 2 + 2 is 4."},
            {"role": "assistant", "content": "The answer is 4."},
            {"role": "user", "content": "Now what's 3 * 5?"}
        ]
        
        # Make a completion request using our prefix LLM class
        tool_response = llm.completion(messages=tool_messages)
        
        # Print the response
        logger.info("\nTool Response received:")
        logger.info(f"Content: {tool_response.choices[0].message.content}")
        logger.info(f"Full response: {tool_response}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing prefix LLM class: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_prefix_llm()
    sys.exit(0 if success else 1)