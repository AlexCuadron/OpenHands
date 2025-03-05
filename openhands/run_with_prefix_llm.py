"""Script to run OpenHands with the PrefixLLM class.

This script directly uses the PrefixLLM class by monkey patching the LLM class in the llm module.
This approach is different from the prefix_provider approach, which uses a custom LiteLLM provider.
"""

import os
import sys
import logging
import importlib.util

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the prefix LLM class
from openhands.prefix_llm import PrefixLLM

# Monkey patch the LLM class in the llm module
import openhands.llm.llm
original_LLM = openhands.llm.llm.LLM
openhands.llm.llm.LLM = PrefixLLM
logger.info("Monkey patched LLM class with PrefixLLM")

# Create a configuration file for our model
def create_config_file():
    """Create a configuration file for our model."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prefix_direct_config.toml")
    
    config_content = """[llm.sft]
model = "hosted_vllm/AlexCuadron/DSR1-Qwen-14B-8a4e8f3a-checkpoint-64"
temperature = 0.0
api_key = "ddd"
max_input_tokens = 4096
max_output_tokens = 4096
base_url = "http://127.0.0.1:8001/v1/"

[core]
workspace_base = "./workspace"
default_agent = "CodeActAgent"

[agent]
codeact_enable_browsing = true
codeact_enable_jupyter = true
enable_history_truncation = true
"""
    
    with open(config_path, "w") as f:
        f.write(config_content)
    
    logger.info(f"Created configuration file at {config_path}")
    return config_path

# Import OpenHands main module
from openhands.core.main import main

if __name__ == "__main__":
    # Add the current directory to the Python path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Create the configuration file
    config_path = create_config_file()
    
    # Run OpenHands with our modified LLM class
    sys.argv = [
        sys.argv[0],
        "--config", config_path,
        "--llm", "sft"
    ]
    
    logger.info("Starting OpenHands with PrefixLLM")
    try:
        main()
    finally:
        # Restore the original LLM class
        openhands.llm.llm.LLM = original_LLM
        logger.info("Restored original LLM class")