"""Script to run OpenHands with the prefix-based LiteLLM provider.

This script registers the prefix provider with LiteLLM and then runs OpenHands
with a custom configuration that uses the prefix-based LLM.
"""

import os
import sys
import logging
import importlib.util

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

# Import OpenHands main module
from openhands.core.main import main
from openhands.core.config import LLMConfig
from openhands.prefix_llm import PrefixLLM

# Monkey patch the LLM creation function to use our PrefixLLM
from openhands.core.main import create_llm

def create_prefix_llm(llm_config: LLMConfig):
    """Create a PrefixLLM instance from the given config."""
    logger.info(f"Creating PrefixLLM with config: {llm_config}")
    return PrefixLLM(llm_config)

# Replace the create_llm function with our custom function
create_llm_original = create_llm
create_llm = create_prefix_llm

if __name__ == "__main__":
    # Add the current directory to the Python path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Run OpenHands with our custom configuration
    sys.argv = [
        sys.argv[0],
        "--config", os.path.join(os.path.dirname(os.path.abspath(__file__)), "prefix_config.toml"),
        "--llm", "sft"
    ]
    
    logger.info("Starting OpenHands with prefix-based LLM")
    main()