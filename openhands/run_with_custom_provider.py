"""Script to run OpenHands with the custom LiteLLM provider."""

import os
import sys
import importlib.util

# Import our custom LiteLLM provider
spec = importlib.util.spec_from_file_location(
    "custom_litellm_provider", 
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_litellm_provider.py")
)
custom_litellm_provider = importlib.util.module_from_spec(spec)
spec.loader.exec_module(custom_litellm_provider)

# Import OpenHands main module
from openhands.core.main import main

if __name__ == "__main__":
    # Run OpenHands with our custom configuration
    sys.argv = [
        sys.argv[0],
        "--config", "vllm_config.toml",
        "--llm", "sft"
    ]
    main()