"""Script to run OpenHands with custom Qwen provider."""

import sys
import os
from openhands.core.main import main
from openhands.custom_qwen_provider import custom_qwen_completion  # Import to register the provider

if __name__ == "__main__":
    # Run OpenHands with our Qwen configuration
    sys.argv = [
        sys.argv[0],
        "--config", "qwen_config.toml",
        "--llm", "sft"
    ]
    main()