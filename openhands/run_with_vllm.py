"""Script to run OpenHands with vLLM."""

import sys
from openhands.core.main import main

if __name__ == "__main__":
    # Run OpenHands with our vLLM configuration
    sys.argv = [
        sys.argv[0],
        "--config", "vllm_config.toml",
        "--llm", "sft"
    ]
    main()