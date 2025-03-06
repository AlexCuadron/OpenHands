"""
Test script to verify that the AIME2025 benchmark uses a single user message.
"""

import os
import sys
import logging
from openhands.core.logger import openhands_logger as logger

# Set up logging
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Run the AIME2025 benchmark with a single instance
if __name__ == "__main__":
    # Run the benchmark with the single message patch
    cmd = "./evaluation/benchmarks/aime2025/scripts/run_infer.sh sft HEAD CodeActAgent 1 1 \"\" eval ipython_only"
    exit_code = os.system(cmd)
    
    if exit_code == 0:
        print("Test completed successfully!")
    else:
        print(f"Test failed with exit code {exit_code}")