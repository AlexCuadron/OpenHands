"""Conditional Prefix LLM module.

This module provides a wrapper that conditionally uses the prefix-based LLM
approach when running the AIME2025 benchmark, and the standard LLM approach otherwise.
"""

import os
import sys
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the original LLM class and the PrefixLLM class
from openhands.llm.llm import LLM as OriginalLLM
from openhands.prefix_llm import PrefixLLM
from openhands.core.config import LLMConfig

def is_running_aime2025():
    """Check if we're running the AIME2025 benchmark.
    
    This function checks the command line arguments and environment variables
    to determine if we're running the AIME2025 benchmark.
    
    Returns:
        bool: True if we're running the AIME2025 benchmark, False otherwise.
    """
    # Check command line arguments
    cmd_args = ' '.join(sys.argv)
    if 'aime2025' in cmd_args:
        return True
    
    # Check environment variables
    env_vars = os.environ.get('OPENHANDS_BENCHMARK', '')
    if 'aime2025' in env_vars.lower():
        return True
    
    # Check if the script path contains aime2025
    script_path = os.path.abspath(sys.argv[0])
    if 'aime2025' in script_path:
        return True
    
    return False

def create_conditional_llm(llm_config: LLMConfig):
    """Create an LLM instance based on the current context.
    
    If we're running the AIME2025 benchmark, this function creates a PrefixLLM instance.
    Otherwise, it creates a standard LLM instance.
    
    Args:
        llm_config: The LLM configuration.
        
    Returns:
        An LLM instance.
    """
    if is_running_aime2025():
        logger.info("Creating PrefixLLM for AIME2025 benchmark")
        return PrefixLLM(llm_config)
    else:
        logger.info("Creating standard LLM")
        return OriginalLLM(llm_config)

# Monkey patch the LLM creation function in the main module
def patch_llm_creation():
    """Patch the LLM creation function in the main module."""
    from openhands.core.main import create_llm
    
    # Store the original function
    original_create_llm = create_llm
    
    # Define the new function
    def new_create_llm(llm_config: LLMConfig):
        return create_conditional_llm(llm_config)
    
    # Replace the original function
    import openhands.core.main
    openhands.core.main.create_llm = new_create_llm
    
    logger.info("Patched LLM creation function")
    
    return original_create_llm

# Restore the original LLM creation function
def restore_llm_creation(original_create_llm):
    """Restore the original LLM creation function."""
    import openhands.core.main
    openhands.core.main.create_llm = original_create_llm
    logger.info("Restored original LLM creation function")