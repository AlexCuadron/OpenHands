"""Test script for the conditional prefix LLM module."""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from openhands.conditional_prefix_llm import is_running_aime2025, create_conditional_llm
from openhands.core.config import LLMConfig

def test_is_running_aime2025():
    """Test the is_running_aime2025 function."""
    # Test with command line arguments
    original_argv = sys.argv.copy()
    
    # Test with aime2025 in command line arguments
    sys.argv = ['test.py', 'aime2025', 'arg2']
    result = is_running_aime2025()
    logger.info(f"is_running_aime2025() with 'aime2025' in argv: {result}")
    assert result is True
    
    # Test without aime2025 in command line arguments
    sys.argv = ['test.py', 'arg1', 'arg2']
    result = is_running_aime2025()
    logger.info(f"is_running_aime2025() without 'aime2025' in argv: {result}")
    assert result is False
    
    # Test with environment variable
    os.environ['OPENHANDS_BENCHMARK'] = 'aime2025'
    result = is_running_aime2025()
    logger.info(f"is_running_aime2025() with OPENHANDS_BENCHMARK='aime2025': {result}")
    assert result is True
    
    # Test with different environment variable
    os.environ['OPENHANDS_BENCHMARK'] = 'other'
    result = is_running_aime2025()
    logger.info(f"is_running_aime2025() with OPENHANDS_BENCHMARK='other': {result}")
    assert result is False
    
    # Restore original argv and environment
    sys.argv = original_argv
    if 'OPENHANDS_BENCHMARK' in os.environ:
        del os.environ['OPENHANDS_BENCHMARK']

def test_create_conditional_llm():
    """Test the create_conditional_llm function."""
    # Create a dummy LLM config
    llm_config = LLMConfig(model="dummy")
    
    # Test with aime2025 in command line arguments
    original_argv = sys.argv.copy()
    sys.argv = ['test.py', 'aime2025', 'arg2']
    
    llm = create_conditional_llm(llm_config)
    logger.info(f"create_conditional_llm() with 'aime2025' in argv: {type(llm).__name__}")
    
    # Restore original argv
    sys.argv = original_argv

if __name__ == "__main__":
    logger.info("Testing conditional_prefix_llm.py")
    test_is_running_aime2025()
    test_create_conditional_llm()
    logger.info("All tests passed!")