"""
This module provides a simple patch for the AIME2025 benchmark.
"""

from openhands.controller.state.state import State
from openhands.core.logger import openhands_logger as logger
from openhands.llm.llm_aime2025 import patch_llm_for_aime2025


def patch_llm_for_aime2025_benchmark(state: State) -> None:
    """
    Patch the LLM for the AIME2025 benchmark.
    
    Args:
        state: The current state of the agent
    """
    if not hasattr(state, 'agent') or not hasattr(state.agent, 'llm'):
        logger.warning('Cannot patch LLM for AIME2025 benchmark: agent or llm not found')
        return
    
    # Patch the LLM
    patch_llm_for_aime2025(state.agent.llm)
    
    logger.info('LLM patched for AIME2025 benchmark')