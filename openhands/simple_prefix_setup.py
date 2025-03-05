"""Simple setup script for prefix-based LLM.

This script provides a simplified way to use the prefix-based LLM approach
without relying on the full OpenHands infrastructure.
"""

import os
import sys
import logging
import importlib.util
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our custom prefix provider
prefix_provider_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prefix_provider.py")
spec = importlib.util.spec_from_file_location("prefix_provider", prefix_provider_path)
prefix_provider = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prefix_provider)

# Import litellm
import litellm

# Simple PrefixLLM class that can be used directly
class SimplePrefixLLM:
    """A simple class that wraps litellm.completion to use prefix-based conversations."""
    
    def __init__(self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        """Initialize the SimplePrefixLLM.
        
        Args:
            model: The model to use for completion
            api_key: The API key to use
            base_url: The base URL for the API
            **kwargs: Additional arguments to pass to litellm.completion
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.kwargs = kwargs
        logger.info(f"Initialized SimplePrefixLLM with model: {model}")
    
    def completion(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        """Call litellm.completion with prefix-based messages.
        
        Args:
            messages: The messages to send to the model
            **kwargs: Additional arguments to pass to litellm.completion
            
        Returns:
            The response from litellm.completion
        """
        # Transform messages to prefix format
        transformed_messages = prefix_provider.transform_to_prefix_format(messages)
        
        # Log the transformed messages
        logger.debug(f"Original messages: {messages}")
        logger.debug(f"Transformed messages: {transformed_messages}")
        
        # Merge kwargs with self.kwargs
        all_kwargs = {**self.kwargs, **kwargs}
        
        # Call litellm.completion with the transformed messages
        try:
            if all_kwargs.get('custom_llm_provider') == 'prefix_provider':
                response = prefix_provider.prefix_completion(
                    model=self.model,
                    messages=transformed_messages,
                    api_key=self.api_key,
                    base_url=self.base_url,
                    **all_kwargs
                )
            else:
                response = litellm.completion(
                    model=self.model,
                    messages=transformed_messages,
                    api_key=self.api_key,
                    base_url=self.base_url,
                    **all_kwargs
                )
            return response
        except Exception as e:
            logger.error(f"Error calling litellm.completion: {e}")
            raise

# Function to patch litellm.completion to use prefix-based messages
def patch_litellm_completion():
    """Patch litellm.completion to use prefix-based messages."""
    original_completion = litellm.completion
    
    def patched_completion(model: str, messages: List[Dict[str, Any]], **kwargs):
        """Patched version of litellm.completion that uses prefix-based messages."""
        # Transform messages to prefix format
        transformed_messages = prefix_provider.transform_to_prefix_format(messages)
        
        # Log the transformed messages
        logger.debug(f"Original messages: {messages}")
        logger.debug(f"Transformed messages: {transformed_messages}")
        
        # Call the original completion function with the transformed messages
        return original_completion(model=model, messages=transformed_messages, **kwargs)
    
    # Replace the original completion function with our patched version
    litellm.completion = patched_completion
    
    return original_completion

# Function to restore the original litellm.completion
def restore_litellm_completion(original_completion):
    """Restore the original litellm.completion function."""
    litellm.completion = original_completion

if __name__ == "__main__":
    # Example usage
    original_completion = patch_litellm_completion()
    
    try:
        # Use litellm.completion with prefix-based messages
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"}
            ]
        )
        print(response)
    finally:
        # Restore the original litellm.completion
        restore_litellm_completion(original_completion)