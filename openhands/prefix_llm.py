"""Modified LLM module that uses prefix-based conversations.

This module provides a custom LLM class that transforms standard OpenHands message format
into a prefix-based format where the assistant's previous responses and observations are
combined into a growing narrative that's included as a prefix in subsequent turns.
"""

import copy
import logging
from typing import List, Dict, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the original LLM class
from openhands.llm.llm import LLM as OriginalLLM

# Import the transform function from prefix_provider to ensure consistency
from openhands.prefix_provider import transform_to_prefix_format

class PrefixLLM(OriginalLLM):
    """Modified LLM class that uses prefix-based conversations.
    
    This class overrides the completion method to transform messages into a prefix-based format
    where the assistant's previous responses and observations are combined into a growing
    narrative that's included as a prefix in subsequent turns.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the PrefixLLM."""
        super().__init__(*args, **kwargs)
        logger.info("Initialized PrefixLLM with prefix-based conversation format")
    
    def completion(self, *args, **kwargs):
        """Override the completion method to transform messages to prefix format.
        
        This method extracts the messages from args or kwargs, transforms them into
        prefix-based format, and then calls the parent completion method with the
        transformed messages.
        
        Args:
            *args: Positional arguments to pass to the parent completion method
            **kwargs: Keyword arguments to pass to the parent completion method
            
        Returns:
            The response from the parent completion method
        """
        # Extract messages from args or kwargs
        messages = None
        if len(args) > 0:
            messages = args[0]
        elif 'messages' in kwargs:
            messages = kwargs['messages']
        
        if messages:
            # Log original messages for debugging
            logger.debug(f"Original messages: {messages}")
            
            # Transform messages to prefix format
            transformed_messages = transform_to_prefix_format(messages)
            
            # Log transformed messages for debugging
            logger.debug(f"Transformed messages: {transformed_messages}")
            
            # Update args or kwargs with transformed messages
            if len(args) > 0:
                args = (transformed_messages,) + args[1:]
            else:
                kwargs['messages'] = transformed_messages
        
        # Call the parent completion method with transformed messages
        return super().completion(*args, **kwargs)
    
    def format_messages_for_llm(self, messages):
        """Override the format_messages_for_llm method to handle prefix-based messages.
        
        This method ensures that the prefix attribute is preserved when formatting messages
        for the LLM.
        
        Args:
            messages: The messages to format
            
        Returns:
            The formatted messages
        """
        formatted_messages = super().format_messages_for_llm(messages)
        
        # Ensure prefix attribute is preserved
        for i, msg in enumerate(formatted_messages):
            if i > 0 and msg.get('role') == 'assistant' and i < len(messages):
                if hasattr(messages[i], 'prefix') and messages[i].prefix:
                    msg['prefix'] = True
        
        return formatted_messages