"""Custom LiteLLM provider for Qwen models with <|im_start|> chat template."""

import copy
from typing import Dict, List, Any, Optional
import litellm
from litellm.utils import ModelResponse

def custom_qwen_completion(
    model: str,
    messages: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs
) -> ModelResponse:
    """Custom completion function for Qwen models with <|im_start|> chat template.
    
    This function modifies the request to use the /completions endpoint instead of /chat/completions.
    """
    # Deep copy the messages to avoid modifying the original
    messages_copy = copy.deepcopy(messages)
    
    # Format the prompt with <|im_start|> and <|im_end|> tags
    formatted_prompt = ""
    for msg in messages_copy:
        role = msg["role"]
        content = msg.get("content", "")
        formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    
    # Add the assistant start tag to prompt the model to continue
    formatted_prompt += "<|im_start|>assistant\n"
    
    # Make the API call using LiteLLM's completion endpoint
    response = litellm.completion(
        model=model,
        prompt=formatted_prompt,
        api_key=api_key,
        base_url=base_url,
        **kwargs
    )
    
    # Convert the completion response to chat completion format
    if response and hasattr(response, "choices") and len(response.choices) > 0:
        # Extract the generated text
        generated_text = response.choices[0].text
        
        # Remove any trailing <|im_end|> tags if present
        if "<|im_end|>" in generated_text:
            generated_text = generated_text.split("<|im_end|>")[0]
        
        # Update the response to match chat completion format
        response.choices[0].message = {"role": "assistant", "content": generated_text}
        
        # Remove text field which is specific to completion endpoint
        if hasattr(response.choices[0], "text"):
            delattr(response.choices[0], "text")
    
    return response

# Register our custom provider with LiteLLM
litellm.register_provider("custom_qwen", custom_qwen_completion)