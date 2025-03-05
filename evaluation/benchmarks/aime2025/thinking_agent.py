"""
ThinkingAgent integration for AIME2025 benchmark.
This module provides functions to analyze model responses for overthinking behavior
and filter out solutions with high overthinking scores.
"""

import json
import os
import re
from typing import Dict, List, Tuple

from openhands.core.config import load_from_toml
from openhands.core.config.llm_config import LLMConfig
from openhands.core.logger import openhands_logger as logger
from openhands.llm.llm import LLM


def format_interaction_for_thinking_agent(history: List[Dict]) -> str:
    """
    Format the interaction history into a format suitable for the ThinkingAgent.

    Args:
        history: List of interaction events from the agent's history

    Returns:
        str: Formatted interaction string
    """
    formatted_str = ''

    # Extract the initial problem statement
    initial_message = None
    for event in history:
        if hasattr(event, 'message') and getattr(event, 'role', None) == 'user':
            initial_message = event.message
            break

    if initial_message:
        formatted_str += f'INITIAL PROBLEM:\n{initial_message}\n\n'
    else:
        formatted_str += 'INITIAL PROBLEM:\nNo initial message found\n\n'

    # Extract the interactions (assistant responses and tool calls/results)
    for i, event in enumerate(history):
        if (
            hasattr(event, 'role')
            and event.role == 'assistant'
            and hasattr(event, 'message')
        ):
            formatted_str += f'RESPONSE:\n{event.message}\n\n'
        elif hasattr(event, 'action'):
            if event.action == 'execute_ipython_cell':
                formatted_str += f'TOOL CALL: execute_ipython_cell\nCODE:\n{event.params["code"]}\n\n'
                if hasattr(event, 'result'):
                    formatted_str += f'RESULT:\n{event.result}\n\n'
            elif event.action == 'finish':
                formatted_str += f'TOOL CALL: finish\nSOLUTION: {event.params.get("solution", "No solution provided")}\n\n'

    return formatted_str


def get_thinking_agent_llm() -> LLM:
    """
    Create and return an LLM instance configured for the ThinkingAgent.

    Returns:
        LLM: An LLM instance configured for the ThinkingAgent
    """
    # Load the thinking agent config
    config_path = os.path.join(
        os.path.dirname(__file__), 'thinking_agent_config.toml'
    )
    config = load_from_toml(config_path)
    llm_config = LLMConfig(**config['llm'])
    return LLM(llm_config)


def analyze_overthinking(
    history: List[Dict], threshold: float = 0.5
) -> Tuple[bool, float, str]:
    """
    Analyze the agent's interaction history for overthinking behavior.

    Args:
        history: List of interaction events from the agent's history
        threshold: Threshold for determining overthinking (default: 0.5)

    Returns:
        Tuple[bool, float, str]: (is_overthinking, overthinking_score, analysis)
    """
    # Format the interaction history for the thinking agent
    formatted_interaction = format_interaction_for_thinking_agent(history)

    # Create the prompt for the thinking agent
    prompt = f"""You are an expert at analyzing mathematical problem-solving approaches. Your task is to analyze the following interaction between an AI assistant and a user solving a math problem from the American Invitational Mathematics Examination (AIME).

Specifically, I want you to evaluate whether the AI assistant is "overthinking" the problem - making it more complex than necessary, exploring tangential approaches, or failing to recognize simpler solutions.

Here's the interaction:

{formatted_interaction}

Please analyze this interaction and provide:
1. An overthinking score from 0.0 to 1.0, where:
   - 0.0 means the approach is direct and efficient
   - 1.0 means the approach is severely overthinking the problem
2. A brief explanation of your score
3. Suggestions for a more direct approach (if applicable)

Format your response as a JSON object with the following keys:
- "overthinking_score": a float between 0.0 and 1.0
- "explanation": a string explaining your score
- "suggestions": a string with suggestions for improvement

JSON:"""

    # Get the thinking agent LLM
    llm = get_thinking_agent_llm()

    # Get the analysis from the thinking agent
    try:
        response = llm.complete(prompt)
        # Extract the JSON from the response
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response

        # Clean up the JSON string
        json_str = json_str.strip()
        if json_str.startswith('```') and json_str.endswith('```'):
            json_str = json_str[3:-3].strip()

        # Parse the JSON
        analysis = json.loads(json_str)
        overthinking_score = float(analysis.get('overthinking_score', 0.0))
        explanation = analysis.get('explanation', '')
        suggestions = analysis.get('suggestions', '')

        # Determine if the agent is overthinking
        is_overthinking = overthinking_score >= threshold

        # Format the analysis
        analysis_str = f"Overthinking Score: {overthinking_score:.2f}\n\nExplanation: {explanation}\n\nSuggestions: {suggestions}"

        return is_overthinking, overthinking_score, analysis_str
    except Exception as e:
        logger.error(f"Error analyzing overthinking: {e}")
        return False, 0.0, f"Error analyzing overthinking: {e}"


def should_discard_solution(
    history: List[Dict], overthinking_threshold: float = 0.5
) -> Tuple[bool, str]:
    """
    Determine if a solution should be discarded due to overthinking.

    Args:
        history: List of interaction events from the agent's history
        overthinking_threshold: Threshold for determining overthinking (default: 0.5)

    Returns:
        Tuple[bool, str]: (should_discard, reason)
    """
    # Analyze the interaction for overthinking
    is_overthinking, score, analysis = analyze_overthinking(
        history, overthinking_threshold
    )

    # If the agent is overthinking, discard the solution
    if is_overthinking:
        reason = f"Solution discarded due to overthinking (score: {score:.2f}).\n\n{analysis}"
        return True, reason

    return False, ""