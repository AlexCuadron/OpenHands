"""
ThinkingAgent integration for AIME2024 benchmark.
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
            # This is a tool call
            action = event.action
            action_input = getattr(event, 'action_input', {})
            formatted_str += f'OBSERVATION:\n[Tool Call: {action}]\n{json.dumps(action_input, indent=2)}\n\n'
        elif hasattr(event, 'result'):
            # This is a tool result
            formatted_str += f'OBSERVATION:\n{event.result}\n\n'

    return formatted_str


def save_interaction_to_file(
    history: List[Dict], output_dir: str, instance_id: str
) -> str:
    """
    Save the interaction history to a file in the format expected by the ThinkingAgent.

    Args:
        history: List of interaction events from the agent's history
        output_dir: Directory to save the file
        instance_id: ID of the instance

    Returns:
        str: Path to the saved file
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Format the interaction history
    formatted_interaction = format_interaction_for_thinking_agent(history)

    # Save to file
    file_path = os.path.join(output_dir, f'responses_observations_{instance_id}.txt')
    with open(file_path, 'w') as f:
        f.write(formatted_interaction)

    return file_path


def create_overthinking_analysis_prompt(interaction_content: str) -> str:
    """
    Create a prompt for the LLM to analyze overthinking behavior.

    Args:
        interaction_content: Formatted interaction content

    Returns:
        str: Analysis prompt
    """
    prompt = """
You are an AI judge focused on detecting when models prefer their internal reasoning chain over interacting with the environment.

<INTERACTION>
"""

    prompt += interaction_content
    prompt += """

    </INTERACTION>

    Analyze the <INTERACTION> and determine if the model is preferring their internal reasoning chain over interacting with the environment:

    How could this be detected?
    <CORE PRINCIPLE>
    - The model suffers from Analysis Paralysis, it focuses on heavy planning instead of interacting with the environment.
    - The model suffers from Rogue actions, after facing setbacks, it generates multiple actions without waiting for the environment to process the previous action.
    - The model suffers from Premature Disengagement, it concludes the task without checking with the environment. Either because it is overconfident in the solution or because it thinks it can't solve the problem.
    </CORE PRINCIPLE>

    <SCORING SYSTEM (0-10)>
    0-3: Always interacting with the environment
    - A summary of what has been done so far is good, even if done multiple times.
    - A brief summary of the steps to take is good if the model interacts with the environment following steps one by one.
    - Only one action per turn, finish and other actions are NOT allowed.
    - Alternating between two operations is good.
    - Trying the same approach over and over is good, even with long or complex actions, as long as the model waits for environment feedback each time.
    - Repeating similar patterns or configurations is fine as long as the model interacts with the environment between attempts.
    - Detailed reasoning and planning is good if it leads to concrete actions with environment interaction.

    4-7: Sometimes relies too much on their internal reasoning chain, but still interacts with the environment.
    - It engages in heavy planning, but still interacts with the environment.
    - It NEVER concludes the task without checking with the environment.
    - It might output multiple steps ONE time, but at subsequent turns it interacts one step at a time.
    - Long theoretical discussions are acceptable if they eventually result in concrete actions.

    8-10: Completely relies on their internal reasoning chain.
    - Focuses solely on their internal reasoning chain, with no concrete actions following the analysis.
    - Generates multiple actions without waiting for environment response.
    - The model prematurely concludes the task. Either because it is overconfident in the solution or because it thinks it can't solve the problem.
    - Generates many steps without any environment interaction.
    - Gets stuck in endless theoretical discussion without attempting solutions.
    </SCORING SYSTEM>

    <ANALYSIS STEPS>
    1. Analysis Paralysis
       - Is the model focusing on heavy planning instead of interacting with the environment?
       - Does the model interact with the environment at all?
       - Does the model follows its planned steps starting from the first one?
    2. Rogue Actions
       - Does the model generate multiple actions without waiting for the environment to process the previous action?
       - Is this behavior after a facing a setback?
       - Does this behaviour happen often?
    3. Premature Disengagement
       - Does the model prematurely conclude the task?
       - Is the model overconfident in the solution?
       - Is the model thinking it can't solve the problem?
    </ANALYSIS STEPS>

    <EXAMPLES>

    Example 1 - Persistent Retries (Good):
    EXECUTION RESULT: "Error: Invalid configuration"
    Model: *tries complex configuration A*
    EXECUTION RESULT: "Error: Invalid configuration"
    Model: *tries similar complex configuration A with slight modification*
    EXECUTION RESULT: "Error: Invalid configuration"
    Model: *tries complex configuration A again with another modification*
    Score: 0 - The model is persistently trying to solve the problem, waiting for environment feedback between each attempt. Even though the attempts are similar and complex, it's properly interacting with the environment.

    Example 2 - Thoughtful Planning (Good):
    Model: *provides detailed analysis of the problem and potential approaches*
    Model: *tries specific solution based on analysis*
    EXECUTION RESULT: "Error in implementation"
    Model: *refines approach based on error and tries again*
    Score: 0 - While the model engages in detailed planning, it follows through with concrete actions and responds to environment feedback.

    Example 3 - Stuck in a loop (Good):
    EXECUTION RESULT: "ERROR"
    Model: *apply fix_0*
    EXECUTION RESULT: "ERROR"
    Model: *apply SAME fix_0*
    EXECUTION RESULT: "ERROR"
    Model: *apply SAME fix_0*
    Score: 0 - Stuck in a loop is good.

    Example 4 - Analysis Paralysis:
    EXECUTION RESULT: "Invalid indentation line 10"
    Model: *Maybe I should... Perhaps I should... It should be... Let me try to start again rewriting the class*
    EXECUTION RESULT: "Still invalid line 10"
    Model: *Its not working... We also need to fix this other thing...*
    EXECUTION RESULT:  "Same error line 10"
    Score: 10 - focuses on its internal reasoning chain instead of the environment.

    Example 5 - Premature Disengagement:
    EXECUTION RESULT: "Invalid indentation line 10"
    Model: *This fixes it! I'll conclude the task. <function=finish>*
    Score: 10 - The model concludes the task without applying the fix or overconfidence in the solution.

    Example 6 - Rogue Actions:
    EXECUTION RESULT: "Invalid indentation line 10"
    Model: *Oh no, I forgot to add the old string, let me call the function again <function=str_replace_editor>...</function> and then we do this other thing <function=str_replace_editor>...</function>*
    Score: 10 - The model generates multiple actions after facing a setback without waiting for the environment to process the previous action.

    </EXAMPLES>

    <IMPORTANT>
    Format your response as:
    <answer>
    {
        "overthinking_score": "[0-10]",
        "reasoning": "Explain your reasoning for the score, be careful with new lines as they might break the JSON parsing"
    }
    </answer>
    Always surround your answer with <answer> and </answer> tags.
    Take your time to understand the interaction and analyze it carefully.
    Think step by step if models prefer their internal reasoning chain over interacting with the environment.
    </IMPORTANT>
"""
    return prompt


def analyze_overthinking(
    history: List[Dict], llm: LLM, output_dir: str = None, instance_id: str = None
) -> Tuple[int, Dict]:
    """
    Analyze the interaction history for overthinking behavior.

    Args:
        history: List of interaction events from the agent's history
        llm: LLM instance to use for analysis
        output_dir: Directory to save interaction files (optional)
        instance_id: ID of the instance (optional)

    Returns:
        Tuple[int, Dict]: Overthinking score and detailed analysis
    """
    # Save the interaction to a file if output_dir and instance_id are provided
    if output_dir and instance_id:
        interaction_file = save_interaction_to_file(history, output_dir, instance_id)
        logger.info(f'Saved interaction to file: {interaction_file}')

        # Read the interaction content from the file
        with open(interaction_file, 'r') as f:
            interaction_content = f.read()
    else:
        # Format the interaction history directly
        interaction_content = format_interaction_for_thinking_agent(history)

    # Create the analysis prompt
    prompt = create_overthinking_analysis_prompt(interaction_content)

    # Get the analysis from the LLM
    messages = [{'role': 'user', 'content': prompt}]
    response = llm.completion(messages=messages)

    # Extract the JSON response
    try:
        # Extract content from the response
        if hasattr(response, 'choices') and len(response.choices) > 0:
            if hasattr(response.choices[0], 'message'):
                content = response.choices[0].message.content
            elif hasattr(response.choices[0], 'text'):
                content = response.choices[0].text
            else:
                logger.warning("Unexpected response format from LLM")
                content = str(response)
        else:
            logger.warning("Unexpected response format from LLM")
            content = str(response)
            
        # Find JSON content using regex
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            analysis = json.loads(json_match.group(0))
            overthinking_score = int(analysis.get('overthinking_score', 0))

            # Save the analysis to a file if output_dir and instance_id are provided
            if output_dir and instance_id:
                analysis_file = os.path.join(
                    output_dir, f'overthinking_analysis_{instance_id}.json'
                )
                with open(analysis_file, 'w') as f:
                    json.dump(analysis, f, indent=2)
                logger.info(f'Saved overthinking analysis to file: {analysis_file}')

                # Also save the full LLM response
                response_file = os.path.join(
                    output_dir, f'overthinking_response_{instance_id}.txt'
                )
                with open(response_file, 'w') as f:
                    f.write(content)
                logger.info(f'Saved overthinking response to file: {response_file}')

            return overthinking_score, analysis
        else:
            logger.warning('Could not extract JSON from LLM response')
            return 0, {'error': 'Could not extract JSON from LLM response'}
    except Exception as e:
        logger.error(f'Error analyzing overthinking: {e}')
        return 0, {'error': str(e)}


def should_discard_solution(overthinking_score: int, threshold: int) -> bool:
    """
    Determine if a solution should be discarded based on its overthinking score.

    Args:
        overthinking_score: The overthinking score (0-10)
        threshold: The threshold above which solutions should be discarded

    Returns:
        bool: True if the solution should be discarded, False otherwise
    """
    return overthinking_score > threshold


def get_thinking_agent_llm() -> LLM:
    """
    Initialize an LLM instance for the ThinkingAgent.

    Returns:
        LLM: Initialized LLM instance
    """
    # Try to load config from the ThinkingAgent config file if it exists
    thinking_agent_config_path = os.path.join(
        os.path.dirname(__file__), 'thinking_agent_config.toml'
    )

    if os.path.exists(thinking_agent_config_path):
        # Import toml directly to avoid issues with load_from_toml
        import toml
        try:
            config_data = toml.load(thinking_agent_config_path)
            llm_config = LLMConfig.model_validate(config_data.get('llm', {}))
        except Exception as e:
            logger.warning(f"Error loading thinking agent config: {e}. Using default config.")
            # Use default configuration
            llm_config = LLMConfig(
                model='claude-3-5-sonnet-20241022', temperature=0.0, max_output_tokens=4096
            )
    else:
        # Use default configuration
        llm_config = LLMConfig(
            model='claude-3-5-sonnet-20241022', temperature=0.0, max_output_tokens=4096
        )

    return LLM(llm_config)
