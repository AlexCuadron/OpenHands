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


def save_interaction_to_file(
    history: List[Dict], output_dir: str, instance_id: str
) -> str:
    """
    Save the interaction history to a file.

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
    interaction_content = format_interaction_for_thinking_agent(history)

    # Save the interaction to a file
    interaction_file = os.path.join(output_dir, f'interaction_{instance_id}.txt')
    with open(interaction_file, 'w') as f:
        f.write(interaction_content)

    return interaction_file


def create_overthinking_analysis_prompt(interaction_content: str) -> str:
    """
    Create a prompt for analyzing overthinking behavior.

    Args:
        interaction_content: Formatted interaction content

    Returns:
        str: Prompt for the LLM
    """
    prompt = f"""You are an expert at analyzing mathematical problem-solving approaches. Your task is to analyze the following interaction between an AI assistant and a user solving a math problem from the American Invitational Mathematics Examination (AIME).

I want you to evaluate whether the AI assistant is "overthinking" the problem - making it more complex than necessary, exploring tangential approaches, or failing to recognize simpler solutions.

Here's the interaction:

{interaction_content}

<EXAMPLES>
Example 1: Direct and Efficient Approach
INITIAL PROBLEM: Find the sum of all integer bases b>9 for which 17_b is a divisor of 97_b.
Model: *I'll convert these numbers to base 10 first to understand what we're working with.*
EXECUTION RESULT: *Shows conversion calculations*
Model: *Now I'll check which bases satisfy the divisibility condition.*
EXECUTION RESULT: *Shows systematic checking of bases*
Model: *After checking all possibilities, I've found that bases 11, 16, 19, and 24 work.*
EXECUTION RESULT: *Verification confirms these are the only solutions*
Score: 0 - The model takes a direct approach, uses code to verify each step, and systematically solves the problem without unnecessary detours.

Example 2: Moderate Overthinking
INITIAL PROBLEM: Find the number of ordered pairs (x,y), where both x and y are integers between -100 and 100, inclusive, such that 12x^2-xy-6y^2=0.
Model: *I'll start by rearranging this equation to see if I can factor it.*
EXECUTION RESULT: *Shows algebraic manipulation*
Model: *Now I'll try a different approach using the quadratic formula.*
EXECUTION RESULT: *Shows quadratic formula application*
Model: *Let me try yet another approach using matrix diagonalization.*
EXECUTION RESULT: *Shows complex matrix calculations*
Model: *Actually, let me go back to a simpler approach and just check all possible values directly.*
Score: 5 - The model explores several approaches, some unnecessarily complex, before settling on a more direct method.

Example 3: Severe Overthinking
INITIAL PROBLEM: In triangle ABC, points D, E, and F are on sides BC, CA, and AB respectively such that BD = 2DC, CE = 3EA, and AF = 4FB. If the area of triangle ABC is 210, find the area of triangle DEF.
Model: *I'll use barycentric coordinates to solve this problem.*
EXECUTION RESULT: *Shows complex coordinate calculations*
Model: *Now I need to compute the determinant of this transformation matrix.*
EXECUTION RESULT: *Error in calculation*
Model: *Let me try using homogeneous coordinates instead.*
EXECUTION RESULT: *More complex calculations*
Model: *Actually, I should use the menelaus theorem here.*
EXECUTION RESULT: *Even more complex approach*
Score: 8 - The model immediately jumps to advanced techniques when simpler methods (like using the fact that D, E, F divide the sides in given ratios) would be more direct.

Example 4: Overthinking with Fixation
INITIAL PROBLEM: Find the smallest positive integer n such that n! is divisible by 10^10.
Model: *I need to find how many factors of 2 and 5 are in n!*
EXECUTION RESULT: *Shows calculation of prime factorization*
Model: *Wait, I made an error. Let me recalculate.*
EXECUTION RESULT: *Shows similar calculation with small correction*
Model: *I still don't have the right answer. Let me try a completely different approach using logarithms.*
EXECUTION RESULT: *Shows logarithmic calculations*
Model: *That's not working either. Let me go back to prime factorization but be more careful.*
EXECUTION RESULT: *Shows very similar calculation to the first attempt*
Score: 7 - The model keeps switching between approaches without fully developing any of them, and returns to the same approach multiple times without significant changes.

Example 5: Overthinking with Tool Misuse
INITIAL PROBLEM: Find the sum of all positive integers n less than 1000 such that n^2 + 100 is divisible by 16.
Model: *Let me write a function to check all numbers from 1 to 999.*
EXECUTION RESULT: "Error: function takes too long to execute"
Model: *Let me try a different approach with a more efficient algorithm.*
EXECUTION RESULT: "Error: memory limit exceeded"
Model: *I'll try to solve this mathematically instead of brute force.*
EXECUTION RESULT: *Shows mathematical derivation*
Model: *Now I'll implement this more efficient solution.*
EXECUTION RESULT: "Error: syntax error in code"
Score: 6 - The model repeatedly tries approaches that don't work with the available tools, rather than adapting to a simpler approach that would work within the constraints.

Example 6: Overthinking with Failure to Use Tools
INITIAL PROBLEM: Find the remainder when 7^100 is divided by 13.
Model: *I'll calculate 7^100 directly and then find the remainder.*
EXECUTION RESULT: "Number too large to represent"
Model: *Let me try to compute this step by step.*
EXECUTION RESULT: *Shows manual calculation of powers of 7*
Model: *This is getting unwieldy. Let me try a different approach.*
EXECUTION RESULT: *More manual calculations*
Score: 9 - The model fails to use modular arithmetic or Fermat's Little Theorem, which would make this problem trivial, and instead attempts direct computation which is impractical.

Example 7: Overthinking with Failure to Respond to Feedback
INITIAL PROBLEM: Solve for x: log_2(x) + log_4(x) + log_8(x) = 7
Model: *I'll start by converting all logarithms to base 2.*
EXECUTION RESULT: *Shows conversion calculations*
Model: *Now I'll solve the resulting equation.*
EXECUTION RESULT: "Error in calculation"
Model: *Let me try again with a different approach.*
EXECUTION RESULT: "Similar error occurs"
Model: *I'll try yet another method.*
EXECUTION RESULT: "Still getting errors"
Score: 8 - The model doesn't effectively learn from the errors in its calculations and keeps trying similar approaches without addressing the fundamental issue.

Example 8: Overthinking with Multiple Unnecessary Actions
INITIAL PROBLEM: Find the value of x that satisfies 2^x = 3^(x-1).
Model: *I'll take the natural logarithm of both sides.*
EXECUTION RESULT: *Shows logarithm calculation*
Model: *Now I'll solve for x.*
EXECUTION RESULT: *Shows algebraic steps*
Model: *Let me verify this with a different method.*
EXECUTION RESULT: *Shows alternative calculation*
Model: *I'll check this with yet another approach.*
EXECUTION RESULT: *Shows third calculation method*
Model: *Let me verify numerically as well.*
EXECUTION RESULT: *Shows numerical verification*
Score: 4 - The model finds the correct answer relatively quickly but then spends unnecessary effort verifying it through multiple different methods.

Example 9: Overthinking with Failure to Recognize Patterns
INITIAL PROBLEM: Find the sum of all positive integers n such that n^2 + n + 1 is a perfect square.
Model: *I'll set n^2 + n + 1 = m^2 and solve for n in terms of m.*
EXECUTION RESULT: *Shows algebraic manipulation*
Model: *Now I need to find all integer solutions to this Diophantine equation.*
EXECUTION RESULT: *Shows complex Diophantine equation solving*
Model: *Let me try a different parameterization.*
EXECUTION RESULT: *Shows more complex calculations*
Score: 7 - The model fails to recognize that this can be approached by completing the square, which would quickly reveal that n^2 + n + 1 = (n + 1/2)^2 + 3/4, making the pattern much clearer.

Example 10: Overthinking with Multiple Tool Calls
INITIAL PROBLEM: Find the smallest positive integer that leaves remainder 1 when divided by 2, remainder 2 when divided by 3, remainder 3 when divided by 4, and remainder 4 when divided by 5.
Model: *I'll use the Chinese remainder theorem to solve this.*
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
    history: List[Dict], llm: LLM = None, output_dir: str = None, instance_id: str = None
) -> Tuple[int, str]:
    """
    Analyze the interaction history for overthinking behavior.

    Args:
        history: List of interaction events from the agent's history
        llm: LLM instance to use for analysis (optional)
        output_dir: Directory to save interaction files (optional)
        instance_id: ID of the instance (optional)

    Returns:
        Tuple[int, str]: Overthinking score and detailed analysis
    """
    # Initialize LLM if not provided
    if llm is None:
        llm = get_thinking_agent_llm()
        
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
        answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
        if answer_match:
            answer_content = answer_match.group(1).strip()
            analysis = json.loads(answer_content)
            overthinking_score = int(analysis.get('overthinking_score', 0))
            reasoning = analysis.get('reasoning', '')

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

            return overthinking_score, reasoning
        else:
            logger.warning('Could not extract answer from LLM response')
            return 0, 'Could not extract answer from LLM response'
    except Exception as e:
        logger.error(f'Error analyzing overthinking: {e}')
        return 0, str(e)


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