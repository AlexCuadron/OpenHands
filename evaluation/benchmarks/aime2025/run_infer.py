import asyncio
import copy
import os
import re
from typing import Optional, Dict, List, Any

import pandas as pd
from datasets import load_dataset

import openhands.agenthub.codeact_agent.function_calling as codeact_function_calling
from evaluation.benchmarks.aime2025.helper import (
    FAKE_RESPONSES,
    INST_SUFFIXES,
    INSTRUCTIONS_ADDENDUM,
)
from evaluation.benchmarks.aime2025.thinking_agent import (
    analyze_overthinking,
    get_thinking_agent_llm,
    should_discard_solution,
)
from evaluation.utils.shared import (
    EvalMetadata,
    EvalOutput,
    compatibility_for_eval_history_pairs,
    get_default_sandbox_config_for_eval,
    make_metadata,
    prepare_dataset,
    reset_logger_for_multiprocessing,
    run_evaluation,
    update_llm_config_for_completions_logging,
)
from openhands.controller.state.state import State
from openhands.core.config import (
    AppConfig,
    get_llm_config_arg,
    get_parser,
    load_from_toml,
)
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime, run_controller
from openhands.events.action import AgentFinishAction, MessageAction
from openhands.runtime.base import Runtime
from openhands.utils.async_utils import call_async_from_sync


def get_config(
    instance: pd.Series,
    metadata: EvalMetadata,
) -> AppConfig:
    sandbox_config = get_default_sandbox_config_for_eval()

    # Get the LLM config
    llm_config = get_llm_config_arg(metadata.llm_config)
    if metadata.log_completions:
        llm_config = update_llm_config_for_completions_logging(llm_config)

    # Get the agent class
    agent_cls = metadata.agent_cls

    # Get the instructions
    instructions = instance['problem']

    # Add the instructions addendum
    if INSTRUCTIONS_ADDENDUM:
        instructions = f"{instructions}\n\n{INSTRUCTIONS_ADDENDUM}"

    # Add the agent-specific instructions suffix
    if agent_cls in INST_SUFFIXES:
        instructions = f"{instructions}\n\n{INST_SUFFIXES[agent_cls]}"

    # Create the config
    config = AppConfig(
        llm=llm_config,
        agent=agent_cls,
        sandbox=sandbox_config,
        instructions=instructions,
        allowed_tools=metadata.allowed_tools,
    )

    return config


def extract_answer_from_finish_action(
    history: List[Dict], default_answer: str = ''
) -> str:
    """
    Extract the answer from the finish action in the history.

    Args:
        history: List of interaction events from the agent's history
        default_answer: Default answer to return if no finish action is found

    Returns:
        str: The extracted answer
    """
    # Find the finish action in the history
    finish_action = next(
        (
            event
            for event in reversed(history)
            if hasattr(event, 'action') and event.action == 'finish'
        ),
        None,
    )

    # If a finish action is found, extract the solution
    if finish_action and hasattr(finish_action, 'params'):
        solution = finish_action.params.get('solution', default_answer)
        # Clean up the solution (remove non-numeric characters)
        solution = re.sub(r'[^0-9-]', '', str(solution))
        return solution

    return default_answer


def extract_answer_from_boxed(history: List[Dict], default_answer: str = '') -> str:
    """
    Extract the answer from a boxed expression in the history.

    Args:
        history: List of interaction events from the agent's history
        default_answer: Default answer to return if no boxed expression is found

    Returns:
        str: The extracted answer
    """
    # Find messages in the history
    messages = [
        event.message
        for event in history
        if hasattr(event, 'message') and event.message
    ]

    # Look for boxed expressions in the messages
    for message in reversed(messages):
        boxed_match = re.search(r'\\boxed{([^}]*)}', message)
        if boxed_match:
            boxed_content = boxed_match.group(1)
            # Clean up the boxed content (remove non-numeric characters)
            boxed_content = re.sub(r'[^0-9-]', '', boxed_content)
            return boxed_content

    return default_answer


def extract_answer_from_text(history: List[Dict], default_answer: str = '') -> str:
    """
    Extract the answer from text in the history.

    Args:
        history: List of interaction events from the agent's history
        default_answer: Default answer to return if no answer is found in text

    Returns:
        str: The extracted answer
    """
    # Find messages in the history
    messages = [
        event.message
        for event in history
        if hasattr(event, 'message') and event.message
    ]

    # Look for "the answer is" or "final answer is" in the messages
    for message in reversed(messages):
        answer_match = re.search(
            r'(?:the|final)\s+answer\s+is\s+(?:\\boxed{)?([0-9-]+)(?:})?',
            message,
            re.IGNORECASE,
        )
        if answer_match:
            return answer_match.group(1)

    return default_answer


def extract_answer(history: List[Dict], ground_truth: str) -> str:
    """
    Extract the answer from the history using multiple methods.

    Args:
        history: List of interaction events from the agent's history
        ground_truth: The ground truth answer

    Returns:
        str: The extracted answer
    """
    # Try to extract the answer from the finish action
    answer = extract_answer_from_finish_action(history)
    if answer:
        return answer

    # Try to extract the answer from a boxed expression
    answer = extract_answer_from_boxed(history)
    if answer:
        return answer

    # Try to extract the answer from text
    answer = extract_answer_from_text(history)
    if answer:
        return answer

    # If no answer is found, return an empty string
    return ''


def evaluate_answer(
    predicted_answer: str, ground_truth: str, history: List[Dict], metadata: EvalMetadata
) -> Dict[str, Any]:
    """
    Evaluate the predicted answer against the ground truth.

    Args:
        predicted_answer: The predicted answer
        ground_truth: The ground truth answer
        history: List of interaction events from the agent's history
        metadata: Evaluation metadata

    Returns:
        Dict[str, Any]: Evaluation results
    """
    # Clean up the ground truth (remove non-numeric characters)
    ground_truth = re.sub(r'[^0-9-]', '', str(ground_truth))

    # Check if the predicted answer matches the ground truth
    is_correct = predicted_answer == ground_truth

    # Check if the solution should be discarded due to overthinking
    should_discard = False
    discard_reason = ''
    if metadata.overthinking_threshold is not None:
        should_discard, discard_reason = should_discard_solution(
            history, metadata.overthinking_threshold
        )

    # Create the evaluation results
    results = {
        'predicted_answer': predicted_answer,
        'ground_truth': ground_truth,
        'is_correct': is_correct,
        'should_discard': should_discard,
        'discard_reason': discard_reason,
    }

    return results


async def run_instance(
    instance: pd.Series,
    metadata: EvalMetadata,
    runtime: Optional[Runtime] = None,
) -> Dict[str, Any]:
    """
    Run a single instance of the AIME2025 benchmark.

    Args:
        instance: The instance to run
        metadata: Evaluation metadata
        runtime: Optional runtime to use

    Returns:
        Dict[str, Any]: Results of the run
    """
    # Reset the logger for multiprocessing
    reset_logger_for_multiprocessing()

    # Get the config for this instance
    config = get_config(instance, metadata)

    # Create a runtime if one is not provided
    if runtime is None:
        runtime = await create_runtime(config)

    # Create a state for the controller
    state = State()

    # Add the fake response function if available
    if metadata.agent_cls in FAKE_RESPONSES:
        state.fake_response_fn = FAKE_RESPONSES[metadata.agent_cls]

    # Run the controller
    await run_controller(runtime, state, config)

    # Extract the answer from the history
    predicted_answer = extract_answer(state.history, instance['answer'])

    # Evaluate the answer
    eval_results = evaluate_answer(
        predicted_answer, instance['answer'], state.history, metadata
    )

    # Create the output
    output = {
        'instance_id': instance['id'],
        'problem': instance['problem'],
        'predicted_answer': eval_results['predicted_answer'],
        'ground_truth': eval_results['ground_truth'],
        'is_correct': eval_results['is_correct'],
        'should_discard': eval_results['should_discard'],
        'discard_reason': eval_results['discard_reason'],
        'history': compatibility_for_eval_history_pairs(state.history),
    }

    return output


def load_aime2025_dataset() -> pd.DataFrame:
    """
    Load the AIME2025 dataset.

    Returns:
        pd.DataFrame: The AIME2025 dataset
    """
    # Load the dataset from Hugging Face
    dataset = load_dataset("yentinglin/aime_2025")
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(dataset['train'])
    
    return df


def main():
    """Main function for running the AIME2025 benchmark."""
    # Get the parser
    parser = get_parser()
    
    # Add benchmark-specific arguments
    parser.add_argument(
        '--overthinking-threshold',
        type=float,
        default=None,
        help='Threshold for determining overthinking (default: None)',
    )
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Create the metadata
    metadata = make_metadata(args)
    
    # Add the overthinking threshold to the metadata
    metadata.overthinking_threshold = args.overthinking_threshold
    
    # Load the dataset
    df = load_aime2025_dataset()
    
    # Prepare the dataset for evaluation
    df = prepare_dataset(df, args)
    
    # Run the evaluation
    call_async_from_sync(
        run_evaluation,
        df,
        metadata,
        run_instance,
        'AIME2025',
    )


if __name__ == '__main__':
    main()