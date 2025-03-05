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

    # Use the default Python image
    sandbox_config.base_container_image = 'python:3.11-bookworm'

    # Add extra dependencies to install math libraries
    # This will be added to the Dockerfile
    sandbox_config.runtime_extra_deps = (
        'pip install --no-cache-dir sympy numpy scipy matplotlib pandas'
    )

    # Get the instructions
    instructions = instance['problem']

    # Add the instructions addendum
    if INSTRUCTIONS_ADDENDUM:
        instructions = f"{instructions}\n\n{INSTRUCTIONS_ADDENDUM}"

    # Add the agent-specific instructions suffix
    if metadata.agent_cls in INST_SUFFIXES:
        instructions = f"{instructions}\n\n{INST_SUFFIXES[metadata.agent_cls]}"

    config = AppConfig(
        default_agent=metadata.agent_cls,
        run_as_openhands=False,
        runtime=os.environ.get('RUNTIME', 'docker'),
        max_iterations=metadata.max_iterations,
        sandbox=sandbox_config,
        # do not mount workspace
        workspace_base=None,
        workspace_mount_path=None,
        instructions=instructions,
    )
    
    # Update llm_config to enable completions logging
    llm_config = update_llm_config_for_completions_logging(
        metadata.llm_config, metadata.eval_output_dir, str(instance.instance_id)
    )
    
    # Set temperature to 0.6 as recommended for mathematical problems
    llm_config.temperature = 0.6
    logger.info(f'Set temperature to 0.6 for AIME2025 benchmark')

    # Disable native tool calling for Together.ai models
    if llm_config and (
        llm_config.model.startswith('deepseek')
        or (llm_config.base_url and 'together.xyz' in llm_config.base_url)
    ):
        llm_config.native_tool_calling = False
        logger.info(f'Disabled native tool calling for model: {llm_config.model}')

    config.set_llm_config(llm_config)
    agent_config = config.get_agent_config(metadata.agent_cls)
    agent_config.enable_prompt_extensions = False

    # For AIME2025 benchmark, configure the agent with the right tools based on the allowed_tools parameter
    if metadata.agent_cls == 'CodeActAgent':
        # Default configuration - disable browsing
        agent_config.codeact_enable_browsing = False

        # Get the allowed tools from the metadata details
        allowed_tools = (
            metadata.details.get('allowed_tools', 'all') if hasattr(metadata, 'details') else 'all'
        )

        if allowed_tools == 'ipython_only':
            # Only enable IPython tool
            agent_config.codeact_enable_jupyter = True
            agent_config.codeact_enable_llm_editor = False
            # We'll override the tools after agent initialization
            if not hasattr(metadata, 'details'):
                metadata.details = {}
            metadata.details['override_tools'] = [
                codeact_function_calling.IPythonTool,
                codeact_function_calling.FinishTool,
            ]
            logger.info(
                'Configured CodeActAgent for AIME2025 benchmark with IPython tool only'
            )
        elif allowed_tools == 'bash_only':
            # Only enable Bash tool
            agent_config.codeact_enable_jupyter = False
            agent_config.codeact_enable_llm_editor = False
            # We'll override the tools after agent initialization
            if not hasattr(metadata, 'details'):
                metadata.details = {}
            metadata.details['override_tools'] = [
                codeact_function_calling.CmdRunTool,
                codeact_function_calling.FinishTool,
            ]
            logger.info(
                'Configured CodeActAgent for AIME2025 benchmark with Bash tool only'
            )
        elif allowed_tools == 'no_editor':
            # Enable Bash and IPython but no editor
            agent_config.codeact_enable_jupyter = True
            agent_config.codeact_enable_llm_editor = False
            # We'll override the tools after agent initialization
            if not hasattr(metadata, 'details'):
                metadata.details = {}
            metadata.details['override_tools'] = [
                codeact_function_calling.CmdRunTool,
                codeact_function_calling.IPythonTool,
                codeact_function_calling.FinishTool,
            ]
            logger.info(
                'Configured CodeActAgent for AIME2025 benchmark with Bash and IPython tools (no editor)'
            )
        else:  # 'all' or any other value
            # Enable all tools except browsing
            agent_config.codeact_enable_jupyter = True
            agent_config.codeact_enable_llm_editor = False
            # No need to override tools
            if not hasattr(metadata, 'details'):
                metadata.details = {}
            metadata.details['override_tools'] = None
            logger.info(
                'Configured CodeActAgent for AIME2025 benchmark with all tools (except browsing)'
            )

    # copy 'draft_editor' config if exists
    config_copy = copy.deepcopy(config)
    load_from_toml(config_copy)
    if 'draft_editor' in config_copy.llms:
        config.set_llm_config(config_copy.llms['draft_editor'], 'draft_editor')
        
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


def parse_aime2025_arguments():
    """Parse command-line arguments for the AIME2025 benchmark."""
    parser = get_parser()

    # Add custom argument for allowed tools
    parser.add_argument(
        '--allowed-tools',
        type=str,
        default='all',
        help='Comma-separated list of allowed tools for the agent. Options: all, ipython_only, bash_only, no_editor',
    )
    
    # Add custom argument for overthinking threshold
    parser.add_argument(
        '--overthinking-threshold',
        type=float,
        default=None,
        help='Threshold for determining overthinking (default: None)',
    )
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_aime2025_arguments()

    # Load the AIME2025 dataset
    df = load_aime2025_dataset()
    
    # Add instance_id if not present
    if 'instance_id' not in df.columns:
        df['instance_id'] = df['id'].apply(lambda x: f'aime2025_{x}')

    llm_config = None
    if args.llm_config:
        llm_config = get_llm_config_arg(args.llm_config)
        if llm_config is not None:
            # modify_params must be False for evaluation purpose, for reproducibility and accuracy of results
            llm_config.modify_params = False

    if llm_config is None:
        raise ValueError(f'Could not find LLM config: --llm_config {args.llm_config}')

    # Create metadata for evaluation
    metadata = make_metadata(args)
    metadata.llm_config = llm_config
    
    # Add details to metadata if not present
    if not hasattr(metadata, 'details'):
        metadata.details = {}
    
    # Add the allowed tools to the metadata details
    metadata.details['allowed_tools'] = args.allowed_tools
    
    # Add the overthinking threshold if provided
    if args.overthinking_threshold is not None:
        metadata.overthinking_threshold = args.overthinking_threshold
        metadata.details['overthinking_threshold'] = args.overthinking_threshold
        logger.info(f'\nUsing overthinking threshold: {args.overthinking_threshold}\n')
    
    # Parse dataset IDs if provided
    eval_ids = None
    if args.eval_ids:
        eval_ids = str(args.eval_ids).split(',')
        logger.info(f'\nUsing specific dataset IDs: {eval_ids}\n')
    
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


