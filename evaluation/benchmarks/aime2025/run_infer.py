import asyncio
import copy
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import load_dataset

import openhands.agenthub.codeact_agent.function_calling as codeact_function_calling
from evaluation.benchmarks.aime2025.aime2025_llm_patch import patch_llm_for_aime2025_benchmark
from evaluation.benchmarks.aime2025.helper import (
    FAKE_RESPONSES,
    INST_SUFFIXES,
    INSTRUCTIONS_ADDENDUM,
    USE_PREFIX_FOR_ASSISTANT,
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

    config = AppConfig(
        default_agent=metadata.agent_class,
        run_as_openhands=False,
        runtime=os.environ.get('RUNTIME', 'docker'),
        max_iterations=metadata.max_iterations,
        sandbox=sandbox_config,
        # do not mount workspace
        workspace_base=None,
        workspace_mount_path=None,
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
    agent_config = config.get_agent_config(metadata.agent_class)
    agent_config.enable_prompt_extensions = False

    # For AIME2025 benchmark, configure the agent with the right tools based on the allowed_tools parameter
    if metadata.agent_class == 'CodeActAgent':
        # Default configuration - disable browsing
        agent_config.codeact_enable_browsing = False

        # Get the allowed tools from the metadata details
        allowed_tools = (
            metadata.details.get('allowed_tools', 'all') if metadata.details else 'all'
        )

        if allowed_tools == 'ipython_only':
            # Only enable IPython tool
            agent_config.codeact_enable_jupyter = True
            agent_config.codeact_enable_llm_editor = False
            # We'll override the tools after agent initialization
            if metadata.details is None:
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
            if metadata.details is None:
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
            if metadata.details is None:
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
            if metadata.details is None:
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


def process_instance(
    instance: pd.Series,
    metadata: EvalMetadata,
    reset_logger: bool = True,
) -> EvalOutput:
    config = get_config(instance, metadata)

    # Setup the logger properly, so you can run multi-processing to parallelize the evaluation
    if reset_logger:
        log_dir = os.path.join(metadata.eval_output_dir, 'infer_logs')
        reset_logger_for_multiprocessing(logger, str(instance.instance_id), log_dir)
    else:
        logger.info(
            f'\nStarting evaluation for instance {str(instance.instance_id)}.\n'
        )

    # =============================================
    # build instruction
    # =============================================

    # Prepare instruction
    logger.info(instance)
    instruction = f'Problem: {instance.problem}\n\n'
    instruction += INSTRUCTIONS_ADDENDUM

    # NOTE: You can actually set slightly different instruction for different agents
    instruction += INST_SUFFIXES[metadata.agent_class]

    # =============================================
    # create sandbox and run the agent
    # =============================================

    runtime: Runtime = create_runtime(config)
    call_async_from_sync(runtime.connect)

    # Apply the AIME2025 LLM patch
    logger.info('Applying AIME2025 LLM patch')
    
    # Get the override_tools from metadata details if it exists
    override_tools = (
        metadata.details.get('override_tools', None) if metadata.details else None
    )

    # Define a custom run_controller function that overrides the tools if needed
    async def custom_run_controller():
        # Run the controller normally
        state = await run_controller(
            config=config,
            initial_user_action=MessageAction(content=instruction),
            runtime=runtime,
            fake_user_response_fn=FAKE_RESPONSES[metadata.agent_class],
        )

        # If we need to override the tools, do it after the agent is initialized
        if (
            override_tools is not None
            and hasattr(state, 'agent')
            and hasattr(state.agent, 'tools')
        ):
            # Override the tools
            state.agent.tools = override_tools
            logger.info(
                f'Overriding agent tools with: {[tool.function.name for tool in override_tools]}'
            )

        # Apply the AIME2025 LLM patch
        if hasattr(state, 'agent') and hasattr(state.agent, 'llm'):
            logger.info('Applying AIME2025 LLM patch')
            patch_llm_for_aime2025_benchmark(state)
            logger.info('AIME2025 LLM patch applied')

        return state

    # Here's how you can run the agent (similar to the `main` function) and get the final task state
    state: State | None = asyncio.run(custom_run_controller())

    # =============================================
    # extract answer and evaluate
    # =============================================

    # Extract the answer from the agent's history
    predicted_answer = extract_answer(state.history, instance.answer)

    # Evaluate the answer
    results = evaluate_answer(predicted_answer, instance.answer, state.history, metadata)

    # Create the evaluation output
    output = EvalOutput(
        instance_id=instance.instance_id,
        history=compatibility_for_eval_history_pairs(state.history),
        results=results,
    )

    return output


def run_infer(
    llm_config_name: str,
    llm_config_hash: str,
    agent_class: str,
    eval_n_limit: int,
    eval_ids: list[str] | None = None,
    skip_num: int | None = None,
    allowed_tools: str = 'all',
    overthinking_threshold: float | None = None,
    use_mp: bool = False,
    mp_workers: int = 4,
    output_dir: str | None = None,
    data_split: str | None = None,
):
    """
    Run inference on the AIME2025 dataset.

    Args:
        llm_config_name: The name of the LLM config to use
        llm_config_hash: The hash of the LLM config to use
        agent_class: The agent class to use
        eval_n_limit: The number of instances to evaluate
        eval_ids: Optional list of instance IDs to evaluate
        skip_num: Optional number of instances to skip
        allowed_tools: The tools to allow the agent to use
        overthinking_threshold: Optional threshold for overthinking detection
        use_mp: Whether to use multiprocessing
        mp_workers: The number of workers to use for multiprocessing
        output_dir: Optional output directory
        data_split: Optional data split to use
    """
    # Load the dataset
    dataset = load_dataset('AlexCuadron/AIME2025', split='train').to_pandas()

    # Create the metadata
    llm_config = get_llm_config_arg(llm_config_name, llm_config_hash)
    metadata = make_metadata(
        agent_class=agent_class,
        llm_config=llm_config,
        max_iterations=30,
        dataset='AIME2025',
        data_split=data_split,
        details={'allowed_tools': allowed_tools},
        overthinking_threshold=overthinking_threshold,
        output_dir=output_dir,
    )

    # Prepare the dataset
    output_file = os.path.join(metadata.eval_output_dir, 'output.jsonl')
    dataset = prepare_dataset(dataset, output_file, eval_n_limit, eval_ids, skip_num)

    # Run the evaluation
    run_evaluation(
        dataset=dataset,
        metadata=metadata,
        process_instance_func=process_instance,
        output_file=output_file,
        use_mp=use_mp,
        mp_workers=mp_workers,
    )


if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument(
        '--llm-config-hash',
        type=str,
        default='HEAD',
        help='The hash of the LLM config to use',
    )
    parser.add_argument(
        '--allowed_tools',
        type=str,
        default='all',
        choices=['all', 'ipython_only', 'bash_only', 'no_editor'],
        help='The tools to allow the agent to use',
    )
    parser.add_argument(
        '--overthinking_threshold',
        type=float,
        default=None,
        help='The threshold for overthinking detection',
    )
    parser.add_argument(
        '--skip-num',
        type=int,
        default=None,
        help='The number of instances to skip',
    )
    parser.add_argument(
        '--use-mp',
        action='store_true',
        help='Whether to use multiprocessing',
    )
    parser.add_argument(
        '--data-split',
        type=str,
        default=None,
        help='The data split to use',
    )
    args = parser.parse_args()

    run_infer(
        llm_config_name=args.llm_config,
        llm_config_hash=args.llm_config_hash,
        agent_class=args.agent_cls,
        eval_n_limit=args.eval_n_limit,
        eval_ids=args.eval_ids,
        skip_num=args.skip_num,
        allowed_tools=args.allowed_tools,
        overthinking_threshold=args.overthinking_threshold,
        use_mp=args.use_mp,
        mp_workers=args.eval_num_workers,
        output_dir=args.eval_output_dir,
        data_split=args.data_split,
    )