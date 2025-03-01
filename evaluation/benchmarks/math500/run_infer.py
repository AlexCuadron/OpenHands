"""Run inference on the MATH-500 benchmark."""

import asyncio
import copy
import os
from typing import Optional

import pandas as pd
from datasets import load_dataset

from evaluation.benchmarks.math500.helper import (
    FAKE_RESPONSES,
    INST_SUFFIXES,
    INSTRUCTIONS_ADDENDUM,
    compare_answers,
)
from evaluation.benchmarks.math500.patch import patch_codeact_agent
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
    load_from_toml,
    parse_arguments,
)
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime, run_controller
from openhands.events.action import AgentFinishAction, CmdRunAction, MessageAction
from openhands.events.observation import CmdOutputObservation
from openhands.runtime.base import Runtime
from openhands.utils.async_utils import call_async_from_sync

# Patch the CodeActAgent to use our custom finish tool
patch_codeact_agent()


def get_config(
    instance: pd.Series,
    metadata: EvalMetadata,
) -> AppConfig:
    """Get the configuration for the agent.

    Args:
        instance: The instance to evaluate
        metadata: The evaluation metadata

    Returns:
        The agent configuration
    """
    sandbox_config = get_default_sandbox_config_for_eval()
    
    # Use a custom Docker image with math libraries pre-installed
    # If the image doesn't exist, it will be built from the Dockerfile
    math500_image = "openhands-math500:latest"
    
    # Check if the image exists, if not build it
    import subprocess
    try:
        # Check if the image exists
        result = subprocess.run(
            ["docker", "image", "inspect", math500_image], 
            capture_output=True, 
            text=True
        )
        if result.returncode != 0:
            # Image doesn't exist, build it
            dockerfile_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 
                "Dockerfile"
            )
            build_dir = os.path.dirname(dockerfile_path)
            subprocess.run(
                ["docker", "build", "-t", math500_image, build_dir],
                check=True
            )
            logger.info(f"Built Docker image {math500_image} for MATH-500 benchmark")
        else:
            logger.info(f"Using existing Docker image {math500_image} for MATH-500 benchmark")
    except Exception as e:
        logger.warning(f"Failed to build custom Docker image: {e}. Using default image.")
        math500_image = 'python:3.11-bookworm'
    
    sandbox_config.base_container_image = math500_image
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
    config.set_llm_config(llm_config)
    agent_config = config.get_agent_config(metadata.agent_class)
    agent_config.enable_prompt_extensions = False

    # copy 'draft_editor' config if exists
    config_copy = copy.deepcopy(config)
    load_from_toml(config_copy)
    if 'draft_editor' in config_copy.llms:
        config.set_llm_config(config_copy.llms['draft_editor'], 'draft_editor')

    # No need to monkey patch anymore, we're using our custom agent

    return config


def initialize_runtime(
    runtime: Runtime,
):
    """Initialize the runtime for the agent.

    This function is called before the runtime is used to run the agent.
    The Docker image already has the necessary packages installed.
    """
    logger.info(f"\n{'-' * 50} BEGIN Runtime Initialization Fn {'-' * 50}\n")
    
    # No initialization needed as the Docker image already has everything set up
    logger.info("Using pre-built Docker image with math libraries installed")
    
    logger.info(f"\n{'-' * 50} END Runtime Initialization Fn {'-' * 50}\n")


def extract_answer_from_history(state: State) -> Optional[str]:
    """Extract the answer from the agent's history.

    Args:
        state: The agent's state

    Returns:
        The extracted answer, or None if no answer was found
    """
    # First, look for a finish action with a solution
    for event in reversed(state.history):
        if isinstance(event, AgentFinishAction):
            # Check for solution parameter first
            if event.solution:
                return event.solution
            # Fall back to outputs dictionary for backward compatibility
            elif 'answer' in event.outputs:
                return event.outputs['answer']

    # If no finish action with an answer was found, return None
    return None


def process_instance(
    instance: pd.Series,
    metadata: EvalMetadata,
    reset_logger: bool = True,
) -> EvalOutput:
    """Process a single instance of the benchmark.

    Args:
        instance: The instance to evaluate
        metadata: The evaluation metadata
        reset_logger: Whether to reset the logger

    Returns:
        The evaluation output
    """
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

    # Add agent-specific instructions
    instruction += INST_SUFFIXES[metadata.agent_class]

    # =============================================
    # create sandbox and run the agent
    # =============================================

    runtime: Runtime = create_runtime(config)
    call_async_from_sync(runtime.connect)

    initialize_runtime(runtime)

    # Here's how you can run the agent and get the final task state
    state: State | None = asyncio.run(
        run_controller(
            config=config,
            initial_user_action=MessageAction(content=instruction),
            runtime=runtime,
            fake_user_response_fn=FAKE_RESPONSES[metadata.agent_class],
        )
    )
    if state is None:
        raise ValueError('State should not be None.')

    # =============================================
    # result evaluation
    # =============================================

    # Extract the answer from the agent's history
    predicted_answer = extract_answer_from_history(state)

    # Compare with the ground truth
    is_correct = False
    if predicted_answer:
        is_correct = compare_answers(predicted_answer, instance.answer)

    test_result = {
        'predicted_answer': predicted_answer,
        'reference_answer': instance.answer,
        'is_correct': is_correct,
        'subject': instance.subject,
        'level': instance.level,
    }

    # Convert history to the expected format
    histories = compatibility_for_eval_history_pairs(state.history)
    metrics = state.metrics.get() if state.metrics else None

    # Save the output
    output = EvalOutput(
        instance_id=str(instance.instance_id),
        instance=instance.to_dict(),
        instruction=instruction,
        metadata=metadata,
        history=histories,
        metrics=metrics,
        error=state.last_error if state and state.last_error else None,
        test_result=test_result,
    )

    # Close the runtime
    runtime.close()

    return output


if __name__ == '__main__':
    args = parse_arguments()

    # Load the MATH-500 dataset
    dataset = load_dataset('HuggingFaceH4/MATH-500')
    math500_df = dataset['test'].to_pandas()

    # Add instance_id column if it doesn't exist
    if 'instance_id' not in math500_df.columns:
        math500_df['instance_id'] = math500_df['unique_id'].apply(
            lambda x: x.replace('/', '_').replace('.json', '')
        )

    llm_config = None
    if args.llm_config:
        llm_config = get_llm_config_arg(args.llm_config)
        # modify_params must be False for evaluation purpose, for reproducibility and accuracy of results
        llm_config.modify_params = False

    if llm_config is None:
        raise ValueError(f'Could not find LLM config: --llm_config {args.llm_config}')

    # Create details dictionary with agent configuration
    agent_details = {
        'agent_config': {
            'codeact_enable_jupyter': True,  # Enable Jupyter for Python interpreter
            'codeact_enable_browsing': False,
            'codeact_enable_llm_editor': False,
        }
    }

    # Use the CodeActAgent with our patched finish tool
    agent_cls = 'CodeActAgent'

    metadata = make_metadata(
        llm_config,
        'MATH500',
        agent_cls,
        args.max_iterations,
        args.eval_note,
        args.eval_output_dir,
        details=agent_details,
    )
    output_file = os.path.join(metadata.eval_output_dir, 'output.jsonl')

    # Parse dataset IDs if provided
    eval_ids = None
    if args.eval_ids:
        eval_ids = str(args.eval_ids).split(',')
        logger.info(f'\nUsing specific dataset IDs: {eval_ids}\n')

    instances = prepare_dataset(
        math500_df,
        output_file,
        args.eval_n_limit,
        eval_ids=eval_ids,
    )

    run_evaluation(
        instances,
        metadata,
        output_file,
        args.eval_num_workers,
        process_instance,
    )
