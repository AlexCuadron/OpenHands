import asyncio
import copy
import os
import re
from typing import Any, Optional

import pandas as pd
from datasets import load_dataset

from evaluation.benchmarks.math500.helper import (
    FAKE_RESPONSES,
    INST_SUFFIXES,
    INSTRUCTIONS_ADDENDUM,
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
    load_from_toml,
    parse_arguments,
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
    sandbox_config.base_container_image = 'python:3.11-bookworm'
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
        metadata.llm_config,
        metadata.eval_output_dir,
        str(instance.instance_id)
    )
    config.set_llm_config(llm_config)
    agent_config = config.get_agent_config(metadata.agent_class)
    agent_config.enable_prompt_extensions = False

    # copy 'draft_editor' config if exists
    config_copy = copy.deepcopy(config)
    load_from_toml(config_copy)
    if 'draft_editor' in config_copy.llms:
        config.set_llm_config(config_copy.llms['draft_editor'], 'draft_editor')

    return config


def extract_answer(text: str) -> Optional[str]:
    """Extract the answer from the agent's response."""
    # Look for answer in solution tags
    solution_pattern = r'<solution>(.*?)</solution>'
    solution_match = re.search(solution_pattern, text, re.DOTALL)
    if solution_match:
        return solution_match.group(1).strip()
    
    # Look for answer in boxed notation
    boxed_pattern = r'\\boxed{([^{}]*)}'
    boxed_match = re.search(boxed_pattern, text, re.DOTALL)
    if boxed_match:
        return boxed_match.group(0).strip()  # Return the whole match including \boxed{}
    
    # Look for "The answer is" pattern
    answer_pattern = r'[Tt]he\s+answer\s+is\s+([\d\w\s\.\-\+\/\*\^\{\}\\\(\)]+)'
    answer_match = re.search(answer_pattern, text, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    
    # Look for "Therefore" pattern
    therefore_pattern = r'[Tt]herefore,?\s+([\d\w\s\.\-\+\/\*\^\{\}\\\(\)=]+)'
    therefore_match = re.search(therefore_pattern, text, re.DOTALL)
    if therefore_match:
        return therefore_match.group(1).strip()
    
    return None


def normalize_answer(answer: str) -> str:
    """Normalize the answer for comparison."""
    # Remove LaTeX commands and whitespace
    answer = re.sub(r'\\boxed{|}\\left\(|\\right\)', '', answer)
    answer = re.sub(r'\\', '', answer)
    answer = re.sub(r'\s+', '', answer)
    return answer


def check_answer_correctness(predicted: str, reference: str) -> bool:
    """Check if the predicted answer matches the reference answer."""
    if predicted is None:
        return False
    
    # Normalize both answers
    predicted_norm = normalize_answer(predicted)
    reference_norm = normalize_answer(reference)
    
    return predicted_norm == reference_norm


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
    instruction = f"Problem: {instance.problem}\n\n"
    instruction += INSTRUCTIONS_ADDENDUM
    
    # NOTE: You can actually set slightly different instruction for different agents
    instruction += INST_SUFFIXES[metadata.agent_class]

    # =============================================
    # create sandbox and run the agent
    # =============================================

    runtime: Runtime = create_runtime(config)
    call_async_from_sync(runtime.connect)

    # Here's how you can run the agent (similar to the `main` function) and get the final task state
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

    # Extract the answer from the agent's response
    predicted_answer = None
    
    # Check if the agent used the finish tool with a solution
    finish_action = next(
        (event for event in reversed(state.history) if isinstance(event, AgentFinishAction)),
        None
    )
    
    if finish_action and hasattr(finish_action, 'solution') and finish_action.solution:
        predicted_answer = finish_action.solution
    else:
        # Extract from the last message from the agent
        last_message = next(
            (event.message for event in reversed(state.history) 
             if hasattr(event, 'message') and event.message),
            None
        )
        if last_message:
            predicted_answer = extract_answer(last_message)
    
    # Check if the answer is correct
    is_correct = check_answer_correctness(predicted_answer, instance.answer)
    
    test_result = {
        'predicted_answer': predicted_answer,
        'reference_answer': instance.answer,
        'is_correct': is_correct,
        'subject': instance.subject,
        'level': instance.level,
    }

    # history is now available as a stream of events, rather than list of pairs of (Action, Observation)
    # for compatibility with the existing output format, we can remake the pairs here
    # remove when it becomes unnecessary
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
    return output


if __name__ == '__main__':
    args = parse_arguments()
    
    # Load the MATH-500 dataset
    dataset = load_dataset('HuggingFaceH4/MATH-500')
    math500_df = dataset['test'].to_pandas()
    
    # Add instance_id if not present
    if 'instance_id' not in math500_df.columns:
        math500_df['instance_id'] = math500_df['unique_id'].apply(lambda x: x.replace('/', '_'))

    llm_config = None
    if args.llm_config:
        llm_config = get_llm_config_arg(args.llm_config)
        if llm_config is not None:
            # modify_params must be False for evaluation purpose, for reproducibility and accurancy of results
            llm_config.modify_params = False

    if llm_config is None:
        raise ValueError(f'Could not find LLM config: --llm_config {args.llm_config}')

    # Create details dictionary with agent configuration
    agent_details = {
        "agent_config": {
            "codeact_enable_jupyter": False,
            "codeact_enable_browsing": False,
            "codeact_enable_llm_editor": False,
        }
    }
    
    metadata = make_metadata(
        llm_config,
        'MATH500',
        args.agent_cls,
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