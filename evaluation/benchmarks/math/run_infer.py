import asyncio
import copy
import os
import tempfile
from typing import Any

import pandas as pd
from datasets import load_dataset

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
from openhands.events.action import CmdRunAction, MessageAction
from openhands.events.observation import CmdOutputObservation
from openhands.runtime.base import Runtime
from openhands.utils.async_utils import call_async_from_sync

# Configure any environment variables
SKIP_NUM = os.environ.get('SKIP_NUM')
SKIP_NUM = (
    int(SKIP_NUM) if SKIP_NUM and SKIP_NUM.isdigit() and int(SKIP_NUM) >= 0 else None
)


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


def initialize_runtime(
    runtime: Runtime,
    instance: pd.Series,
):
    """Initialize the runtime for the agent.

    This function is called before the runtime is used to run the agent.
    """
    logger.info(f"\n{'-' * 50} BEGIN Runtime Initialization Fn {'-' * 50}\n")
    obs: CmdOutputObservation

    # Set up workspace
    action = CmdRunAction(command='mkdir -p /workspace')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    assert obs.exit_code == 0

    action = CmdRunAction(command='cd /workspace')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    assert obs.exit_code == 0

    # Create problem file
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, 'problem.txt')
        with open(file_path, 'w') as f:
            f.write(instance.problem)
        runtime.copy_to(
            file_path,
            '/workspace',
        )

    logger.info(f"\n{'-' * 50} END Runtime Initialization Fn {'-' * 50}\n")


def complete_runtime(
    runtime: Runtime,
    instance: pd.Series,
) -> dict[str, Any]:
    """Complete the runtime for the agent.

    This function is called after the agent has run.
    If you need to do something in the sandbox to get the correctness metric after
    the agent has run, modify this function.
    """
    logger.info(f"\n{'-' * 50} BEGIN Runtime Completion Fn {'-' * 50}\n")
    obs: CmdOutputObservation

    # Check if solution.txt exists
    action = CmdRunAction(command='ls -la /workspace')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    # Get the solution content
    solution_content = ""
    if "solution.txt" in obs.content:
        action = CmdRunAction(command='cat /workspace/solution.txt')
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        solution_content = obs.content

    logger.info(f"\n{'-' * 50} END Runtime Completion Fn {'-' * 50}\n")

    runtime.close()

    # For MATH problems, we need to manually evaluate the solution
    # Here we just return the solution content for manual evaluation
    return {
        'solution': solution_content,
        'correct_answer': instance.answer,
    }


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
    instruction = f"""You are given a mathematics problem to solve. The problem is in the file 'problem.txt'.

Please read the problem carefully and solve it step by step. Write your solution in a file named 'solution.txt'.

Your solution should include:
1. A clear understanding of the problem
2. Step-by-step working
3. The final answer

IMPORTANT: You should ONLY interact with the environment provided to you AND NEVER ASK FOR HUMAN HELP.
"""

    # =============================================
    # create sandbox and run the agent
    # =============================================

    runtime: Runtime = create_runtime(config)
    call_async_from_sync(runtime.connect)

    initialize_runtime(runtime, instance=instance)

    # Here's how you can run the agent (similar to the `main` function) and get the final task state
    state: State | None = asyncio.run(
        run_controller(
            config=config,
            initial_user_action=MessageAction(content=instruction),
            runtime=runtime,
        )
    )
    if state is None:
        raise ValueError('State should not be None.')

    # =============================================
    # result evaluation
    # =============================================

    return_val = complete_runtime(runtime, instance)
    solution = return_val['solution']
    correct_answer = return_val['correct_answer']

    # Simple evaluation - check if the correct answer appears in the solution
    # In a real implementation, you would need a more sophisticated evaluation
    is_correct = correct_answer in solution

    test_result = {
        'solution': solution,
        'correct_answer': correct_answer,
        'is_correct': is_correct,
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


def prepare_math_dataset():
    """Prepare the MATH dataset for evaluation."""
    # In a real implementation, you would load the MATH dataset
    # For now, we'll create a simple mock dataset
    data = {
        'instance_id': list(range(10)),
        'problem': [
            "Find the value of x in the equation 2x + 3 = 7.",
            "Solve for y: 3y - 5 = 10.",
            "Calculate the area of a circle with radius 5 cm.",
            "Find the derivative of f(x) = x^2 + 3x + 2.",
            "Solve the system of equations: 2x + y = 5, x - y = 1.",
            "Find the indefinite integral of g(x) = 2x + 3.",
            "Calculate the limit of (x^2 - 1)/(x - 1) as x approaches 1.",
            "Find the value of sin(30°) + cos(60°).",
            "Solve the quadratic equation x^2 - 5x + 6 = 0.",
            "Find the sum of the first 10 terms of the arithmetic sequence with a_1 = 3 and d = 2."
        ],
        'answer': [
            "x = 2",
            "y = 5",
            "78.54 cm²",
            "f'(x) = 2x + 3",
            "x = 2, y = 1",
            "∫(2x + 3)dx = x² + 3x + C",
            "2",
            "1",
            "x = 2, x = 3",
            "75"
        ],
        'level': ['Algebra'] * 10,
        'type': ['Equation'] * 5 + ['Calculus'] * 3 + ['Equation'] * 2
    }
    
    return pd.DataFrame(data)


if __name__ == '__main__':
    args = parse_arguments()
    
    # Prepare the MATH dataset
    math_dataset = prepare_math_dataset()

    llm_config = None
    if args.llm_config:
        llm_config = get_llm_config_arg(args.llm_config)
        # modify_params must be False for evaluation purpose, for reproducibility and accuracy of results
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
        'MATH',
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
        math_dataset,
        output_file,
        args.eval_n_limit,
        eval_ids=eval_ids,
        skip_num=SKIP_NUM,
    )

    run_evaluation(
        instances,
        metadata,
        output_file,
        args.eval_num_workers,
        process_instance,
    )