import asyncio
import copy
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import load_dataset

from evaluation.benchmarks.polyglot_aider_bench.helper.prompts import (
    INSTRUCTIONS_ADDENDUM,
    TEST_FAILURES,
)
from evaluation.utils.shared import (
    EvalMetadata,
    EvalOutput,
    compatibility_for_eval_history_pairs,
    make_metadata,
    prepare_dataset,
    reset_logger_for_multiprocessing,
    run_evaluation,
    update_llm_config_for_completions_logging,
)
from openhands.controller.state.state import State
from openhands.core.config import (
    AppConfig,
    SandboxConfig,
    get_llm_config_arg,
    parse_arguments,
)
# Override openhands logger to use our color module
import sys
from openhands.core import logger as oh_logger
from .helper.color import colored
oh_logger.colored = colored
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime, run_controller
from openhands.events.action import CmdRunAction, MessageAction
from openhands.events.observation import CmdOutputObservation
from openhands.runtime.base import Runtime
from openhands.utils.async_utils import call_async_from_sync

# Configure visibility of unit tests to the Agent.
USE_UNIT_TESTS = os.environ.get('USE_UNIT_TESTS', 'true').lower() == 'true'

# Map of file extensions to test commands
TEST_COMMANDS = {
    ".py": ["python3", "-m", "pytest"],
    ".rs": ["cargo", "test", "--", "--include-ignored"],
    ".go": ["go", "test", "./..."],
    ".js": ["npm", "test"],
    ".cpp": ["make", "test"],
    ".java": ["./gradlew", "test"],
}

def get_config(
    instance: pd.Series,
    metadata: EvalMetadata,
) -> AppConfig:
    config = AppConfig(
        default_agent=metadata.agent_class,
        run_as_openhands=False,
        runtime=os.environ.get('RUNTIME', 'docker'),
        max_iterations=metadata.max_iterations,
        sandbox=SandboxConfig(
            base_container_image='ghcr.io/opendevin/eval-polyglot:v1.0.0',  # TODO: Create this image
            enable_auto_lint=True,
            use_host_network=False,
            timeout=300,  # Longer timeout for compilation
            api_key=os.environ.get('ALLHANDS_API_KEY', None),
            remote_runtime_api_url=os.environ.get('SANDBOX_REMOTE_RUNTIME_API_URL'),
            keep_runtime_alive=False,
            remote_runtime_init_timeout=1800,
            remote_runtime_enable_retries=True,
        ),
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
    # Enable logging of LLM completions
    llm_config.log_completions = True
    config.set_llm_config(llm_config)

    agent_config = config.get_agent_config(metadata.agent_class)
    agent_config.enable_prompt_extensions = False

    return config

def initialize_runtime(
    runtime: Runtime,
    instance: pd.Series,
):
    """Initialize the runtime for the agent."""
    logger.info('-' * 30)
    logger.info('BEGIN Runtime Initialization Fn')
    logger.info('-' * 30)
    obs: CmdOutputObservation

    # Create workspace
    action = CmdRunAction(command='mkdir -p /workspace')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    assert obs.exit_code == 0

    action = CmdRunAction(command='cd /workspace')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    assert obs.exit_code == 0

    # Copy files to workspace
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy solution files
        for file_path in instance.solution_files:
            file_path = Path(file_path)
            temp_file = Path(tmpdir) / file_path.name
            with open(temp_file, 'w') as f:
                f.write(instance.solution_content[file_path.name])
            runtime.copy_to(
                str(temp_file),
                '/workspace',
            )

        # Copy test files if enabled
        if USE_UNIT_TESTS:
            for file_path in instance.test_files:
                file_path = Path(file_path)
                temp_file = Path(tmpdir) / file_path.name
                with open(temp_file, 'w') as f:
                    f.write(instance.test_content[file_path.name])
                runtime.copy_to(
                    str(temp_file),
                    '/workspace',
                )

    logger.info('-' * 30)
    logger.info('END Runtime Initialization Fn')
    logger.info('-' * 30)

def run_unit_tests(
    testdir: Path,
    test_files: List[str],
    history_fname: Path,
) -> Optional[str]:
    """Run unit tests and return error output if any."""
    timeout = 180  # 3 minutes timeout

    # Get unique file extensions from test files
    extensions = {Path(f).suffix for f in test_files}

    # Find matching test command
    command = None
    for ext in extensions:
        if ext in TEST_COMMANDS:
            command = TEST_COMMANDS[ext]
            break

    if not command:
        raise ValueError(f"No test command found for files with extensions: {extensions}")

    # Run tests
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            cwd=testdir,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        error = "Tests timed out!"
        with history_fname.open("a") as fh:
            fh.write(f"```\n{error}\n```")
        return error

    success = result.returncode == 0
    output = result.stdout

    # Clean up output
    output = output.replace(str(testdir), str(testdir.name))
    output = output.strip()

    with history_fname.open("a") as fh:
        fh.write(f"```\n{output}\n```")

    if not success:
        logger.info(f"Tests failed: {testdir}")
        return output

    return None

def complete_runtime(
    runtime: Runtime,
    instance: pd.Series,
) -> Dict[str, Any]:
    """Complete the runtime for the agent."""
    logger.info('-' * 30)
    logger.info('BEGIN Runtime Completion Fn')
    logger.info('-' * 30)

    # Run tests
    if USE_UNIT_TESTS:
        test_output = run_unit_tests(
            Path('/workspace'),
            instance.test_files,
            Path('/workspace/.aider.chat.history.md'),
        )
        exit_code = 1 if test_output else 0
    else:
        test_output = ""
        exit_code = 0

    logger.info('-' * 30)
    logger.info('END Runtime Completion Fn')
    logger.info('-' * 30)

    runtime.close()

    return {
        'test_output': test_output,
        'exit_code': exit_code,
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
    instruction = instance.instruction

    # Add file list to instruction
    file_list = " ".join(instance.solution_files)
    instruction += INSTRUCTIONS_ADDENDUM.format(file_list=file_list)

    if USE_UNIT_TESTS:
        test_files = " ".join(instance.test_files)
        logger.info(f'\nTest files: {test_files}\n')
        instruction += (
            f'Use the appropriate test command to run the tests and verify your solution. '
            'DO NOT EDIT the test files.\n\n'
        )

    instruction += (
        'IMPORTANT: You should ONLY interact with the environment provided '
        'to you AND NEVER ASK FOR HUMAN HELP.\n'
    )

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
    exit_code = return_val['exit_code']
    test_output = return_val['test_output']

    errors = []
    test_cases = None
    if test_output:
        if 'SyntaxError' in test_output:
            errors.append('SyntaxError')
        elif 'IndentationError' in test_output:
            errors.append('IndentationError')
        else:
            test_cases = test_output

    test_result = {
        'exit_code': exit_code,
        'test_cases': test_cases,
        'errors': errors,
    }

    # history is now available as a stream of events, rather than list of pairs of (Action, Observation)
    # for compatibility with the existing output format, we can remake the pairs here
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

    # Load the polyglot benchmark dataset
    dataset = load_dataset('Aider-AI/polyglot-benchmark')
    polyglot_tests = dataset['train'].to_pandas()

    llm_config = None
    if args.llm_config:
        llm_config = get_llm_config_arg(args.llm_config)
        # modify_params must be False for evaluation purpose, for reproducibility and accurancy of results
        llm_config.modify_params = False
        # Enable logging of LLM completions
        llm_config.log_completions = True

    if llm_config is None:
        raise ValueError(f'Could not find LLM config: --llm_config {args.llm_config}')

    metadata = make_metadata(
        llm_config,
        'PolyglotAiderBench',
        args.agent_cls,
        args.max_iterations,
        args.eval_note,
        args.eval_output_dir,
    )
    output_file = os.path.join(metadata.eval_output_dir, 'output.jsonl')

    # Parse dataset IDs if provided
    eval_ids = None
    if args.eval_ids:
        eval_ids = str(args.eval_ids).split(',')
        logger.info(f'\nUsing specific dataset IDs: {eval_ids}\n')

    instances = prepare_dataset(
        polyglot_tests,
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