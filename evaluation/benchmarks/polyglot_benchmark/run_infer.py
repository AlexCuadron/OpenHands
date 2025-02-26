import asyncio
import copy
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

# NOTE: This benchmark has been modified to use only the same tools as SWE-Bench:
# - execute_bash
# - finish
# - str_replace_editor

import pandas as pd

from evaluation.benchmarks.polyglot_benchmark.helper.prompts import (
    INSTRUCTIONS_ADDENDUM,
    INST_SUFFIXES,
    TEST_FAILURES,
    FAKE_RESPONSES,
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
    codeact_user_response,
)
from openhands.controller.state.state import State
from openhands.core.config import (
    AppConfig,
    SandboxConfig,
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

# Update fake responses with the actual function
FAKE_RESPONSES['CodeActAgent'] = codeact_user_response

def get_config(
    instance: pd.Series,
    metadata: EvalMetadata,
) -> AppConfig:
    # Determine runtime type based on environment variable
    runtime_type = os.environ.get('RUNTIME', 'docker')
    
    # Check if NO_DOCKER is set to skip Docker container creation
    if os.environ.get('NO_DOCKER', 'false').lower() == 'true':
        runtime_type = 'local'
        logger.info("Using local runtime instead of Docker due to NO_DOCKER=true")
    
    config = AppConfig(
        default_agent=metadata.agent_class,
        run_as_openhands=False,
        runtime=runtime_type,
        max_iterations=metadata.max_iterations,
        sandbox=SandboxConfig(
            base_container_image=os.environ.get('POLYGLOT_DOCKER_IMAGE', 'ghcr.io/opendevin/eval-polyglot:v1.0.0'),
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
    
    # Restrict tools to match SWE-Bench (only execute_bash, finish, and str_replace_editor)
    agent_config.codeact_enable_jupyter = False
    agent_config.codeact_enable_browsing = False
    agent_config.codeact_enable_llm_editor = False

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

def complete_runtime(
    runtime: Runtime,
    instance: pd.Series,
) -> Dict[str, Any]:
    """Complete the runtime for the agent."""
    logger.info('-' * 30)
    logger.info('BEGIN Runtime Completion Fn')
    logger.info('-' * 30)

    # Run tests
    test_output = ""
    exit_code = 1
    
    if USE_UNIT_TESTS:
        # Get unique file extensions from test files
        extensions = {Path(f).suffix for f in instance.test_files}
        
        # Find matching test command
        command = None
        for ext in extensions:
            if ext in TEST_COMMANDS:
                command = TEST_COMMANDS[ext]
                break
                
        if command:
            try:
                # Use the runtime to run the command inside the Docker container
                cmd_str = " ".join(command)
                logger.info(f"Running test command: {cmd_str}")
                
                action = CmdRunAction(command=cmd_str)
                logger.info(action, extra={'msg_type': 'ACTION'})
                
                obs = runtime.run_action(action)
                logger.info(obs, extra={'msg_type': 'OBSERVATION'})
                
                if isinstance(obs, CmdOutputObservation):
                    exit_code = obs.exit_code
                    test_output = obs.content
                else:
                    logger.error(f"Unexpected observation type: {type(obs)}")
                    exit_code = 1
                    test_output = f"Error: Unexpected observation type: {type(obs)}"
                
                # Clean up output
                test_output = test_output.replace("/workspace", "workspace")
                
                # Log test output to history file
                with tempfile.TemporaryDirectory() as tmpdir:
                    history_path = os.path.join(tmpdir, ".aider.chat.history.md")
                    with open(history_path, 'w') as f:
                        f.write(f"```\n{test_output}\n```")
                    runtime.copy_to(
                        history_path,
                        '/workspace',
                    )
                    
            except Exception as e:
                logger.error(f"Error running tests: {e}")
                test_output = f"Tests failed with error: {e}"
                exit_code = 1

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
    
    # Add agent-specific instruction suffix
    if metadata.agent_class in INST_SUFFIXES:
        instruction += INST_SUFFIXES[metadata.agent_class]

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
            fake_user_response_fn=FAKE_RESPONSES[metadata.agent_class],
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

def load_polyglot_dataset():
    """Load the polyglot benchmark dataset from the repository."""
    import glob
    import json
    import os
    from pathlib import Path
    
    # Try to find the polyglot-benchmark repository
    # First check the environment variable
    repo_path = os.environ.get('POLYGLOT_BENCHMARK_PATH')
    
    # If not set, try common locations
    if not repo_path or not os.path.exists(repo_path):
        possible_paths = [
            '/workspace/polyglot-benchmark',
            str(Path.home() / 'polyglot-benchmark'),
            str(Path.home() / 'thereal' / 'polyglot-benchmark'),
            str(Path(__file__).parent.parent.parent.parent.parent / 'polyglot-benchmark'),
            str(Path.cwd() / 'polyglot-benchmark'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                repo_path = path
                logger.info(f"Found polyglot-benchmark repository at: {repo_path}")
                break
    
    if not repo_path or not os.path.exists(repo_path):
        logger.error("Could not find polyglot-benchmark repository. Please set POLYGLOT_BENCHMARK_PATH environment variable.")
        return pd.DataFrame()
    
    all_tests = []
    instance_id = 0
    
    # Process each language directory
    for lang_dir in ['python', 'javascript', 'rust', 'go', 'cpp', 'java']:
        lang_path = os.path.join(repo_path, lang_dir, 'exercises', 'practice')
        if not os.path.exists(lang_path):
            logger.warning(f"Language directory not found: {lang_path}")
            continue
            
        # Process each exercise directory
        for exercise_dir in os.listdir(lang_path):
            exercise_path = os.path.join(lang_path, exercise_dir)
            if not os.path.isdir(exercise_path):
                continue
                
            # Check for config.json
            config_file = os.path.join(exercise_path, '.meta', 'config.json')
            if not os.path.exists(config_file):
                logger.warning(f"Config file not found: {config_file}")
                continue
                
            # Load config
            with open(config_file, 'r') as f:
                config = json.load(f)
                
            # Get solution and test files
            solution_files = config.get('files', {}).get('solution', [])
            test_files = config.get('files', {}).get('test', [])
            
            if not solution_files or not test_files:
                logger.warning(f"Missing solution or test files in {exercise_path}")
                continue
                
            # Load instructions
            instruction = ""
            intro_file = os.path.join(exercise_path, '.docs', 'introduction.md')
            if os.path.exists(intro_file):
                with open(intro_file, 'r') as f:
                    instruction += f.read() + "\n\n"
                    
            instructions_file = os.path.join(exercise_path, '.docs', 'instructions.md')
            if os.path.exists(instructions_file):
                with open(instructions_file, 'r') as f:
                    instruction += f.read() + "\n\n"
                    
            if not instruction:
                logger.warning(f"No instructions found for {exercise_path}")
                continue
                
            # Load solution and test content
            solution_content = {}
            for file_path in solution_files:
                full_path = os.path.join(exercise_path, file_path)
                if os.path.exists(full_path):
                    with open(full_path, 'r') as f:
                        solution_content[os.path.basename(file_path)] = f.read()
                        
            test_content = {}
            for file_path in test_files:
                full_path = os.path.join(exercise_path, file_path)
                if os.path.exists(full_path):
                    with open(full_path, 'r') as f:
                        test_content[os.path.basename(file_path)] = f.read()
                        
            # Create test instance
            test_instance = {
                'instance_id': instance_id,
                'instance_name': exercise_dir,
                'language': lang_dir,
                'instruction': instruction,
                'solution_files': [os.path.basename(f) for f in solution_files],
                'test_files': [os.path.basename(f) for f in test_files],
                'solution_content': solution_content,
                'test_content': test_content,
            }
            
            all_tests.append(test_instance)
            instance_id += 1
            
    return pd.DataFrame(all_tests)

def add_arguments(parser):
    """Add polyglot benchmark specific arguments to the parser."""
    parser.add_argument(
        '--eval-languages',
        type=str,
        help='Comma-separated list of languages to test (e.g., "python,javascript,rust")',
    )
    return parser

if __name__ == '__main__':
    # Get the argument parser and add custom arguments
    import argparse
    from openhands.core.config import get_parser
    
    parser = get_parser()
    add_arguments(parser)
    args = parse_arguments()
    
    # Load the polyglot benchmark dataset
    polyglot_tests = load_polyglot_dataset()
    
    if polyglot_tests.empty:
        logger.error("Failed to load polyglot benchmark dataset")
        exit(1)
        
    logger.info(f"Loaded {len(polyglot_tests)} test instances from polyglot benchmark")

    llm_config = None
    if args.llm_config:
        llm_config = get_llm_config_arg(args.llm_config)
        # modify_params must be False for evaluation purpose, for reproducibility and accuracy of results
        llm_config.modify_params = False
        # Enable logging of LLM completions
        llm_config.log_completions = True

    if llm_config is None:
        raise ValueError(f'Could not find LLM config: --llm_config {args.llm_config}')

    metadata = make_metadata(
        llm_config,
        'PolyglotBenchmark',
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
        
    # Filter by language if specified
    if hasattr(args, 'eval_languages') and args.eval_languages:
        languages = [lang.strip().lower() for lang in args.eval_languages.split(',')]
        polyglot_tests = polyglot_tests[polyglot_tests['language'].str.lower().isin(languages)]
        logger.info(f'\nFiltered to languages: {languages}, {len(polyglot_tests)} instances remaining\n')

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