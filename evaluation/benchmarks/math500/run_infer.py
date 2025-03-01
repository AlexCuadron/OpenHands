import asyncio
import copy
import os
import re
import argparse
from typing import Any, Optional, List

import pandas as pd
from datasets import load_dataset
import openhands.agenthub.codeact_agent.function_calling as codeact_function_calling

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
    get_parser,
)
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime, run_controller
from openhands.events.action import AgentFinishAction, MessageAction
from openhands.runtime.base import Runtime
from openhands.utils.async_utils import call_async_from_sync
import openhands.agenthub.codeact_agent.function_calling as codeact_function_calling


def get_config(
    instance: pd.Series,
    metadata: EvalMetadata,
) -> AppConfig:
    sandbox_config = get_default_sandbox_config_for_eval()
    sandbox_config.base_container_image = 'python:3.11-bookworm'
    
    # Add extra dependencies to install math libraries
    runtime_extra_deps = """
# Install math libraries
pip install --no-cache-dir sympy numpy scipy matplotlib pandas

# Create IPython startup directory and script
mkdir -p /root/.ipython/profile_default/startup
cat > /root/.ipython/profile_default/startup/00-math-imports.py << 'EOF'
import numpy as np
import sympy as sp
from sympy import symbols, solve, Eq, simplify, expand, factor, integrate, diff
from sympy import sin, cos, tan, exp, log, pi, oo
from sympy.abc import x, y, z, a, b, c, n, m
from sympy import Matrix, Rational
import matplotlib.pyplot as plt
print("Math libraries pre-loaded: numpy, sympy, scipy, matplotlib")
EOF
"""
    sandbox_config.runtime_extra_deps = runtime_extra_deps
    
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
    
    # Disable native tool calling for Together.ai models
    if llm_config and (
        llm_config.model.startswith("deepseek") or 
        (llm_config.base_url and "together.xyz" in llm_config.base_url)
    ):
        llm_config.native_tool_calling = False
        logger.info(f"Disabled native tool calling for model: {llm_config.model}")
    
    config.set_llm_config(llm_config)
    agent_config = config.get_agent_config(metadata.agent_class)
    agent_config.enable_prompt_extensions = False
    
    # For MATH500 benchmark, configure the agent with the right tools based on the allowed_tools parameter
    if metadata.agent_class == "CodeActAgent":
        # Default configuration - disable browsing
        agent_config.codeact_enable_browsing = False
        
        # Get the allowed tools from the metadata details
        allowed_tools = metadata.details.get('allowed_tools', 'all') if metadata.details else 'all'
        
        if allowed_tools == 'ipython_only':
            # Only enable IPython tool
            agent_config.codeact_enable_jupyter = True
            agent_config.codeact_enable_llm_editor = False
            # We'll override the tools after agent initialization
            if metadata.details is None:
                metadata.details = {}
            metadata.details['override_tools'] = [codeact_function_calling.IPythonTool, codeact_function_calling.FinishTool]
            logger.info(f"Configured CodeActAgent for MATH500 benchmark with IPython tool only")
        elif allowed_tools == 'bash_only':
            # Only enable Bash tool
            agent_config.codeact_enable_jupyter = False
            agent_config.codeact_enable_llm_editor = False
            # We'll override the tools after agent initialization
            if metadata.details is None:
                metadata.details = {}
            metadata.details['override_tools'] = [codeact_function_calling.CmdRunTool, codeact_function_calling.FinishTool]
            logger.info(f"Configured CodeActAgent for MATH500 benchmark with Bash tool only")
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
                codeact_function_calling.FinishTool
            ]
            logger.info(f"Configured CodeActAgent for MATH500 benchmark with Bash and IPython tools (no editor)")
        else:  # 'all' or any other value
            # Enable all tools except browsing
            agent_config.codeact_enable_jupyter = True
            agent_config.codeact_enable_llm_editor = False
            # No need to override tools
            if metadata.details is None:
                metadata.details = {}
            metadata.details['override_tools'] = None
            logger.info(f"Configured CodeActAgent for MATH500 benchmark with all tools (except browsing)")

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

    # Get the override_tools from metadata details if it exists
    override_tools = metadata.details.get('override_tools', None) if metadata.details else None
    
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
        if override_tools is not None and hasattr(state, 'agent') and hasattr(state.agent, 'tools'):
            # Override the tools
            state.agent.tools = override_tools
            logger.info(f"Overriding agent tools with: {[tool.function.name for tool in override_tools]}")
        
        return state
    
    # Here's how you can run the agent (similar to the `main` function) and get the final task state
    state: State | None = asyncio.run(custom_run_controller())
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


# Custom argument parser for MATH500 benchmark
def parse_math500_arguments():
    parser = get_parser()
    
    # Add custom argument for allowed tools
    parser.add_argument(
        '--allowed-tools',
        type=str,
        default='all',
        help='Comma-separated list of allowed tools for the agent. Options: all, ipython_only, bash_only, no_editor',
    )
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_math500_arguments()
    
    # No need to change the agent class
    
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
    
    # Add the allowed_tools parameter to the metadata details
    if metadata.details is None:
        metadata.details = {}
    metadata.details['allowed_tools'] = args.allowed_tools
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