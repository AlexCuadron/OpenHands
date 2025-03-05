import asyncio
import copy
import os
import re
from typing import Optional

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

    config = AppConfig(
        default_agent=metadata.agent_class,
        run_as_openhands=False,
        runtime=os.environ.get('RUNTIME', 'local'),  # Use local runtime instead of docker
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
    logger.info('Set temperature to 0.6 for AIME2025 benchmark')

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


def extract_answer(text: str) -> Optional[str]:
    """Extract the answer from the agent's response."""
    if not text:
        return None

    # Look for answer in solution tags
    solution_pattern = r'<solution>(.*?)</solution>'
    solution_match = re.search(solution_pattern, text, re.DOTALL)
    if solution_match:
        return solution_match.group(1).strip()

    # Look for boxed answers (common in LaTeX)
    boxed_pattern = r'\\boxed{([^{}]*)}'
    boxed_match = re.search(boxed_pattern, text, re.DOTALL)
    if boxed_match:
        return boxed_match.group(1).strip()

    # Look for "The answer is" pattern with variations
    answer_patterns = [
        r'[Tt]he\s+(?:final\s+)?answer\s+is\s+([\d\w\s\.\-\+\/\*\^\{\}\\\(\)]+)',
        r'[Tt]he\s+(?:final\s+)?answer\s+is\s*[:=]\s*([\d\w\s\.\-\+\/\*\^\{\}\\\(\)]+)',
        r'[Tt]he\s+(?:final\s+)?answer\s*[:=]\s*([\d\w\s\.\-\+\/\*\^\{\}\\\(\)]+)',
        r'[Aa]nswer\s*[:=]\s*([\d\w\s\.\-\+\/\*\^\{\}\\\(\)]+)',
        r'[Aa]nswer\s+is\s+([\d\w\s\.\-\+\/\*\^\{\}\\\(\)]+)',
    ]

    for pattern in answer_patterns:
        answer_match = re.search(pattern, text, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()

    # Look for "Therefore" pattern with variations
    therefore_patterns = [
        r'[Tt]herefore,?\s+([\d\w\s\.\-\+\/\*\^\{\}\\\(\)=]+)',
        r'[Tt]hus,?\s+([\d\w\s\.\-\+\/\*\^\{\}\\\(\)=]+)',
        r'[Ss]o,?\s+([\d\w\s\.\-\+\/\*\^\{\}\\\(\)=]+)',
        r'[Hh]ence,?\s+([\d\w\s\.\-\+\/\*\^\{\}\\\(\)=]+)',
    ]

    for pattern in therefore_patterns:
        therefore_match = re.search(pattern, text, re.DOTALL)
        if therefore_match:
            return therefore_match.group(1).strip()

    # Look for "Our answer is" pattern and variations
    our_answer_patterns = [
        r'[Oo]ur\s+answer\s+is\s+([\d\w\s\.\-\+\/\*\^\{\}\\\(\)]+)',
        r'[Ww]e\s+get\s+([\d\w\s\.\-\+\/\*\^\{\}\\\(\)=]+)',
        r'[Ww]e\s+have\s+([\d\w\s\.\-\+\/\*\^\{\}\\\(\)=]+)',
        r'[Ww]e\s+find\s+([\d\w\s\.\-\+\/\*\^\{\}\\\(\)=]+)',
        r'[Tt]his\s+gives\s+us\s+([\d\w\s\.\-\+\/\*\^\{\}\\\(\)=]+)',
    ]

    for pattern in our_answer_patterns:
        our_answer_match = re.search(pattern, text, re.DOTALL)
        if our_answer_match:
            return our_answer_match.group(1).strip()

    # Look for a standalone number at the end of the text (common in AIME problems)
    final_number_patterns = [
        r'(?:^|\n|\.)[\s\t]*(\d+)[\s\t]*$',
        r'(?:^|\n|\.)[^\d]*(\d+)[^\d]*$',
    ]

    for pattern in final_number_patterns:
        final_number_match = re.search(pattern, text)
        if final_number_match:
            return final_number_match.group(1).strip()

    # Look for a number in the last line
    last_line = text.strip().split('\n')[-1].strip()
    if last_line.isdigit():
        return last_line

    # Look for a number surrounded by special characters in the last few lines
    last_few_lines = text.strip().split('\n')[-5:]
    for line in last_few_lines:
        # Look for numbers surrounded by special formatting
        number_in_line = re.search(r'[^\d](\d+)[^\d]', line)
        if number_in_line:
            return number_in_line.group(1).strip()

    return None


def normalize_answer(answer: str) -> str:
    """Normalize the answer for comparison."""
    if answer is None:
        return ''

    # Convert to string if not already
    answer = str(answer)

    # Store the original answer for debugging
    original_answer = answer

    # Remove LaTeX commands
    answer = re.sub(r'\\boxed{(.*?)}', r'\1', answer)  # Extract content from \boxed{}
    answer = re.sub(r'\\left\(|\\right\)', '', answer)

    # Check if the answer contains mathematical expressions like sqrt
    has_math_expr = 'sqrt' in answer.lower() or '\\sqrt' in answer

    # Check if the answer contains currency symbols
    has_currency = '$' in answer or '\\$' in answer or '£' in answer or '€' in answer

    # Remove LaTeX backslashes but keep 'sqrt' intact
    answer = re.sub(r'\\sqrt', 'sqrt', answer)

    # Handle currency symbols - preserve the $ symbol for currency values
    answer = re.sub(r'\\$', '$', answer)  # Convert LaTeX \$ to $

    # Remove other LaTeX backslashes
    answer = re.sub(r'\\', '', answer)

    # Remove all whitespace
    answer = re.sub(r'\s+', '', answer)

    # Remove any text that's not part of the actual answer
    answer = re.sub(r'[Tt]he(final)?answeris', '', answer)
    answer = re.sub(r'[Tt]herefore,?', '', answer)
    answer = re.sub(r'[Tt]hus,?', '', answer)
    answer = re.sub(r'[Ss]o,?', '', answer)
    answer = re.sub(r'[Hh]ence,?', '', answer)
    answer = re.sub(r'[Oo]uranswer(is)?', '', answer)
    answer = re.sub(r'[Ww]eget', '', answer)
    answer = re.sub(r'[Ww]ehave', '', answer)
    answer = re.sub(r'[Ww]efind', '', answer)

    # Handle common mathematical notations
    answer = re.sub(r'[{}()\[\]]', '', answer)  # Remove brackets

    # Log the normalization process
    logger.debug(f"Normalizing answer: '{original_answer}' -> '{answer}'")

    # If the answer has mathematical expressions, return the normalized form without extracting numbers
    if has_math_expr:
        return answer

    # Handle currency values specially
    if has_currency:
        # Extract the full currency value (including dollars and cents)
        currency_match = re.search(r'(\$\d+\.\d+|\$\d+)', answer)
        if currency_match:
            currency_value = currency_match.group(1)
            # For comparison, keep the full value including the $ symbol
            return currency_value

    # For AIME problems with pure numbers, we typically want just the number
    # Check if the answer is purely numeric
    if re.match(r'^\d+$', answer) or re.match(r'^\d+\.\d+$', answer):
        return answer

    # First, try to extract just the number if it's the last thing in the string
    number_match = re.search(r'(\d+\.\d+|\d+)$', answer)
    if number_match:
        return number_match.group(1)

    # If that fails, try to extract any number from the string
    number_match = re.search(r'(\d+\.\d+|\d+)', answer)
    if number_match:
        return number_match.group(1)

    return answer


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
    instruction = f'Problem: {instance.question}\n\n'
    instruction += INSTRUCTIONS_ADDENDUM

    # NOTE: You can actually set slightly different instruction for different agents
    instruction += INST_SUFFIXES[metadata.agent_class]

    # =============================================
    # create sandbox and run the agent
    # =============================================

    runtime: Runtime = create_runtime(config)
    call_async_from_sync(runtime.connect)

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
        (
            event
            for event in reversed(state.history)
            if isinstance(event, AgentFinishAction)
        ),
        None,
    )

    # Try multiple methods to extract the answer
    possible_answers = []

    # Method 1: Extract from finish action solution attribute
    if finish_action and hasattr(finish_action, 'solution') and finish_action.solution:
        # The solution attribute is available and not empty
        possible_answers.append(finish_action.solution)
        logger.info(f'Found solution in finish action: {finish_action.solution}')

    # Method 2: Extract from finish action outputs dictionary
    if finish_action and hasattr(finish_action, 'outputs') and finish_action.outputs:
        if 'solution' in finish_action.outputs:
            # The solution key is available in the outputs dictionary
            possible_answers.append(finish_action.outputs['solution'])
            logger.info(
                f'Found solution in finish action outputs: {finish_action.outputs["solution"]}'
            )

    # Method 3: Extract from the last assistant message
    last_assistant_message = next(
        (
            event.message
            for event in reversed(state.history)
            if hasattr(event, 'role')
            and event.role == 'assistant'
            and hasattr(event, 'message')
        ),
        None,
    )

    if last_assistant_message:
        # Try to extract the answer from the last assistant message
        extracted_answer = extract_answer(last_assistant_message)
        if extracted_answer:
            possible_answers.append(extracted_answer)
            logger.info(f'Found answer in last assistant message: {extracted_answer}')

    # Choose the best answer from the possible answers
    if possible_answers:
        # Prefer the finish action solution if available
        if finish_action and hasattr(finish_action, 'solution') and finish_action.solution:
            predicted_answer = finish_action.solution
        # Otherwise, use the first available answer
        else:
            predicted_answer = possible_answers[0]
    else:
        logger.warning('No answer found in agent response')

    # Normalize the predicted answer for comparison
    predicted_normalized = normalize_answer(predicted_answer) if predicted_answer else None

    # Get the reference answer from the instance
    reference_answer = instance.answer
    reference_normalized = normalize_answer(reference_answer) if reference_answer else None

    # Compare the normalized answers
    is_correct = False
    comparison_method = 'string'  # Default comparison method

    if predicted_normalized and reference_normalized:
        # Try numerical comparison first if both are numeric
        try:
            # Check if both are numeric
            predicted_num = float(predicted_normalized)
            reference_num = float(reference_normalized)

            # For AIME problems, we typically want exact matches
            is_correct = predicted_num == reference_num
            comparison_method = 'numerical'
            logger.info(
                f'Numerical comparison: {predicted_num} {"==" if is_correct else "!="} {reference_num}'
            )
        except (ValueError, TypeError):
            # If numerical comparison fails, fall back to string comparison
            is_correct = predicted_normalized == reference_normalized
            logger.info(
                f'String comparison: "{predicted_normalized}" {"==" if is_correct else "!="} "{reference_normalized}"'
            )

    # Analyze overthinking if enabled
    overthinking_score = None
    solution_discarded = False
    overthinking_analysis = None

    # Check if overthinking analysis is enabled
    overthinking_threshold = (
        metadata.details.get('overthinking_threshold', None)
        if metadata.details
        else None
    )

    if overthinking_threshold is not None:
        # Get the thinking agent LLM
        thinking_agent_llm = get_thinking_agent_llm()

        # Analyze overthinking
        overthinking_score, overthinking_analysis = analyze_overthinking(
            state.history,
            thinking_agent_llm,
            metadata.eval_output_dir,
            str(instance.instance_id),
        )

        logger.info(f'Overthinking score: {overthinking_score}')

        # Check if the solution should be discarded
        solution_discarded = should_discard_solution(
            overthinking_score, overthinking_threshold
        )

        if solution_discarded:
            logger.warning(
                f'Solution discarded due to overthinking (score: {overthinking_score}, threshold: {overthinking_threshold})'
            )
            # If the solution is discarded, mark it as incorrect
            is_correct = False

    # Create the test result
    test_result = {
        'id': instance.instance_id,
        'is_correct': is_correct,
        'predicted_answer': predicted_answer,
        'reference_answer': reference_answer,
        'predicted_normalized': predicted_normalized,
        'reference_normalized': reference_normalized,
        'comparison_method': comparison_method,
    }

    # Add overthinking information if available
    if overthinking_score is not None:
        test_result['overthinking_score'] = overthinking_score
        test_result['solution_discarded'] = solution_discarded

    # Create the output
    output = EvalOutput(
        instance_id=instance.instance_id,
        history=compatibility_for_eval_history_pairs(state.history),
        test_result=test_result,
    )
    return output


# Custom argument parser for AIME2025 benchmark
def parse_aime2025_arguments():
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
        type=int,
        default=None,
        help='Threshold for overthinking score (0-10). Solutions with scores above this threshold will be discarded.',
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_aime2025_arguments()

    # Load the AIME2025 dataset
    # Combine both AIME2025-I and AIME2025-II datasets
    logger.info("Loading AIME2025 dataset...")
    try:
        dataset_i = load_dataset('opencompass/AIME2025', 'AIME2025-I')
        dataset_ii = load_dataset('opencompass/AIME2025', 'AIME2025-II')
        
        # Convert to pandas DataFrames
        aime_i_df = dataset_i['test'].to_pandas()
        aime_ii_df = dataset_ii['test'].to_pandas()
        
        # Add source information to distinguish between I and II
        aime_i_df['source'] = 'AIME2025-I'
        aime_ii_df['source'] = 'AIME2025-II'
        
        # Combine the datasets
        aime_df = pd.concat([aime_i_df, aime_ii_df], ignore_index=True)
        
        logger.info(f"Successfully loaded AIME2025 dataset with {len(aime_df)} problems")
        logger.info(f"Sample problem: {aime_df.iloc[0]['question']}")
        logger.info(f"Sample answer: {aime_df.iloc[0]['answer']}")
    except Exception as e:
        logger.error(f"Error loading AIME2025 dataset: {e}")
        # As a fallback, try loading from the local directory
        logger.info("Trying to load from local directory...")
        try:
            # Load from the local AIME2025 directory
            aime_i_path = "/workspace/OpenHands/AIME2025/aime2025-I.jsonl"
            aime_ii_path = "/workspace/OpenHands/AIME2025/aime2025-II.jsonl"
            
            aime_i_df = pd.read_json(aime_i_path, lines=True)
            aime_ii_df = pd.read_json(aime_ii_path, lines=True)
            
            # Add source information
            aime_i_df['source'] = 'AIME2025-I'
            aime_ii_df['source'] = 'AIME2025-II'
            
            # Combine the datasets
            aime_df = pd.concat([aime_i_df, aime_ii_df], ignore_index=True)
            
            logger.info(f"Successfully loaded AIME2025 dataset from local files with {len(aime_df)} problems")
            logger.info(f"Sample problem: {aime_df.iloc[0]['question']}")
            logger.info(f"Sample answer: {aime_df.iloc[0]['answer']}")
        except Exception as e2:
            logger.error(f"Error loading from local directory: {e2}")
            raise ValueError("Failed to load AIME2025 dataset")

    # Add instance_id if not present
    if 'instance_id' not in aime_df.columns:
        aime_df['instance_id'] = aime_df.index.map(lambda x: f'aime2025_{x}')
    
    # Print the dataset structure
    logger.info(f"Dataset columns: {aime_df.columns.tolist()}")
    logger.info(f"Dataset instance_id dtype: {aime_df['instance_id'].dtype}")
    logger.info(f"First 5 instance_ids: {aime_df['instance_id'].head(5).tolist()}")

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
        'agent_config': {
            'codeact_enable_jupyter': False,
            'codeact_enable_browsing': False,
            'codeact_enable_llm_editor': False,
        }
    }

    metadata = make_metadata(
        llm_config,
        'AIME2025',
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

    # Add the overthinking threshold if provided
    if args.overthinking_threshold is not None:
        metadata.details['overthinking_threshold'] = args.overthinking_threshold
        logger.info(f'\nUsing overthinking threshold: {args.overthinking_threshold}\n')

    output_file = os.path.join(metadata.eval_output_dir, 'output.jsonl')
    
    # Delete the output file if it exists to ensure we start fresh
    if os.path.exists(output_file):
        logger.info(f"Deleting existing output file: {output_file}")
        os.remove(output_file)

    # Parse dataset IDs if provided
    eval_ids = None
    if args.eval_ids:
        eval_ids = str(args.eval_ids).split(',')
        logger.info(f'\nUsing specific dataset IDs: {eval_ids}\n')
        
        # Convert eval_ids to match the instance_id format in the dataset
        if eval_ids and eval_ids[0].isdigit():
            # If eval_ids are just numbers, convert them to the aime2025_X format
            eval_ids = [f'aime2025_{id}' for id in eval_ids]
            logger.info(f"Converted eval_ids to: {eval_ids}")

    logger.info(f"Preparing dataset with {len(aime_df)} problems")
    logger.info(f"Eval limit: {args.eval_n_limit}")
    logger.info(f"Eval IDs: {eval_ids}")
    
    instances = prepare_dataset(
        aime_df,
        output_file,
        args.eval_n_limit,
        eval_ids=eval_ids,
    )
    
    logger.info(f"Prepared {len(instances)} instances for evaluation")

    run_evaluation(
        instances,
        metadata,
        output_file,
        args.eval_num_workers,
        process_instance,
    )