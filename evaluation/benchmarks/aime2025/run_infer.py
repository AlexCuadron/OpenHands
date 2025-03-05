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
    if finish_action and hasattr(finish_action, 'outputs'):
        outputs = finish_action.outputs
        if isinstance(outputs, dict) and 'solution' in outputs and outputs['solution']:
            # The outputs dictionary has a solution key with a non-empty value
            possible_answers.append(outputs['solution'])
            logger.info(f'Found solution in finish action outputs: {outputs["solution"]}')

    # Method 3: Extract from finish action params dictionary
    if finish_action and hasattr(finish_action, 'params'):
        params = finish_action.params
        if isinstance(params, dict) and 'solution' in params and params['solution']:
            # The params dictionary has a solution key with a non-empty value
            possible_answers.append(params['solution'])
            logger.info(f'Found solution in finish action params: {params["solution"]}')

    # Method 4: Look for boxed expressions in the last few messages
    last_messages = [
        event.message
        for event in reversed(state.history)
        if hasattr(event, 'message') and event.message and hasattr(event, 'role') and event.role == 'assistant'
    ][:3]  # Look at the last 3 assistant messages

    for message in last_messages:
        boxed_matches = re.findall(r'\\boxed{([^}]*)}', message)
        for match in boxed_matches:
            possible_answers.append(match)
            logger.info(f'Found boxed expression: {match}')

    # Method 5: Look for "the answer is" or similar phrases
    for message in last_messages:
        answer_matches = re.findall(
            r'(?:the|final)\s+answer\s+is\s+(?:\\boxed{)?([^}]+)(?:})?',
            message,
            re.IGNORECASE,
        )
        for match in answer_matches:
            possible_answers.append(match)
            logger.info(f'Found answer in text: {match}')

    # If we found any possible answers, use the first one
    if possible_answers:
        # Normalize the answers (remove non-numeric characters)
        normalized_answers = [re.sub(r'[^0-9-]', '', str(ans)) for ans in possible_answers]
        logger.info(f'Normalized possible answers: {normalized_answers}')

        # For AIME problems, prefer answers that are just numbers
        numeric_answers = [ans for ans in normalized_answers if ans.isdigit()]
        if numeric_answers:
            predicted_answer = numeric_answers[0]
            logger.info(f'Selected numeric answer: {predicted_answer}')
        else:
            predicted_answer = possible_answers[0]
            logger.info(f'Selected first available answer: {predicted_answer}')
    else:
        predicted_answer = None
        logger.warning("Could not find any answer in the agent's response")

    # Normalize answers for comparison
    predicted_norm = predicted_answer if predicted_answer is not None else ''
    reference_norm = instance.answer if instance.answer is not None else ''
    
    # Check if the predicted answer matches the reference answer
    is_correct = predicted_norm == reference_norm
    
    # Create the test result
    test_result = {
        'predicted_answer': predicted_norm,
        'reference_answer': reference_norm,
        'is_correct': is_correct,
    }
    
    # Check if we should discard the solution due to overthinking
    if hasattr(metadata, 'overthinking_threshold') and metadata.overthinking_threshold is not None:
        try:
            # Get the overthinking threshold from metadata
            overthinking_threshold = int(metadata.overthinking_threshold)
            
            # Analyze the solution for overthinking
            overthinking_score, analysis = analyze_overthinking(state.history)
            
            # Save the analysis to a file
            overthinking_dir = os.path.join(metadata.eval_output_dir, 'overthinking_analysis')
            os.makedirs(overthinking_dir, exist_ok=True)
            
            analysis_file = os.path.join(overthinking_dir, f'instance_{instance.instance_id}.txt')
            with open(analysis_file, 'w') as f:
                f.write(f"Overthinking Score: {overthinking_score}/10\n\n")
                f.write(analysis)
            
            # Add overthinking analysis to test_result
            test_result['overthinking_score'] = overthinking_score
            test_result['overthinking_analysis'] = analysis
            
            logger.info(f"Overthinking analysis completed. Score: {overthinking_score}/10")
            logger.info(f"Overthinking analysis files saved to: {overthinking_dir}")
            
            # Check if the solution should be discarded based on the overthinking score
            if should_discard_solution(overthinking_score, int(overthinking_threshold)):
                logger.warning(f"Solution discarded due to high overthinking score: {overthinking_score} > {overthinking_threshold}")
                
                # Instead of just marking as incorrect, raise an exception to trigger a retry
                raise Exception(f"Overthinking detected with score {overthinking_score} > threshold {overthinking_threshold}. Retrying...")
            else:
                test_result['solution_discarded'] = False
        except Exception as e:
            logger.error(f"Error during overthinking analysis: {e}")
            test_result['overthinking_error'] = str(e)
    
    # Save the output
    output = EvalOutput(
        instance_id=str(instance.instance_id),
        instance=instance.to_dict(),
        instruction=instruction,
        metadata=metadata,
        history=compatibility_for_eval_history_pairs(state.history),
        metrics={
            'is_correct': is_correct,
            'predicted_answer': predicted_norm,
            'ground_truth': reference_norm,
        },
        error=state.last_error if state and state.last_error else None,
        test_result=test_result,
    )
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

    # Create agent details dictionary
    agent_details = {}
    
    # Create metadata for evaluation
    metadata = make_metadata(
        llm_config,
        'AIME2025',
        args.agent_cls,  # This is the argument name from the command line, but make_metadata expects agent_class
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
        metadata.overthinking_threshold = args.overthinking_threshold
        metadata.details['overthinking_threshold'] = args.overthinking_threshold
        logger.info(f'\nUsing overthinking threshold: {args.overthinking_threshold}\n')
    
    output_file = os.path.join(metadata.eval_output_dir, 'output.jsonl')
    
    # Parse dataset IDs if provided
    eval_ids = None
    if args.eval_ids:
        eval_ids = str(args.eval_ids).split(',')
        logger.info(f'\nUsing specific dataset IDs: {eval_ids}\n')
    
    # Prepare the dataset for evaluation
    instances = prepare_dataset(
        df,
        output_file,
        args.eval_n_limit,
        eval_ids=eval_ids,
    )
    
    # Run the evaluation
    run_evaluation(
        instances,
        metadata,
        output_file,
        args.eval_num_workers,
        process_instance,
    )


