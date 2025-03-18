import asyncio
import copy
import json
import os
import tempfile
from typing import Any

import pandas as pd
import requests

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

    # Create question file
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, 'question.txt')
        with open(file_path, 'w') as f:
            f.write(instance.question)
        runtime.copy_to(
            file_path,
            '/workspace',
        )

        # Create context files
        for i, context in enumerate(instance.context):
            file_path = os.path.join(tmpdir, f'context_{i}.txt')
            with open(file_path, 'w') as f:
                f.write(context)
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

    # Check if answer.txt exists
    action = CmdRunAction(command='ls -la /workspace')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    # Get the answer content
    answer_content = ""
    if "answer.txt" in obs.content:
        action = CmdRunAction(command='cat /workspace/answer.txt')
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        answer_content = obs.content

    logger.info(f"\n{'-' * 50} END Runtime Completion Fn {'-' * 50}\n")

    runtime.close()

    # For HotpotQA, we need to evaluate the answer against the ground truth
    # Here we just return the answer content for evaluation
    return {
        'answer': answer_content,
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
    instruction = f"""You are given a question and some context documents to help you answer it. The question is in the file 'question.txt'.

The context documents are in files named 'context_0.txt', 'context_1.txt', etc. You should read all the context files to gather information needed to answer the question.

Please write your answer in a file named 'answer.txt'. Your answer should be concise and directly address the question.

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
    answer = return_val['answer']
    correct_answer = return_val['correct_answer']

    # Simple evaluation - check if the answer matches the correct answer
    # In a real implementation, you would need a more sophisticated evaluation
    is_correct = answer.strip().lower() == correct_answer.strip().lower()

    test_result = {
        'answer': answer,
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


def prepare_hotpotqa_dataset():
    """Prepare the HotpotQA dataset for evaluation."""
    # In a real implementation, you would download and process the HotpotQA dataset
    # For now, we'll create a simple mock dataset
    data = {
        'instance_id': list(range(10)),
        'question': [
            "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
            "Were Scott Derrickson and Ed Wood of the same nationality?",
            "What is the name of the professional wrestler who had a role in the film The Princess Bride?",
            "Which magazine was started first Arthur's Magazine or First for Women?",
            "What city was the birthplace of the actor who played Humpty Dumpty in the 2010 adaptation of Alice in Wonderland?",
            "What is the difference in years between the release of The Innocents and The Others?",
            "What is the name of the actor who played the character Wolverine in the X-Men film series?",
            "Which country is the birthplace of the actor who played James Bond in the film Skyfall?",
            "What is the name of the director who directed the film Inception?",
            "Which film won more Academy Awards, The Lord of the Rings: The Return of the King or Titanic?"
        ],
        'context': [
            ["Shirley Temple was an American actress, singer, dancer, businesswoman, and diplomat who was Hollywood's number one box-office draw as a child actress from 1935 to 1938.", "Shirley Temple Black (April 23, 1928 – February 10, 2014) was an American actress, singer, dancer, businesswoman, and diplomat who was Hollywood's number one box-office draw as a child actress from 1935 to 1938. As an adult, she was named United States ambassador to Ghana and to Czechoslovakia, and also served as Chief of Protocol of the United States."],
            ["Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer. He lives in Los Angeles, California. He is best known for directing horror films such as Sinister, The Exorcism of Emily Rose, and Deliver Us From Evil, as well as the 2016 Marvel Cinematic Universe installment, Doctor Strange.", "Edward Davis Wood Jr. (October 10, 1924 – December 10, 1978) was an American filmmaker, actor, writer, producer, and director."],
            ["André René Roussimoff (May 19, 1946 – January 27, 1993), best known as André the Giant, was a French professional wrestler and actor.", "The Princess Bride is a 1987 American fantasy comedy film directed and co-produced by Rob Reiner, starring Cary Elwes, Robin Wright, Mandy Patinkin, Chris Sarandon, Wallace Shawn, André the Giant, and Christopher Guest."],
            ["Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century.", "First for Women is a woman's magazine published by Bauer Media Group in the USA. The magazine was started in 1989."],
            ["Sir Sydney Smirke RA (20 October 1798 – 8 December 1877) was a British architect who was born in London, England, the younger brother of Sir Robert Smirke, also an architect. Their father, also Robert Smirke, was a well-known painter.", "Alice in Wonderland is a 2010 American dark fantasy adventure film directed by Tim Burton from a screenplay written by Linda Woolverton."],
            ["The Innocents is a 1961 British supernatural gothic horror film directed and produced by Jack Clayton, and starring Deborah Kerr, Michael Redgrave, and Megs Jenkins.", "The Others (Spanish: Los Otros) is a 2001 English-language Spanish gothic supernatural psychological horror film written, directed, and scored by Alejandro Amenábar."],
            ["Hugh Michael Jackman (born 12 October 1968) is an Australian actor, singer, and producer.", "Wolverine is a fictional character appearing in American comic books published by Marvel Comics, mostly in association with the X-Men."],
            ["Daniel Wroughton Craig (born 2 March 1968) is an English actor.", "Skyfall is a 2012 spy film and the twenty-third in the James Bond series produced by Eon Productions."],
            ["Christopher Edward Nolan CBE (born 30 July 1970) is a British-American film director, producer, and screenwriter.", "Inception is a 2010 science fiction action film written and directed by Christopher Nolan, who also produced the film with his wife, Emma Thomas."],
            ["The Lord of the Rings: The Return of the King is a 2003 epic fantasy adventure film directed by Peter Jackson, based on the third volume of J. R. R. Tolkien's The Lord of the Rings.", "Titanic is a 1997 American epic romance and disaster film directed, written, co-produced, and co-edited by James Cameron."]
        ],
        'answer': [
            "United States ambassador",
            "Yes",
            "André the Giant",
            "Arthur's Magazine",
            "London",
            "40 years",
            "Hugh Jackman",
            "England",
            "Christopher Nolan",
            "The Lord of the Rings: The Return of the King"
        ]
    }
    
    return pd.DataFrame(data)


if __name__ == '__main__':
    args = parse_arguments()
    
    # Prepare the HotpotQA dataset
    hotpotqa_dataset = prepare_hotpotqa_dataset()

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
        'HotpotQA',
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
        hotpotqa_dataset,
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