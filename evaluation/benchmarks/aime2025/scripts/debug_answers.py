#!/usr/bin/env python3
"""
Script to debug the answers extracted from an AIME2025 benchmark run.
"""

import argparse
import json
import os
import re
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd


def load_results(results_file: str) -> List[Dict[str, Any]]:
    """
    Load the results from a JSONL file.

    Args:
        results_file: Path to the results file

    Returns:
        List[Dict[str, Any]]: List of result dictionaries
    """
    results = []
    with open(results_file, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return results


def extract_answer_from_finish_action(
    history: List[Dict], default_answer: str = ''
) -> Tuple[str, str]:
    """
    Extract the answer from the finish action in the history.

    Args:
        history: List of interaction events from the agent's history
        default_answer: Default answer to return if no finish action is found

    Returns:
        Tuple[str, str]: The extracted answer and the source
    """
    # Find the finish action in the history
    finish_action = next(
        (
            event
            for event in reversed(history)
            if event.get('action') == 'finish'
        ),
        None,
    )

    # If a finish action is found, extract the solution
    if finish_action and 'params' in finish_action:
        solution = finish_action['params'].get('solution', default_answer)
        # Clean up the solution (remove non-numeric characters)
        solution = re.sub(r'[^0-9-]', '', str(solution))
        return solution, 'finish_action'

    return default_answer, ''


def extract_answer_from_boxed(
    history: List[Dict], default_answer: str = ''
) -> Tuple[str, str]:
    """
    Extract the answer from a boxed expression in the history.

    Args:
        history: List of interaction events from the agent's history
        default_answer: Default answer to return if no boxed expression is found

    Returns:
        Tuple[str, str]: The extracted answer and the source
    """
    # Find messages in the history
    messages = [
        event.get('message', '')
        for event in history
        if event.get('role') == 'assistant' and 'message' in event
    ]

    # Look for boxed expressions in the messages
    for message in reversed(messages):
        boxed_match = re.search(r'\\boxed{([^}]*)}', message)
        if boxed_match:
            boxed_content = boxed_match.group(1)
            # Clean up the boxed content (remove non-numeric characters)
            boxed_content = re.sub(r'[^0-9-]', '', boxed_content)
            return boxed_content, 'boxed'

    return default_answer, ''


def extract_answer_from_text(
    history: List[Dict], default_answer: str = ''
) -> Tuple[str, str]:
    """
    Extract the answer from text in the history.

    Args:
        history: List of interaction events from the agent's history
        default_answer: Default answer to return if no answer is found in text

    Returns:
        Tuple[str, str]: The extracted answer and the source
    """
    # Find messages in the history
    messages = [
        event.get('message', '')
        for event in history
        if event.get('role') == 'assistant' and 'message' in event
    ]

    # Look for "the answer is" or "final answer is" in the messages
    for message in reversed(messages):
        answer_match = re.search(
            r'(?:the|final)\s+answer\s+is\s+(?:\\boxed{)?([0-9-]+)(?:})?',
            message,
            re.IGNORECASE,
        )
        if answer_match:
            return answer_match.group(1), 'text'

    return default_answer, ''


def extract_answer(
    history: List[Dict], ground_truth: str
) -> Dict[str, Any]:
    """
    Extract the answer from the history using multiple methods.

    Args:
        history: List of interaction events from the agent's history
        ground_truth: The ground truth answer

    Returns:
        Dict[str, Any]: The extracted answers and sources
    """
    # Try to extract the answer from the finish action
    finish_answer, finish_source = extract_answer_from_finish_action(history)
    
    # Try to extract the answer from a boxed expression
    boxed_answer, boxed_source = extract_answer_from_boxed(history)
    
    # Try to extract the answer from text
    text_answer, text_source = extract_answer_from_text(history)
    
    # Determine the final answer
    final_answer = ''
    final_source = ''
    
    if finish_answer:
        final_answer = finish_answer
        final_source = finish_source
    elif boxed_answer:
        final_answer = boxed_answer
        final_source = boxed_source
    elif text_answer:
        final_answer = text_answer
        final_source = text_source
    
    # Check if the final answer matches the ground truth
    is_correct = final_answer == ground_truth
    
    # Create the result dictionary
    result = {
        'finish_answer': finish_answer,
        'boxed_answer': boxed_answer,
        'text_answer': text_answer,
        'final_answer': final_answer,
        'final_source': final_source,
        'ground_truth': ground_truth,
        'is_correct': is_correct,
    }
    
    return result


def main():
    """Main function for debugging the answers extracted from an AIME2025 benchmark run."""
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Debug the answers extracted from an AIME2025 benchmark run.')
    parser.add_argument('results_file', help='Path to the results file')
    parser.add_argument('--output-file', help='Path to the output file', default='debug_answers.csv')
    args = parser.parse_args()
    
    # Load the results
    results = load_results(args.results_file)
    
    # Extract the answers from each result
    debug_results = []
    for result in results:
        instance_id = result.get('instance_id', '')
        problem = result.get('problem', '')
        ground_truth = result.get('ground_truth', '')
        history = result.get('history', [])
        
        # Extract the answers
        answer_result = extract_answer(history, ground_truth)
        
        # Add the instance ID and problem to the result
        answer_result['instance_id'] = instance_id
        answer_result['problem'] = problem
        
        debug_results.append(answer_result)
    
    # Create a DataFrame from the debug results
    df = pd.DataFrame(debug_results)
    
    # Save the DataFrame to a CSV file
    df.to_csv(args.output_file, index=False)
    
    # Print a summary of the results
    total = len(df)
    correct = df['is_correct'].sum()
    accuracy = correct / total if total > 0 else 0.0
    
    print(f"Total problems: {total}")
    print(f"Correct answers: {correct}")
    print(f"Overall accuracy: {accuracy:.2%}")
    print()
    
    # Print the sources of the final answers
    sources = df['final_source'].value_counts()
    print("Sources of final answers:")
    for source, count in sources.items():
        print(f"  {source}: {count} ({count / total:.2%})")
    
    print(f"\nResults saved to: {args.output_file}")


if __name__ == '__main__':
    main()