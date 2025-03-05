#!/usr/bin/env python3
"""
Script to debug answers from the AIME2025 benchmark.
This script extracts answers from the agent's responses and compares them to the reference answers.
"""

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
from datasets import load_dataset

from evaluation.benchmarks.aime2025.run_infer import extract_answer, normalize_answer


def load_results(results_file: str) -> List[Dict]:
    """Load results from a JSONL file."""
    results = []
    with open(results_file, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return results


def load_dataset_answers() -> Dict[str, str]:
    """Load the reference answers from the AIME2025 dataset."""
    # Load the AIME2025 dataset
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

    # Create a dictionary of instance_id -> answer
    answers = {}
    for i, row in aime_df.iterrows():
        instance_id = f'aime2025_{i}'
        answers[instance_id] = row['answer']
    
    return answers


def extract_answers_from_results(
    results: List[Dict],
) -> List[Dict]:
    """Extract answers from the results."""
    extracted_answers = []
    
    for result in results:
        instance_id = result['instance_id']
        history = result['history']
        
        # Extract the last assistant message
        last_assistant_message = None
        for event in reversed(history):
            if event[0] == 'assistant' and isinstance(event[1], str):
                last_assistant_message = event[1]
                break
        
        # Extract the answer from the last assistant message
        extracted_answer = extract_answer(last_assistant_message) if last_assistant_message else None
        normalized_answer = normalize_answer(extracted_answer) if extracted_answer else None
        
        # Get the reference answer from the test_result
        reference_answer = result['test_result']['reference_answer']
        reference_normalized = normalize_answer(reference_answer) if reference_answer else None
        
        # Check if the answer is correct
        is_correct = result['test_result']['is_correct']
        
        extracted_answers.append({
            'instance_id': instance_id,
            'extracted_answer': extracted_answer,
            'normalized_answer': normalized_answer,
            'reference_answer': reference_answer,
            'reference_normalized': reference_normalized,
            'is_correct': is_correct,
        })
    
    return extracted_answers


def main():
    parser = argparse.ArgumentParser(description='Debug answers from AIME2025 benchmark')
    parser.add_argument('results_file', type=str, help='Path to the results JSONL file')
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save debug results',
    )
    args = parser.parse_args()
    
    # Set default output directory if not provided
    if args.output_dir is None:
        output_dir = os.path.join(os.path.dirname(args.results_file), 'debug')
    else:
        output_dir = args.output_dir
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    results = load_results(args.results_file)
    
    # Load dataset answers
    dataset_answers = load_dataset_answers()
    
    # Extract answers from results
    extracted_answers = extract_answers_from_results(results)
    
    # Create a DataFrame with the extracted answers
    df = pd.DataFrame(extracted_answers)
    
    # Add the dataset answers for comparison
    df['dataset_answer'] = df['instance_id'].map(dataset_answers)
    df['dataset_normalized'] = df['dataset_answer'].apply(normalize_answer)
    
    # Check if the normalized answer matches the dataset normalized answer
    df['matches_dataset'] = df.apply(
        lambda row: row['normalized_answer'] == row['dataset_normalized']
        if row['normalized_answer'] is not None and row['dataset_normalized'] is not None
        else False,
        axis=1,
    )
    
    # Save the DataFrame to a CSV file
    output_file = os.path.join(output_dir, 'debug_answers.csv')
    df.to_csv(output_file, index=False)
    print(f'Saved debug answers to {output_file}')
    
    # Print summary statistics
    total = len(df)
    correct = df['is_correct'].sum()
    matches_dataset = df['matches_dataset'].sum()
    
    print(f'Total examples: {total}')
    print(f'Correct answers: {correct} ({correct/total:.2%})')
    print(f'Matches dataset: {matches_dataset} ({matches_dataset/total:.2%})')
    
    # Check for discrepancies between is_correct and matches_dataset
    discrepancies = df[df['is_correct'] != df['matches_dataset']]
    if not discrepancies.empty:
        print(f'\nFound {len(discrepancies)} discrepancies between is_correct and matches_dataset:')
        for i, row in discrepancies.head(5).iterrows():
            print(f"\n{i+1}. Instance ID: {row['instance_id']}")
            print(f"   Extracted: {row['extracted_answer']}")
            print(f"   Normalized: {row['normalized_answer']}")
            print(f"   Reference: {row['reference_answer']}")
            print(f"   Reference normalized: {row['reference_normalized']}")
            print(f"   Dataset: {row['dataset_answer']}")
            print(f"   Dataset normalized: {row['dataset_normalized']}")
            print(f"   is_correct: {row['is_correct']}")
            print(f"   matches_dataset: {row['matches_dataset']}")
        
        if len(discrepancies) > 5:
            print(f'\n... and {len(discrepancies) - 5} more discrepancies (see {output_file})')
        
        # Save discrepancies to a separate CSV file
        discrepancies_file = os.path.join(output_dir, 'discrepancies.csv')
        discrepancies.to_csv(discrepancies_file, index=False)
        print(f'Saved discrepancies to {discrepancies_file}')


if __name__ == '__main__':
    main()