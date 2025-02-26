#!/usr/bin/env python3
import argparse
import json
import os
from collections import defaultdict


def load_jsonl(file_path):
    """Load a jsonl file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def summarize_results(output_file):
    """Summarize the results of the HotpotQA benchmark."""
    print(f"Summarizing results from {output_file}")
    
    # Load the results
    results = load_jsonl(output_file)
    
    # Count the number of instances that passed and failed
    correct = []
    incorrect = []
    
    for result in results:
        instance_id = result['instance_id']
        test_result = result.get('test_result', {})
        is_correct = test_result.get('is_correct', False)
        
        if is_correct:
            correct.append(instance_id)
        else:
            incorrect.append(instance_id)
    
    # Print the summary
    print(f"\nTotal instances: {len(results)}")
    print(f"Correct: {len(correct)} ({len(correct) / len(results) * 100:.2f}%)")
    print(f"Incorrect: {len(incorrect)} ({len(incorrect) / len(results) * 100:.2f}%)")
    
    # Print the list of correct and incorrect instances
    print("\nCorrect instances:")
    for instance_id in correct:
        print(f"  - {instance_id}")
    
    print("\nIncorrect instances:")
    for instance_id in incorrect:
        print(f"  - {instance_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize HotpotQA benchmark results")
    parser.add_argument("output_file", help="Path to the output.jsonl file")
    args = parser.parse_args()
    
    summarize_results(args.output_file)