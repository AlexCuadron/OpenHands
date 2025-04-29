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
    """Summarize the results of the APPS benchmark."""
    print(f"Summarizing results from {output_file}")
    
    # Load the results
    results = load_jsonl(output_file)
    
    # Count the number of instances that passed and failed
    passed = []
    failed = []
    
    for result in results:
        instance_id = result['instance_id']
        test_result = result.get('test_result', {})
        exit_code = test_result.get('exit_code', 1)
        
        if exit_code == 0:
            passed.append(instance_id)
        else:
            failed.append(instance_id)
    
    # Print the summary
    print(f"\nTotal instances: {len(results)}")
    print(f"Passed: {len(passed)} ({len(passed) / len(results) * 100:.2f}%)")
    print(f"Failed: {len(failed)} ({len(failed) / len(results) * 100:.2f}%)")
    
    # Print the list of passed and failed instances
    print("\nPassed instances:")
    for instance_id in passed:
        print(f"  - {instance_id}")
    
    print("\nFailed instances:")
    for instance_id in failed:
        print(f"  - {instance_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize APPS benchmark results")
    parser.add_argument("output_file", help="Path to the output.jsonl file")
    args = parser.parse_args()
    
    summarize_results(args.output_file)