#!/usr/bin/env python3

import argparse
import json
import os
from collections import defaultdict

def load_jsonl(file_path):
    """Load data from a jsonl file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def summarize_results(output_file):
    """Summarize the results of the polyglot benchmark evaluation."""
    if not os.path.exists(output_file):
        print(f"Error: Output file {output_file} does not exist.")
        return
        
    results = load_jsonl(output_file)
    
    # Count total instances
    total_instances = len(results)
    print(f"Total instances: {total_instances}")
    
    # Count by language
    language_counts = defaultdict(int)
    language_passed = defaultdict(int)
    
    # Count passed and failed instances
    passed_instances = []
    failed_instances = []
    
    for result in results:
        instance = result.get('instance', {})
        language = instance.get('language', 'unknown')
        instance_name = instance.get('instance_name', 'unknown')
        instance_id = result.get('instance_id', 'unknown')
        
        language_counts[language] += 1
        
        # Check if all tests passed
        test_result = result.get('test_result', {})
        exit_code = test_result.get('exit_code', 1)
        
        if exit_code == 0:
            passed_instances.append((instance_id, language, instance_name))
            language_passed[language] += 1
        else:
            failed_instances.append((instance_id, language, instance_name))
    
    # Print summary
    print("\nResults by language:")
    print("--------------------")
    for language, count in sorted(language_counts.items()):
        passed = language_passed[language]
        percentage = (passed / count) * 100 if count > 0 else 0
        print(f"{language}: {passed}/{count} ({percentage:.1f}%)")
    
    # Overall pass rate
    total_passed = len(passed_instances)
    overall_percentage = (total_passed / total_instances) * 100 if total_instances > 0 else 0
    print(f"\nOverall pass rate: {total_passed}/{total_instances} ({overall_percentage:.1f}%)")
    
    # Print passed instances
    print("\nPassed instances:")
    print("----------------")
    for instance_id, language, instance_name in sorted(passed_instances):
        print(f"ID: {instance_id}, Language: {language}, Name: {instance_name}")
    
    # Print failed instances
    print("\nFailed instances:")
    print("----------------")
    for instance_id, language, instance_name in sorted(failed_instances):
        print(f"ID: {instance_id}, Language: {language}, Name: {instance_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize polyglot benchmark results")
    parser.add_argument("output_file", help="Path to the output.jsonl file")
    args = parser.parse_args()
    
    summarize_results(args.output_file)