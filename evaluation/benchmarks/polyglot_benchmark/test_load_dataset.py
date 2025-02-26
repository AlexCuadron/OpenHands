#!/usr/bin/env python3

import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from evaluation.benchmarks.polyglot_benchmark.run_infer import load_polyglot_dataset

def main():
    # Set the environment variable for the polyglot benchmark path
    os.environ['POLYGLOT_BENCHMARK_PATH'] = '/workspace/polyglot-benchmark'
    
    # Load the dataset
    dataset = load_polyglot_dataset()
    
    # Print summary
    print(f"Loaded {len(dataset)} test instances")
    
    # Print language distribution
    language_counts = dataset['language'].value_counts()
    print("\nLanguage distribution:")
    for language, count in language_counts.items():
        print(f"{language}: {count}")
    
    # Print a sample instance
    if not dataset.empty:
        print("\nSample instance:")
        sample = dataset.iloc[0]
        print(f"ID: {sample.instance_id}")
        print(f"Name: {sample.instance_name}")
        print(f"Language: {sample.language}")
        print(f"Solution files: {sample.solution_files}")
        print(f"Test files: {sample.test_files}")
        print(f"Instruction (first 100 chars): {sample.instruction[:100]}...")

if __name__ == "__main__":
    main()