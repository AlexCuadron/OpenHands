#!/usr/bin/env python3

import os
import sys
import argparse
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from evaluation.benchmarks.polyglot_benchmark.run_infer import (
    load_polyglot_dataset,
    process_instance,
    make_metadata,
    get_llm_config_arg,
)
from openhands.core.logger import openhands_logger as logger

def main():
    parser = argparse.ArgumentParser(description="Test the polyglot benchmark with a single instance")
    parser.add_argument("--model", default="eval_gpt35_turbo", help="Model configuration name")
    parser.add_argument("--agent", default="CodeActAgent", help="Agent class name")
    parser.add_argument("--instance-id", type=int, default=0, help="Instance ID to test")
    parser.add_argument("--language", help="Filter by language")
    args = parser.parse_args()
    
    # Set the environment variable for the polyglot benchmark path
    os.environ['POLYGLOT_BENCHMARK_PATH'] = '/workspace/polyglot-benchmark'
    
    # Load the dataset
    dataset = load_polyglot_dataset()
    
    if args.language:
        dataset = dataset[dataset['language'].str.lower() == args.language.lower()]
        if dataset.empty:
            print(f"No instances found for language: {args.language}")
            return
    
    # Get the instance to test
    if args.instance_id >= len(dataset):
        print(f"Instance ID {args.instance_id} is out of range. Max ID: {len(dataset) - 1}")
        return
        
    instance = dataset.iloc[args.instance_id]
    print(f"Testing instance {instance.instance_id}: {instance.instance_name} ({instance.language})")
    
    # Get LLM config
    llm_config = get_llm_config_arg(args.model)
    if llm_config is None:
        print(f"Could not find LLM config: {args.model}")
        return
        
    # Create metadata
    metadata = make_metadata(
        llm_config,
        'PolyglotBenchmark',
        args.agent,
        30,  # max_iterations
        "test",
        "evaluation/evaluation_outputs/test",
    )
    
    # Process the instance
    try:
        output = process_instance(instance, metadata, reset_logger=False)
        print("\nTest completed successfully!")
        print(f"Exit code: {output.test_result['exit_code']}")
        print(f"Passed: {output.test_result['exit_code'] == 0}")
    except Exception as e:
        print(f"Error processing instance: {e}")

if __name__ == "__main__":
    main()