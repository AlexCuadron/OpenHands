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

def test_language(language, model, agent):
    """Test the first instance of a specific language."""
    print(f"\n{'=' * 50}")
    print(f"Testing language: {language}")
    print(f"{'=' * 50}\n")
    
    # Set the environment variable for the polyglot benchmark path
    os.environ['POLYGLOT_BENCHMARK_PATH'] = '/workspace/polyglot-benchmark'
    
    # Load the dataset
    dataset = load_polyglot_dataset()
    
    # Filter by language
    dataset = dataset[dataset['language'].str.lower() == language.lower()]
    if dataset.empty:
        print(f"No instances found for language: {language}")
        return False
    
    # Get the first instance
    instance = dataset.iloc[0]
    print(f"Testing instance {instance.instance_id}: {instance.instance_name} ({instance.language})")
    
    # Get LLM config
    llm_config = get_llm_config_arg(model)
    if llm_config is None:
        print(f"Could not find LLM config: {model}")
        return False
    
    # Create metadata
    metadata = make_metadata(
        llm_config,
        'PolyglotBenchmark',
        agent,
        30,  # max_iterations
        f"test_{language}",
        f"evaluation/evaluation_outputs/test_{language}",
    )
    
    # Process the instance
    try:
        output = process_instance(instance, metadata, reset_logger=False)
        print("\nTest completed successfully!")
        print(f"Exit code: {output.test_result['exit_code']}")
        print(f"Passed: {output.test_result['exit_code'] == 0}")
        return output.test_result['exit_code'] == 0
    except Exception as e:
        print(f"Error processing instance: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test the polyglot benchmark with one instance per language")
    parser.add_argument("--model", default="eval_gpt35_turbo", help="Model configuration name")
    parser.add_argument("--agent", default="CodeActAgent", help="Agent class name")
    parser.add_argument("--languages", default="python,rust,go,javascript,cpp,java", 
                        help="Comma-separated list of languages to test")
    args = parser.parse_args()
    
    languages = args.languages.split(',')
    results = {}
    
    for language in languages:
        language = language.strip()
        if not language:
            continue
        
        success = test_language(language, args.model, args.agent)
        results[language] = "PASSED" if success else "FAILED"
    
    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY OF RESULTS")
    print("=" * 50)
    
    for language, result in results.items():
        print(f"{language.ljust(12)}: {result}")
    
    # Check if all tests passed
    all_passed = all(result == "PASSED" for result in results.values())
    print("\nOverall result:", "PASSED" if all_passed else "FAILED")

if __name__ == "__main__":
    main()