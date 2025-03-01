#!/usr/bin/env python3
"""
Script to analyze the results of the AIME2024 benchmark.
"""

import argparse
import json
import os
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt


def load_results(results_file):
    """Load results from a JSONL file."""
    results = []
    with open(results_file, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return results


def analyze_results(results):
    """Analyze the results and return a summary."""
    total = len(results)
    correct = sum(1 for r in results if r['test_result']['is_correct'])
    accuracy = correct / total if total > 0 else 0
    
    # Analyze by problem ID
    by_id = defaultdict(lambda: {'correct': 0, 'total': 0})
    for r in results:
        problem_id = r['test_result']['id']
        by_id[problem_id]['total'] += 1
        if r['test_result']['is_correct']:
            by_id[problem_id]['correct'] += 1
    
    for id_data in by_id.values():
        id_data['accuracy'] = id_data['correct'] / id_data['total'] if id_data['total'] > 0 else 0
    
    return {
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'by_id': dict(by_id)
    }


def plot_results(summary, output_dir):
    """Plot the results and save the figures."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Overall accuracy
    plt.figure(figsize=(10, 6))
    plt.bar(['Correct', 'Incorrect'], [summary['accuracy'], 1 - summary['accuracy']], color=['green', 'red'])
    plt.title(f'Overall Accuracy: {summary["accuracy"]:.2%}')
    plt.ylabel('Percentage')
    plt.ylim(0, 1)
    for i, v in enumerate([summary['accuracy'], 1 - summary['accuracy']]):
        plt.text(i, v + 0.02, f'{v:.2%}', ha='center')
    plt.savefig(os.path.join(output_dir, 'overall_accuracy.png'))
    
    # Accuracy by problem ID
    if summary['by_id']:
        ids = list(summary['by_id'].keys())
        accuracies = [summary['by_id'][id]['accuracy'] for id in ids]
        
        plt.figure(figsize=(12, 6))
        plt.bar(ids, accuracies, color='blue')
        plt.title('Accuracy by Problem ID')
        plt.xlabel('Problem ID')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_by_id.png'))


def main():
    parser = argparse.ArgumentParser(description='Analyze AIME2024 benchmark results')
    parser.add_argument('results_file', type=str, help='Path to the results JSONL file')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save analysis results')
    args = parser.parse_args()
    
    # Set default output directory if not provided
    if args.output_dir is None:
        output_dir = os.path.join(os.path.dirname(args.results_file), 'analysis')
    else:
        output_dir = args.output_dir
    
    # Load results
    results = load_results(args.results_file)
    
    # Analyze results
    summary = analyze_results(results)
    
    # Print summary
    print(f"Total problems: {summary['total']}")
    print(f"Correct answers: {summary['correct']}")
    print(f"Overall accuracy: {summary['accuracy']:.2%}")
    
    # Plot results
    plot_results(summary, output_dir)
    
    # Save summary to file
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create a detailed DataFrame
    details = []
    for r in results:
        details.append({
            'instance_id': r['instance_id'],
            'problem_id': r['test_result']['id'],
            'correct': r['test_result']['is_correct'],
            'predicted_answer': r['test_result']['predicted_answer'],
            'reference_answer': r['test_result']['reference_answer'],
            'url': r['test_result'].get('url', None)
        })
    
    df = pd.DataFrame(details)
    df.to_csv(os.path.join(output_dir, 'detailed_results.csv'), index=False)
    
    print(f"Analysis saved to {output_dir}")


if __name__ == '__main__':
    main()