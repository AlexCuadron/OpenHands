#!/usr/bin/env python3
"""
Script to analyze the results of the MATH-500 benchmark.
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_results(results_file):
    """Load results from a JSONL file."""
    results = []
    with open(results_file, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return results


def analyze_results(results):
    """Analyze the results of the MATH-500 benchmark."""
    # Extract relevant information
    data = []
    for result in results:
        test_result = result.get('test_result', {})
        instance = result.get('instance', {})
        
        data.append({
            'instance_id': result.get('instance_id'),
            'subject': test_result.get('subject', instance.get('subject')),
            'level': test_result.get('level', instance.get('level')),
            'is_correct': test_result.get('is_correct', False),
            'predicted_answer': test_result.get('predicted_answer'),
            'reference_answer': test_result.get('reference_answer', instance.get('answer')),
        })
    
    df = pd.DataFrame(data)
    
    # Overall accuracy
    overall_accuracy = df['is_correct'].mean()
    print(f"Overall accuracy: {overall_accuracy:.2%}")
    
    # Accuracy by subject
    subject_accuracy = df.groupby('subject')['is_correct'].agg(['mean', 'count'])
    subject_accuracy.columns = ['Accuracy', 'Count']
    subject_accuracy = subject_accuracy.sort_values('Accuracy', ascending=False)
    print("\nAccuracy by subject:")
    print(subject_accuracy)
    
    # Accuracy by difficulty level
    level_accuracy = df.groupby('level')['is_correct'].agg(['mean', 'count'])
    level_accuracy.columns = ['Accuracy', 'Count']
    level_accuracy = level_accuracy.sort_index()
    print("\nAccuracy by difficulty level:")
    print(level_accuracy)
    
    return {
        'df': df,
        'overall_accuracy': overall_accuracy,
        'subject_accuracy': subject_accuracy,
        'level_accuracy': level_accuracy,
    }


def plot_results(analysis_results, output_dir):
    """Plot the results of the analysis."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot accuracy by subject
    subject_accuracy = analysis_results['subject_accuracy']
    plt.figure(figsize=(12, 6))
    bars = plt.bar(subject_accuracy.index, subject_accuracy['Accuracy'])
    plt.xlabel('Subject')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Subject')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    
    # Add count labels
    for bar, count in zip(bars, subject_accuracy['Count']):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f'n={count}',
            ha='center',
            va='bottom',
            fontsize=8,
        )
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_by_subject.png'))
    
    # Plot accuracy by difficulty level
    level_accuracy = analysis_results['level_accuracy']
    plt.figure(figsize=(8, 6))
    bars = plt.bar(level_accuracy.index, level_accuracy['Accuracy'])
    plt.xlabel('Difficulty Level')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Difficulty Level')
    plt.ylim(0, 1)
    
    # Add count labels
    for bar, count in zip(bars, level_accuracy['Count']):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f'n={count}',
            ha='center',
            va='bottom',
            fontsize=8,
        )
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_by_level.png'))


def main():
    parser = argparse.ArgumentParser(description='Analyze MATH-500 benchmark results')
    parser.add_argument('results_file', help='Path to the results JSONL file')
    parser.add_argument('--output-dir', default='analysis_results', help='Directory to save analysis results')
    args = parser.parse_args()
    
    results = load_results(args.results_file)
    analysis_results = analyze_results(results)
    plot_results(analysis_results, args.output_dir)
    
    print(f"\nAnalysis results saved to {args.output_dir}")


if __name__ == '__main__':
    main()