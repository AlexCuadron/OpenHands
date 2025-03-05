#!/usr/bin/env python3
"""
Script to analyze the results of an AIME2025 benchmark run.
"""

import argparse
import json
import os
from typing import Dict, List, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
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


def create_results_df(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a DataFrame from the results.

    Args:
        results: List of result dictionaries

    Returns:
        pd.DataFrame: DataFrame containing the results
    """
    # Extract the relevant fields from the results
    data = []
    for result in results:
        data.append({
            'instance_id': result.get('instance_id', ''),
            'problem': result.get('problem', ''),
            'predicted_answer': result.get('predicted_answer', ''),
            'ground_truth': result.get('ground_truth', ''),
            'is_correct': result.get('is_correct', False),
            'should_discard': result.get('should_discard', False),
            'discard_reason': result.get('discard_reason', ''),
        })
    
    # Create the DataFrame
    df = pd.DataFrame(data)
    
    return df


def calculate_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate metrics from the results DataFrame.

    Args:
        df: DataFrame containing the results

    Returns:
        Dict[str, Any]: Dictionary of metrics
    """
    # Calculate the overall accuracy
    total = len(df)
    correct = df['is_correct'].sum()
    accuracy = correct / total if total > 0 else 0.0
    
    # Calculate the accuracy excluding discarded solutions
    non_discarded = df[~df['should_discard']]
    non_discarded_total = len(non_discarded)
    non_discarded_correct = non_discarded['is_correct'].sum()
    non_discarded_accuracy = non_discarded_correct / non_discarded_total if non_discarded_total > 0 else 0.0
    
    # Calculate the number of discarded solutions
    discarded = df[df['should_discard']]
    discarded_total = len(discarded)
    discarded_correct = discarded['is_correct'].sum()
    discarded_accuracy = discarded_correct / discarded_total if discarded_total > 0 else 0.0
    
    # Create the metrics dictionary
    metrics = {
        'total': total,
        'correct': int(correct),
        'accuracy': accuracy,
        'non_discarded_total': non_discarded_total,
        'non_discarded_correct': int(non_discarded_correct),
        'non_discarded_accuracy': non_discarded_accuracy,
        'discarded_total': discarded_total,
        'discarded_correct': int(discarded_correct),
        'discarded_accuracy': discarded_accuracy,
    }
    
    return metrics


def plot_accuracy(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot the accuracy of the results.

    Args:
        df: DataFrame containing the results
        output_dir: Directory to save the plots
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot the overall accuracy
    plt.figure(figsize=(10, 6))
    accuracy = df['is_correct'].mean()
    plt.bar(['Overall'], [accuracy], color='blue')
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.title('Overall Accuracy')
    plt.savefig(os.path.join(output_dir, 'overall_accuracy.png'))
    plt.close()
    
    # Plot the accuracy by instance ID
    plt.figure(figsize=(15, 8))
    df_sorted = df.sort_values('instance_id')
    accuracies = df_sorted.groupby('instance_id')['is_correct'].mean()
    plt.bar(accuracies.index, accuracies, color='blue')
    plt.ylim(0, 1)
    plt.xlabel('Instance ID')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Problem ID')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_by_problem.png'))
    plt.close()
    
    # If there are discarded solutions, plot the accuracy with and without discarded solutions
    if df['should_discard'].any():
        plt.figure(figsize=(10, 6))
        overall_accuracy = df['is_correct'].mean()
        non_discarded = df[~df['should_discard']]
        non_discarded_accuracy = non_discarded['is_correct'].mean()
        plt.bar(['Overall', 'Non-Discarded'], [overall_accuracy, non_discarded_accuracy], color=['blue', 'green'])
        plt.ylim(0, 1)
        plt.ylabel('Accuracy')
        plt.title('Accuracy With and Without Discarded Solutions')
        plt.savefig(os.path.join(output_dir, 'accuracy_with_without_discarded.png'))
        plt.close()


def save_results(df: pd.DataFrame, metrics: Dict[str, Any], output_dir: str) -> None:
    """
    Save the results to files.

    Args:
        df: DataFrame containing the results
        metrics: Dictionary of metrics
        output_dir: Directory to save the results
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the DataFrame to a CSV file
    df.to_csv(os.path.join(output_dir, 'results.csv'), index=False)
    
    # Save the metrics to a JSON file
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save a summary of the results to a text file
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write(f"Total problems: {metrics['total']}\n")
        f.write(f"Correct answers: {metrics['correct']}\n")
        f.write(f"Overall accuracy: {metrics['accuracy']:.2%}\n")
        f.write("\n")
        if metrics['discarded_total'] > 0:
            f.write(f"Discarded solutions: {metrics['discarded_total']}\n")
            f.write(f"Discarded accuracy: {metrics['discarded_accuracy']:.2%}\n")
            f.write(f"Non-discarded solutions: {metrics['non_discarded_total']}\n")
            f.write(f"Non-discarded accuracy: {metrics['non_discarded_accuracy']:.2%}\n")


def main():
    """Main function for analyzing the results of an AIME2025 benchmark run."""
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Analyze the results of an AIME2025 benchmark run.')
    parser.add_argument('results_file', help='Path to the results file')
    parser.add_argument('--output-dir', help='Directory to save the results', default='analysis')
    args = parser.parse_args()
    
    # Load the results
    results = load_results(args.results_file)
    
    # Create a DataFrame from the results
    df = create_results_df(results)
    
    # Calculate metrics
    metrics = calculate_metrics(df)
    
    # Plot the accuracy
    plot_accuracy(df, args.output_dir)
    
    # Save the results
    save_results(df, metrics, args.output_dir)
    
    # Print a summary of the results
    print(f"Total problems: {metrics['total']}")
    print(f"Correct answers: {metrics['correct']}")
    print(f"Overall accuracy: {metrics['accuracy']:.2%}")
    print()
    if metrics['discarded_total'] > 0:
        print(f"Discarded solutions: {metrics['discarded_total']}")
        print(f"Discarded accuracy: {metrics['discarded_accuracy']:.2%}")
        print(f"Non-discarded solutions: {metrics['non_discarded_total']}")
        print(f"Non-discarded accuracy: {metrics['non_discarded_accuracy']:.2%}")
    
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    main()