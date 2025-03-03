#!/usr/bin/env python3
"""
Script to analyze the results of the AIME2024 benchmark.
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd


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
        id_data['accuracy'] = (
            id_data['correct'] / id_data['total'] if id_data['total'] > 0 else 0
        )
    
    # Analyze discrepancies between predicted and reference answers
    discrepancies = []
    comparison_methods = {'numerical': 0, 'string': 0}
    
    for r in results:
        if not r['test_result']['is_correct'] and r['test_result'].get('predicted_answer') is not None:
            discrepancy = {
                'problem_id': r['test_result']['id'],
                'predicted': r['test_result']['predicted_answer'],
                'reference': r['test_result']['reference_answer'],
            }
            
            # Add normalized values if available
            if 'predicted_normalized' in r['test_result']:
                discrepancy['predicted_normalized'] = r['test_result']['predicted_normalized']
            if 'reference_normalized' in r['test_result']:
                discrepancy['reference_normalized'] = r['test_result']['reference_normalized']
            if 'comparison_method' in r['test_result']:
                discrepancy['comparison_method'] = r['test_result']['comparison_method']
                
            discrepancies.append(discrepancy)
        
        # Count comparison methods
        if 'comparison_method' in r['test_result']:
            method = r['test_result']['comparison_method']
            comparison_methods[method] = comparison_methods.get(method, 0) + 1

    # Analyze overthinking scores if available
    overthinking_scores = []
    solutions_discarded = 0
    
    for r in results:
        # Check for overthinking score
        if 'overthinking_score' in r['test_result']:
            overthinking_scores.append(r['test_result']['overthinking_score'])
            
            # Check if solution was discarded due to overthinking
            if r['test_result'].get('solution_discarded', False):
                solutions_discarded += 1
    
    # Calculate overthinking statistics if scores are available
    overthinking_stats = {}
    if overthinking_scores:
        overthinking_stats = {
            'min': min(overthinking_scores),
            'max': max(overthinking_scores),
            'avg': sum(overthinking_scores) / len(overthinking_scores),
            'count': len(overthinking_scores),
            'solutions_discarded': solutions_discarded,
        }
    
    return {
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'by_id': dict(by_id),
        'discrepancies': discrepancies,
        'comparison_methods': comparison_methods,
        'overthinking_stats': overthinking_stats,
    }


def plot_results(summary, output_dir):
    """Plot the results and save the figures."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving plots to {output_dir}")

    # Overall accuracy
    try:
        plt.figure(figsize=(10, 6))
        plt.bar(
            ['Correct', 'Incorrect'],
            [summary['accuracy'], 1 - summary['accuracy']],
            color=['green', 'red'],
        )
        plt.title(f'Overall Accuracy: {summary["accuracy"]:.2%}')
        plt.ylabel('Percentage')
        plt.ylim(0, 1)
        for i, v in enumerate([summary['accuracy'], 1 - summary['accuracy']]):
            plt.text(i, v + 0.02, f'{v:.2%}', ha='center')
        
        accuracy_plot_path = os.path.join(output_dir, 'overall_accuracy.png')
        plt.savefig(accuracy_plot_path)
        print(f"Saved overall accuracy plot to {accuracy_plot_path}")
    except Exception as e:
        print(f"Error creating overall accuracy plot: {e}")

    # Accuracy by problem ID
    if summary['by_id']:
        try:
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
            
            accuracy_by_id_path = os.path.join(output_dir, 'accuracy_by_id.png')
            plt.savefig(accuracy_by_id_path)
            print(f"Saved accuracy by problem ID plot to {accuracy_by_id_path}")
        except Exception as e:
            print(f"Error creating accuracy by problem ID plot: {e}")
    
    # Comparison methods
    if 'comparison_methods' in summary and summary['comparison_methods']:
        try:
            methods = list(summary['comparison_methods'].keys())
            counts = list(summary['comparison_methods'].values())
            
            plt.figure(figsize=(10, 6))
            plt.bar(methods, counts, color='purple')
            plt.title('Comparison Methods Used')
            plt.xlabel('Method')
            plt.ylabel('Count')
            for i, v in enumerate(counts):
                plt.text(i, v + 0.5, str(v), ha='center')
            plt.tight_layout()
            
            comparison_methods_path = os.path.join(output_dir, 'comparison_methods.png')
            plt.savefig(comparison_methods_path)
            print(f"Saved comparison methods plot to {comparison_methods_path}")
        except Exception as e:
            print(f"Error creating comparison methods plot: {e}")
        
        # Correct vs Incorrect by comparison method
        if 'discrepancies' in summary:
            try:
                # Count incorrect answers by method
                incorrect_by_method = {}
                for disc in summary['discrepancies']:
                    if 'comparison_method' in disc:
                        method = disc['comparison_method']
                        incorrect_by_method[method] = incorrect_by_method.get(method, 0) + 1
                
                # Calculate correct answers by method
                correct_by_method = {}
                for method, total in summary['comparison_methods'].items():
                    incorrect = incorrect_by_method.get(method, 0)
                    correct_by_method[method] = total - incorrect
                
                # Create stacked bar chart
                methods = list(summary['comparison_methods'].keys())
                correct_counts = [correct_by_method.get(m, 0) for m in methods]
                incorrect_counts = [incorrect_by_method.get(m, 0) for m in methods]
                
                plt.figure(figsize=(10, 6))
                plt.bar(methods, correct_counts, label='Correct', color='green')
                plt.bar(methods, incorrect_counts, bottom=correct_counts, label='Incorrect', color='red')
                plt.title('Correct vs Incorrect Answers by Comparison Method')
                plt.xlabel('Method')
                plt.ylabel('Count')
                plt.legend()
                plt.tight_layout()
                
                comparison_results_path = os.path.join(output_dir, 'comparison_results.png')
                plt.savefig(comparison_results_path)
                print(f"Saved comparison results plot to {comparison_results_path}")
            except Exception as e:
                print(f"Error creating comparison results plot: {e}")
    
    # Plot overthinking scores if available
    if 'overthinking_stats' in summary and summary['overthinking_stats']:
        try:
            # Create a histogram of overthinking scores
            plt.figure(figsize=(10, 6))
            
            # Get overthinking scores from all results
            scores = []
            for r in results:
                if 'overthinking_score' in r['test_result']:
                    scores.append(r['test_result']['overthinking_score'])
            
            # Create histogram with 11 bins (0-10)
            plt.hist(scores, bins=range(12), color='orange', edgecolor='black', alpha=0.7)
            plt.title('Distribution of Overthinking Scores')
            plt.xlabel('Overthinking Score (0-10)')
            plt.ylabel('Number of Solutions')
            plt.xticks(range(11))
            plt.grid(axis='y', alpha=0.3)
            
            # Add vertical line at the average
            avg_score = summary['overthinking_stats']['avg']
            plt.axvline(x=avg_score, color='red', linestyle='--', label=f'Average: {avg_score:.2f}')
            plt.legend()
            
            overthinking_hist_path = os.path.join(output_dir, 'overthinking_scores.png')
            plt.savefig(overthinking_hist_path)
            print(f"Saved overthinking scores histogram to {overthinking_hist_path}")
            
            # Create a scatter plot of overthinking score vs correctness
            plt.figure(figsize=(10, 6))
            
            # Prepare data
            correct_scores = []
            incorrect_scores = []
            discarded_scores = []
            
            for r in results:
                if 'overthinking_score' in r['test_result']:
                    score = r['test_result']['overthinking_score']
                    if r['test_result'].get('solution_discarded', False):
                        discarded_scores.append(score)
                    elif r['test_result']['is_correct']:
                        correct_scores.append(score)
                    else:
                        incorrect_scores.append(score)
            
            # Create scatter plot
            plt.scatter([0] * len(correct_scores), correct_scores, color='green', label='Correct', alpha=0.7)
            plt.scatter([1] * len(incorrect_scores), incorrect_scores, color='red', label='Incorrect', alpha=0.7)
            plt.scatter([2] * len(discarded_scores), discarded_scores, color='orange', label='Discarded', alpha=0.7)
            
            plt.title('Overthinking Scores by Solution Outcome')
            plt.xlabel('Outcome')
            plt.ylabel('Overthinking Score (0-10)')
            plt.xticks([0, 1, 2], ['Correct', 'Incorrect', 'Discarded'])
            plt.ylim(-0.5, 10.5)
            plt.grid(axis='y', alpha=0.3)
            plt.legend()
            
            overthinking_scatter_path = os.path.join(output_dir, 'overthinking_by_outcome.png')
            plt.savefig(overthinking_scatter_path)
            print(f"Saved overthinking by outcome plot to {overthinking_scatter_path}")
            
        except Exception as e:
            print(f"Error creating overthinking plots: {e}")


def main():
    parser = argparse.ArgumentParser(description='Analyze AIME2024 benchmark results')
    parser.add_argument('results_file', type=str, help='Path to the results JSONL file')
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save analysis results',
    )
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
    
    # Print overthinking statistics if available
    if 'overthinking_stats' in summary and summary['overthinking_stats']:
        print("\nOverthinking statistics:")
        stats = summary['overthinking_stats']
        print(f"  Analyzed solutions: {stats['count']}")
        print(f"  Average overthinking score: {stats['avg']:.2f}")
        print(f"  Min overthinking score: {stats['min']}")
        print(f"  Max overthinking score: {stats['max']}")
        print(f"  Solutions discarded: {stats['solutions_discarded']} ({stats['solutions_discarded']/stats['count']:.2%} of analyzed)")
    
    # Print comparison method statistics
    if 'comparison_methods' in summary:
        print("\nComparison methods used:")
        for method, count in summary['comparison_methods'].items():
            print(f"  {method}: {count} ({count/summary['total']:.2%})")
    
    # Print discrepancy information
    if 'discrepancies' in summary and summary['discrepancies']:
        print(f"\nFound {len(summary['discrepancies'])} answer discrepancies:")
        for i, disc in enumerate(summary['discrepancies'][:5], 1):  # Show first 5 discrepancies
            print(f"\n{i}. Problem ID: {disc['problem_id']}")
            print(f"   Predicted: {disc['predicted']}")
            print(f"   Reference: {disc['reference']}")
            if 'predicted_normalized' in disc and 'reference_normalized' in disc:
                print(f"   Normalized: '{disc['predicted_normalized']}' vs '{disc['reference_normalized']}'")
            if 'comparison_method' in disc:
                print(f"   Comparison method: {disc['comparison_method']}")
        
        if len(summary['discrepancies']) > 5:
            print(f"\n... and {len(summary['discrepancies']) - 5} more discrepancies (see detailed_results.csv)")
            
    # Create a separate CSV file for discrepancies
    if 'discrepancies' in summary and summary['discrepancies']:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the discrepancies to a CSV file
        discrepancies_file = os.path.join(output_dir, 'discrepancies.csv')
        pd.DataFrame(summary['discrepancies']).to_csv(discrepancies_file, index=False)
        print(f"Saved discrepancies to {discrepancies_file}")

    # Plot results
    plot_results(summary, output_dir)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary to file
    summary_file = os.path.join(output_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_file}")

    # Create a detailed DataFrame
    details = []
    for r in results:
        result_dict = {
            'instance_id': r['instance_id'],
            'problem_id': r['test_result']['id'],
            'correct': r['test_result']['is_correct'],
            'predicted_answer': r['test_result']['predicted_answer'],
            'reference_answer': r['test_result']['reference_answer'],
            'url': r['test_result'].get('url', None),
        }
        
        # Add normalized answers if available
        if 'predicted_normalized' in r['test_result']:
            result_dict['predicted_normalized'] = r['test_result']['predicted_normalized']
        if 'reference_normalized' in r['test_result']:
            result_dict['reference_normalized'] = r['test_result']['reference_normalized']
        if 'comparison_method' in r['test_result']:
            result_dict['comparison_method'] = r['test_result']['comparison_method']
            
        # Add overthinking information if available
        if 'overthinking_score' in r['test_result']:
            result_dict['overthinking_score'] = r['test_result']['overthinking_score']
        if 'solution_discarded' in r['test_result']:
            result_dict['solution_discarded'] = r['test_result']['solution_discarded']
            
        details.append(result_dict)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results to CSV
    df = pd.DataFrame(details)
    detailed_results_file = os.path.join(output_dir, 'detailed_results.csv')
    df.to_csv(detailed_results_file, index=False)
    print(f"Saved detailed results to {detailed_results_file}")

    print(f'Analysis saved to {output_dir}')


if __name__ == '__main__':
    main()
