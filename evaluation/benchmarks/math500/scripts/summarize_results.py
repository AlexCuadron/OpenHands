#!/usr/bin/env python3
"""Summarize the results of the MATH-500 benchmark."""

import argparse
import json
import os
from collections import defaultdict

import pandas as pd


def load_results(output_file):
    """Load the results from the output file.

    Args:
        output_file: Path to the output file

    Returns:
        A list of result dictionaries
    """
    results = []
    with open(output_file, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return results


def summarize_results(results):
    """Summarize the results.

    Args:
        results: A list of result dictionaries

    Returns:
        A dictionary with the summary statistics
    """
    total = len(results)
    correct = sum(
        1 for r in results if r.get('test_result', {}).get('is_correct', False)
    )

    # Calculate accuracy by subject
    subject_counts = defaultdict(int)
    subject_correct = defaultdict(int)

    # Calculate accuracy by level
    level_counts = defaultdict(int)
    level_correct = defaultdict(int)

    # Count problems with no answer
    no_answer_count = 0

    for r in results:
        test_result = r.get('test_result', {})
        subject = test_result.get('subject')
        level = test_result.get('level')
        is_correct = test_result.get('is_correct', False)

        # Count problems with no answer
        if test_result.get('predicted_answer') is None:
            no_answer_count += 1

        if subject:
            subject_counts[subject] += 1
            if is_correct:
                subject_correct[subject] += 1

        if level:
            level_counts[level] += 1
            if is_correct:
                level_correct[level] += 1

    # Calculate accuracy percentages
    subject_accuracy = {
        subject: subject_correct[subject] / count * 100
        for subject, count in subject_counts.items()
    }

    level_accuracy = {
        level: level_correct[level] / count * 100
        for level, count in level_counts.items()
    }

    return {
        'total': total,
        'correct': correct,
        'accuracy': correct / total * 100 if total > 0 else 0,
        'no_answer_count': no_answer_count,
        'subject_counts': dict(subject_counts),
        'subject_correct': dict(subject_correct),
        'subject_accuracy': subject_accuracy,
        'level_counts': dict(level_counts),
        'level_correct': dict(level_correct),
        'level_accuracy': level_accuracy,
    }


def print_summary(summary):
    """Print the summary statistics.

    Args:
        summary: A dictionary with the summary statistics
    """
    print(f"Total problems: {summary['total']}")
    print(f"Correct answers: {summary['correct']}")
    print(f"Overall accuracy: {summary['accuracy']:.2f}%")

    # Print no answer count if available
    if 'no_answer_count' in summary:
        no_answer_count = summary['no_answer_count']
        total_problems = summary['total']
        percentage = no_answer_count / total_problems * 100 if total_problems > 0 else 0
        print(
            f'Problems with no answer: {no_answer_count}/{total_problems} ({percentage:.2f}%)'
        )

    print('\nAccuracy by subject:')
    for subject, accuracy in sorted(summary['subject_accuracy'].items()):
        correct = summary['subject_correct'][subject]
        total = summary['subject_counts'][subject]
        print(f'  {subject}: {correct}/{total} ({accuracy:.2f}%)')

    print('\nAccuracy by level:')
    for level, accuracy in sorted(summary['level_accuracy'].items()):
        correct = summary['level_correct'][level]
        total = summary['level_counts'][level]
        print(f'  Level {level}: {correct}/{total} ({accuracy:.2f}%)')


def main():
    parser = argparse.ArgumentParser(description='Summarize MATH-500 benchmark results')
    parser.add_argument(
        '--output-file', type=str, required=True, help='Path to the output.jsonl file'
    )
    parser.add_argument(
        '--csv-output',
        type=str,
        default=None,
        help='Path to save the CSV summary (optional)',
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_file):
        print(f'Error: Output file {args.output_file} does not exist')
        return

    results = load_results(args.output_file)
    summary = summarize_results(results)
    print_summary(summary)

    if args.csv_output:
        # Create a DataFrame for the results
        rows = []

        # Overall accuracy
        rows.append(
            {
                'Category': 'Overall',
                'Subcategory': 'All',
                'Correct': summary['correct'],
                'Total': summary['total'],
                'Accuracy': summary['accuracy'],
            }
        )

        # Subject accuracy
        for subject, accuracy in summary['subject_accuracy'].items():
            rows.append(
                {
                    'Category': 'Subject',
                    'Subcategory': subject,
                    'Correct': summary['subject_correct'][subject],
                    'Total': summary['subject_counts'][subject],
                    'Accuracy': accuracy,
                }
            )

        # Level accuracy
        for level, accuracy in summary['level_accuracy'].items():
            rows.append(
                {
                    'Category': 'Level',
                    'Subcategory': str(level),
                    'Correct': summary['level_correct'][level],
                    'Total': summary['level_counts'][level],
                    'Accuracy': accuracy,
                }
            )

        df = pd.DataFrame(rows)
        df.to_csv(args.csv_output, index=False)
        print(f'\nSummary saved to {args.csv_output}')


if __name__ == '__main__':
    main()