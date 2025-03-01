#!/usr/bin/env python3
"""
Script to debug answer extraction and normalization for AIME2024 benchmark.
"""

import argparse
import json
import os
import re
from typing import Optional, Dict, List, Tuple

import pandas as pd


def extract_answer(text: str) -> Optional[str]:
    """Extract the answer from the agent's response."""
    if not text:
        return None
    
    # Look for answer in solution tags
    solution_pattern = r'<solution>(.*?)</solution>'
    solution_match = re.search(solution_pattern, text, re.DOTALL)
    if solution_match:
        return solution_match.group(1).strip()
    
    # Look for boxed answers (common in LaTeX)
    boxed_pattern = r'\\boxed{([^{}]*)}'
    boxed_match = re.search(boxed_pattern, text, re.DOTALL)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # Look for "The answer is" pattern
    answer_pattern = r'[Tt]he\s+(?:final\s+)?answer\s+is\s+([\d\w\s\.\-\+\/\*\^\{\}\\\(\)]+)'
    answer_match = re.search(answer_pattern, text, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    
    # Look for "Therefore" pattern
    therefore_pattern = r'[Tt]herefore,?\s+([\d\w\s\.\-\+\/\*\^\{\}\\\(\)=]+)'
    therefore_match = re.search(therefore_pattern, text, re.DOTALL)
    if therefore_match:
        return therefore_match.group(1).strip()
    
    # Look for "Our answer is" pattern
    our_answer_pattern = r'[Oo]ur\s+answer\s+is\s+([\d\w\s\.\-\+\/\*\^\{\}\\\(\)]+)'
    our_answer_match = re.search(our_answer_pattern, text, re.DOTALL)
    if our_answer_match:
        return our_answer_match.group(1).strip()
    
    # Look for "We get" pattern (common in math solutions)
    we_get_pattern = r'[Ww]e\s+get\s+([\d\w\s\.\-\+\/\*\^\{\}\\\(\)=]+)'
    we_get_match = re.search(we_get_pattern, text, re.DOTALL)
    if we_get_match:
        return we_get_match.group(1).strip()
    
    # Look for a standalone number at the end of the text (common in AIME problems)
    final_number_pattern = r'(?:^|\n|\.)[\s\t]*(\d+)[\s\t]*$'
    final_number_match = re.search(final_number_pattern, text)
    if final_number_match:
        return final_number_match.group(1).strip()
    
    return None


def normalize_answer(answer: str) -> str:
    """Normalize the answer for comparison."""
    if answer is None:
        return ""
    
    # Remove LaTeX commands
    answer = re.sub(r'\\boxed{(.*?)}', r'\1', answer)  # Extract content from \boxed{}
    answer = re.sub(r'\\left\(|\\right\)', '', answer)
    answer = re.sub(r'\\', '', answer)
    
    # Remove all whitespace
    answer = re.sub(r'\s+', '', answer)
    
    # Remove any text that's not part of the actual answer
    answer = re.sub(r'[Tt]he(final)?answeris', '', answer)
    answer = re.sub(r'[Tt]herefore,?', '', answer)
    
    # Handle common mathematical notations
    answer = re.sub(r'[{}()\[\]]', '', answer)  # Remove brackets
    
    # For AIME problems, we typically want just the number
    # Try to extract just the number if it's the last thing in the string
    number_match = re.search(r'(\d+)$', answer)
    if number_match:
        return number_match.group(1)
    
    return answer


def check_answer_correctness(predicted: str, reference: str) -> bool:
    """Check if the predicted answer matches the reference answer."""
    if predicted is None:
        return False
    
    # Normalize both answers
    predicted_norm = normalize_answer(predicted)
    reference_norm = normalize_answer(reference)
    
    return predicted_norm == reference_norm


def analyze_output_file(output_file: str) -> List[Dict]:
    """Analyze the output file and return a list of results."""
    results = []
    
    with open(output_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                
                # Extract information
                instance_id = data['instance_id']
                problem = data['instance']['problem']
                reference_answer = data['test_result']['reference_answer']
                predicted_answer = data['test_result']['predicted_answer']
                is_correct = data['test_result']['is_correct']
                
                # Find the finish action if any
                finish_action = None
                finish_solution = None
                for event in reversed(data['history']):
                    if event[0].get('action') == 'finish':
                        finish_action = event[0]
                        if hasattr(finish_action, 'solution'):
                            finish_solution = finish_action.get('solution', '')
                        elif 'outputs' in finish_action and 'solution' in finish_action['outputs']:
                            finish_solution = finish_action['outputs']['solution']
                        break
                
                # Find the last message from the agent
                last_message = None
                for event in reversed(data['history']):
                    if event[0].get('role') == 'assistant' and 'message' in event[0]:
                        last_message = event[0]['message']
                        break
                
                # Extract answer from the last message
                extracted_answer = extract_answer(last_message) if last_message else None
                
                # Normalize answers
                normalized_reference = normalize_answer(reference_answer)
                normalized_predicted = normalize_answer(predicted_answer)
                normalized_extracted = normalize_answer(extracted_answer)
                normalized_finish = normalize_answer(finish_solution)
                
                # Check correctness
                extracted_correct = normalized_extracted == normalized_reference
                finish_correct = normalized_finish == normalized_reference
                
                results.append({
                    'instance_id': instance_id,
                    'problem': problem[:100] + '...' if len(problem) > 100 else problem,
                    'reference_answer': reference_answer,
                    'normalized_reference': normalized_reference,
                    'predicted_answer': predicted_answer,
                    'normalized_predicted': normalized_predicted,
                    'extracted_answer': extracted_answer,
                    'normalized_extracted': normalized_extracted,
                    'finish_solution': finish_solution,
                    'normalized_finish': normalized_finish,
                    'is_correct': is_correct,
                    'extracted_correct': extracted_correct,
                    'finish_correct': finish_correct,
                    'should_be_correct': extracted_correct or finish_correct
                })
            except Exception as e:
                print(f"Error processing line: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Debug answer extraction for AIME2024 benchmark')
    parser.add_argument('output_file', type=str, help='Path to the output.jsonl file')
    parser.add_argument('--save-csv', action='store_true', help='Save results to CSV file')
    args = parser.parse_args()
    
    # Analyze the output file
    results = analyze_output_file(args.output_file)
    
    # Count how many should be correct
    should_be_correct = sum(1 for r in results if r['should_be_correct'])
    actually_correct = sum(1 for r in results if r['is_correct'])
    
    print(f"Total problems: {len(results)}")
    print(f"Actually marked correct: {actually_correct} ({actually_correct/len(results):.2%})")
    print(f"Should be correct: {should_be_correct} ({should_be_correct/len(results):.2%})")
    
    # Print problems that should be correct but aren't
    print("\nProblems that should be correct but aren't:")
    for r in results:
        if r['should_be_correct'] and not r['is_correct']:
            print(f"Instance {r['instance_id']}:")
            print(f"  Reference: {r['reference_answer']} (normalized: {r['normalized_reference']})")
            print(f"  Predicted: {r['predicted_answer']} (normalized: {r['normalized_predicted']})")
            print(f"  Extracted: {r['extracted_answer']} (normalized: {r['normalized_extracted']})")
            print(f"  Finish solution: {r['finish_solution']} (normalized: {r['normalized_finish']})")
            print()
    
    # Save to CSV if requested
    if args.save_csv:
        output_dir = os.path.dirname(args.output_file)
        csv_file = os.path.join(output_dir, 'debug_answers.csv')
        pd.DataFrame(results).to_csv(csv_file, index=False)
        print(f"Results saved to {csv_file}")


if __name__ == '__main__':
    main()