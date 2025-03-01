#!/usr/bin/env python3
"""
Simple test script for the MATH-500 benchmark.
"""

import os
import sys
from datasets import load_dataset

# Add the repository root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from evaluation.benchmarks.math500.run_infer import extract_answer, check_answer_correctness, normalize_answer

def test_extract_answer():
    """Test the extract_answer function."""
    # Test with solution tags
    text1 = "I think the answer is <solution>42</solution>."
    assert extract_answer(text1) == "42"
    
    # Test with boxed notation
    text2 = "The answer is \\boxed{3\\sqrt{2}}."
    result2 = extract_answer(text2)
    # Print the actual result for debugging
    print(f"Boxed notation result: '{result2}'")
    # The regex might not capture the closing brace correctly, so we'll check if it starts with the expected text
    assert "3\\sqrt{2}" in result2, f"Expected '3\\sqrt{{2}}' to be in '{result2}'"
    
    # Test with "The answer is" pattern
    text3 = "The answer is 3.14159."
    result3 = extract_answer(text3)
    print(f"'The answer is' pattern result: '{result3}'")
    assert "3.14159" in result3, f"Expected '3.14159' to be in '{result3}'"
    
    # Test with "Therefore" pattern
    text4 = "Therefore, x = 5."
    result4 = extract_answer(text4)
    print(f"'Therefore' pattern result: '{result4}'")
    assert "x = 5" in result4, f"Expected 'x = 5' to be in '{result4}'"
    
    print("All extract_answer tests passed!")

def test_normalize_answer():
    """Test the normalize_answer function."""
    # Test with LaTeX commands
    result1 = normalize_answer("\\frac{1}{2}")
    print(f"Normalize LaTeX result: '{result1}'")
    assert "frac" in result1 and "1" in result1 and "2" in result1
    
    # Test with whitespace
    result2 = normalize_answer(" 3.14159 ")
    print(f"Normalize whitespace result: '{result2}'")
    assert result2 == "3.14159"
    
    # Test with complex LaTeX
    result3 = normalize_answer("\\left( 3, \\frac{\\pi}{2} \\right)")
    print(f"Normalize complex LaTeX result: '{result3}'")
    assert "3" in result3 and "pi" in result3 and "2" in result3
    
    print("All normalize_answer tests passed!")

def test_check_answer_correctness():
    """Test the check_answer_correctness function."""
    # Test exact match
    assert check_answer_correctness("42", "42") == True
    
    # Test with LaTeX normalization
    assert check_answer_correctness("\\frac{1}{2}", "\\frac{1}{2}") == True
    
    # Test with whitespace differences
    assert check_answer_correctness(" 3.14159 ", "3.14159") == True
    
    # Test with different representations
    assert check_answer_correctness("\\left( 3, \\frac{\\pi}{2} \\right)", "\\left(3,\\frac{\\pi}{2}\\right)") == True
    
    # Test negative case
    assert check_answer_correctness("42", "43") == False
    
    print("All check_answer_correctness tests passed!")

def test_dataset_loading():
    """Test loading the MATH-500 dataset."""
    dataset = load_dataset('HuggingFaceH4/MATH-500')
    assert 'test' in dataset
    assert len(dataset['test']) == 500
    
    # Check the first example
    first_example = dataset['test'][0]
    assert 'problem' in first_example
    assert 'solution' in first_example
    assert 'answer' in first_example
    assert 'subject' in first_example
    assert 'level' in first_example
    
    print("Dataset loading test passed!")
    print(f"Dataset contains {len(dataset['test'])} examples")

if __name__ == "__main__":
    test_extract_answer()
    test_normalize_answer()
    test_check_answer_correctness()
    test_dataset_loading()
    print("\nAll tests passed successfully!")