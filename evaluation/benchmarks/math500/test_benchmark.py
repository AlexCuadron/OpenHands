#!/usr/bin/env python3
"""Test the MATH-500 benchmark implementation."""

import os
import sys
from datasets import load_dataset

from evaluation.benchmarks.math500.helper import compare_answers


def test_dataset_loading():
    """Test that the dataset can be loaded."""
    try:
        dataset = load_dataset('HuggingFaceH4/MATH-500')
        print(f"Dataset loaded successfully with {len(dataset['test'])} examples")
        return True
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False


def test_answer_comparison():
    """Test the answer comparison function."""
    test_cases = [
        # Exact matches
        ("42", "42", True),
        # Whitespace differences
        ("42", " 42 ", True),
        # LaTeX formatting differences
        ("\\boxed{42}", "42", True),
        ("\\left(3, \\frac{\\pi}{2}\\right)", "(3, \\frac{\\pi}{2})", True),
        # Different answers
        ("42", "43", False),
        ("\\boxed{42}", "\\boxed{43}", False),
    ]
    
    all_passed = True
    for predicted, reference, expected in test_cases:
        result = compare_answers(predicted, reference)
        if result != expected:
            print(f"Test failed: compare_answers({predicted}, {reference}) = {result}, expected {expected}")
            all_passed = False
        else:
            print(f"Test passed: compare_answers({predicted}, {reference}) = {result}")
    
    return all_passed


def main():
    """Run all tests."""
    tests = [
        ("Dataset loading", test_dataset_loading),
        ("Answer comparison", test_answer_comparison),
    ]
    
    all_passed = True
    for name, test_fn in tests:
        print(f"\nRunning test: {name}")
        if test_fn():
            print(f"✅ {name} test passed")
        else:
            print(f"❌ {name} test failed")
            all_passed = False
    
    print("\nSummary:")
    if all_passed:
        print("✅ All tests passed")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())