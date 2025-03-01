#!/usr/bin/env python3
"""Test the MATH-500 benchmark implementation."""

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
        print(f'Error loading dataset: {e}')
        return False


def test_answer_comparison():
    """Test the answer comparison function."""
    test_cases = [
        # Exact matches
        ('42', '42', True),
        # Whitespace differences
        ('42', ' 42 ', True),
        # LaTeX formatting differences
        ('\\boxed{42}', '42', True),
        ('\\left(3, \\frac{\\pi}{2}\\right)', '(3, \\frac{\\pi}{2})', True),
        # Different answers
        ('42', '43', False),
        ('\\boxed{42}', '\\boxed{43}', False),
    ]

    all_passed = True
    for predicted, reference, expected in test_cases:
        result = compare_answers(predicted, reference)
        if result != expected:
            print(
                f'Test failed: compare_answers({predicted}, {reference}) = {result}, expected {expected}'
            )
            all_passed = False
        else:
            print(f'Test passed: compare_answers({predicted}, {reference}) = {result}')

    return all_passed


def test_solution_parameter():
    """Test that the solution parameter works correctly."""
    from evaluation.benchmarks.math500.run_infer import extract_answer_from_history
    from openhands.controller.state.state import State
    from openhands.events.action import AgentFinishAction

    # Create a test state with a finish action that has a solution
    state = State()
    state.history.append(AgentFinishAction(solution='42'))

    # Test that the solution is extracted correctly
    answer = extract_answer_from_history(state)
    if answer != '42':
        print(
            f'Test failed: extract_answer_from_history returned {answer}, expected 42'
        )
        return False
    else:
        print(
            'Test passed: extract_answer_from_history correctly extracted solution parameter'
        )

    # Create a test state with a finish action that has an answer in outputs
    state = State()
    state.history.append(AgentFinishAction(outputs={'answer': '43'}))

    # Test that the answer is extracted correctly
    answer = extract_answer_from_history(state)
    if answer != '43':
        print(
            f'Test failed: extract_answer_from_history returned {answer}, expected 43'
        )
        return False
    else:
        print(
            'Test passed: extract_answer_from_history correctly extracted answer from outputs'
        )

    # Create a test state with a finish action that has both solution and answer in outputs
    state = State()
    state.history.append(AgentFinishAction(solution='44', outputs={'answer': '45'}))

    # Test that the solution is preferred over the answer in outputs
    answer = extract_answer_from_history(state)
    if answer != '44':
        print(
            f'Test failed: extract_answer_from_history returned {answer}, expected 44'
        )
        return False
    else:
        print(
            'Test passed: extract_answer_from_history correctly preferred solution over outputs'
        )

    return True


def main():
    """Run all tests."""
    tests = [
        ('Dataset loading', test_dataset_loading),
        ('Answer comparison', test_answer_comparison),
        ('Solution parameter', test_solution_parameter),
    ]

    all_passed = True
    for name, test_fn in tests:
        print(f'\nRunning test: {name}')
        if test_fn():
            print(f'✅ {name} test passed')
        else:
            print(f'❌ {name} test failed')
            all_passed = False

    print('\nSummary:')
    if all_passed:
        print('✅ All tests passed')
        return 0
    else:
        print('❌ Some tests failed')
        return 1


if __name__ == '__main__':
    sys.exit(main())
