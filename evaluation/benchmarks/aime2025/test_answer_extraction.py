#!/usr/bin/env python3
"""
Script to test the answer extraction for AIME2025 benchmark.
"""

import re
from typing import Optional

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

    # Look for "The answer is" pattern with variations
    answer_patterns = [
        r'[Tt]he\s+(?:final\s+)?answer\s+is\s+([\d\w\s\.\-\+\/\*\^\{\}\\\(\)]+)',
        r'[Tt]he\s+(?:final\s+)?answer\s+is\s*[:=]\s*([\d\w\s\.\-\+\/\*\^\{\}\\\(\)]+)',
        r'[Tt]he\s+(?:final\s+)?answer\s*[:=]\s*([\d\w\s\.\-\+\/\*\^\{\}\\\(\)]+)',
        r'[Aa]nswer\s*[:=]\s*([\d\w\s\.\-\+\/\*\^\{\}\\\(\)]+)',
        r'[Aa]nswer\s+is\s+([\d\w\s\.\-\+\/\*\^\{\}\\\(\)]+)',
    ]

    for pattern in answer_patterns:
        answer_match = re.search(pattern, text, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()

    # Look for "Therefore" pattern with variations
    therefore_patterns = [
        r'[Tt]herefore,?\s+([\d\w\s\.\-\+\/\*\^\{\}\\\(\)=]+)',
        r'[Tt]hus,?\s+([\d\w\s\.\-\+\/\*\^\{\}\\\(\)=]+)',
        r'[Ss]o,?\s+([\d\w\s\.\-\+\/\*\^\{\}\\\(\)=]+)',
        r'[Hh]ence,?\s+([\d\w\s\.\-\+\/\*\^\{\}\\\(\)=]+)',
    ]

    for pattern in therefore_patterns:
        therefore_match = re.search(pattern, text, re.DOTALL)
        if therefore_match:
            return therefore_match.group(1).strip()

    # Look for "Our answer is" pattern and variations
    our_answer_patterns = [
        r'[Oo]ur\s+answer\s+is\s+([\d\w\s\.\-\+\/\*\^\{\}\\\(\)]+)',
        r'[Ww]e\s+get\s+([\d\w\s\.\-\+\/\*\^\{\}\\\(\)=]+)',
        r'[Ww]e\s+have\s+([\d\w\s\.\-\+\/\*\^\{\}\\\(\)=]+)',
        r'[Ww]e\s+find\s+([\d\w\s\.\-\+\/\*\^\{\}\\\(\)=]+)',
        r'[Tt]his\s+gives\s+us\s+([\d\w\s\.\-\+\/\*\^\{\}\\\(\)=]+)',
    ]

    for pattern in our_answer_patterns:
        our_answer_match = re.search(pattern, text, re.DOTALL)
        if our_answer_match:
            return our_answer_match.group(1).strip()

    # Look for a standalone number at the end of the text (common in AIME problems)
    final_number_patterns = [
        r'(?:^|\n|\.)[\s\t]*(\d+)[\s\t]*$',
        r'(?:^|\n|\.)[^\d]*(\d+)[^\d]*$',
    ]

    for pattern in final_number_patterns:
        final_number_match = re.search(pattern, text)
        if final_number_match:
            return final_number_match.group(1).strip()

    # Look for a number in the last line
    last_line = text.strip().split('\n')[-1].strip()
    if last_line.isdigit():
        return last_line

    # Look for a number surrounded by special characters in the last few lines
    last_few_lines = text.strip().split('\n')[-5:]
    for line in last_few_lines:
        # Look for numbers surrounded by special formatting
        number_in_line = re.search(r'[^\d](\d+)[^\d]', line)
        if number_in_line:
            return number_in_line.group(1).strip()

    return None


def normalize_answer(answer: str) -> str:
    """Normalize the answer for comparison."""
    if answer is None:
        return ''

    # Convert to string if not already
    answer = str(answer)

    # Store the original answer for debugging
    original_answer = answer

    # Remove LaTeX commands
    answer = re.sub(r'\\boxed{(.*?)}', r'\1', answer)  # Extract content from \boxed{}
    answer = re.sub(r'\\left\(|\\right\)', '', answer)

    # Check if the answer contains mathematical expressions like sqrt
    has_math_expr = 'sqrt' in answer.lower() or '\\sqrt' in answer

    # Check if the answer contains currency symbols
    has_currency = '$' in answer or '\\$' in answer or '£' in answer or '€' in answer

    # Remove LaTeX backslashes but keep 'sqrt' intact
    answer = re.sub(r'\\sqrt', 'sqrt', answer)

    # Handle currency symbols - preserve the $ symbol for currency values
    answer = re.sub(r'\\$', '$', answer)  # Convert LaTeX \$ to $

    # Remove other LaTeX backslashes
    answer = re.sub(r'\\', '', answer)

    # Remove all whitespace
    answer = re.sub(r'\s+', '', answer)

    # Remove any text that's not part of the actual answer
    answer = re.sub(r'[Tt]he(final)?answeris', '', answer)
    answer = re.sub(r'[Tt]herefore,?', '', answer)
    answer = re.sub(r'[Tt]hus,?', '', answer)
    answer = re.sub(r'[Ss]o,?', '', answer)
    answer = re.sub(r'[Hh]ence,?', '', answer)
    answer = re.sub(r'[Oo]uranswer(is)?', '', answer)
    answer = re.sub(r'[Ww]eget', '', answer)
    answer = re.sub(r'[Ww]ehave', '', answer)
    answer = re.sub(r'[Ww]efind', '', answer)

    # Handle common mathematical notations
    answer = re.sub(r'[{}()\[\]]', '', answer)  # Remove brackets

    print(f"Normalizing answer: '{original_answer}' -> '{answer}'")

    # If the answer has mathematical expressions, return the normalized form without extracting numbers
    if has_math_expr:
        return answer

    # Handle currency values specially
    if has_currency:
        # Extract the full currency value (including dollars and cents)
        currency_match = re.search(r'(\$\d+\.\d+|\$\d+)', answer)
        if currency_match:
            currency_value = currency_match.group(1)
            # For comparison, keep the full value including the $ symbol
            return currency_value

    # For AIME problems with pure numbers, we typically want just the number
    # Check if the answer is purely numeric
    if re.match(r'^\d+$', answer) or re.match(r'^\d+\.\d+$', answer):
        return answer

    # First, try to extract just the number if it's the last thing in the string
    number_match = re.search(r'(\d+\.\d+|\d+)$', answer)
    if number_match:
        return number_match.group(1)

    # If that fails, try to extract any number from the string
    number_match = re.search(r'(\d+\.\d+|\d+)', answer)
    if number_match:
        return number_match.group(1)

    return answer


def test_answer_extraction():
    """Test the answer extraction function with various formats."""
    test_cases = [
        # Solution tags
        ("<solution>42</solution>", "42"),
        ("<solution>The answer is 42</solution>", "The answer is 42"),
        
        # LaTeX boxed answers
        (r"The answer is \boxed{42}", "42"),
        (r"We get \boxed{123.45}", "123.45"),
        
        # "The answer is" patterns
        ("The answer is 42", "42"),
        ("The final answer is 42", "42"),
        ("The answer is: 42", "42"),
        ("Answer: 42", "42"),
        ("Answer is 42", "42"),
        
        # "Therefore" patterns
        ("Therefore, 42", "42"),
        ("Thus, 42", "42"),
        ("So, 42", "42"),
        ("Hence, 42", "42"),
        
        # "Our answer is" patterns
        ("Our answer is 42", "42"),
        ("We get 42", "42"),
        ("We have 42", "42"),
        ("We find 42", "42"),
        ("This gives us 42", "42"),
        
        # Standalone numbers
        ("After solving the equation, we get\n42", "42"),
        ("The solution is.\n42", "42"),
        
        # Last line
        ("This is a complex problem\nLet's solve it\n42", "42"),
        
        # Numbers with special formatting
        ("The answer is [42]", "42"),
        ("We get (42)", "42"),
    ]
    
    print("Testing answer extraction...")
    for i, (text, expected) in enumerate(test_cases):
        extracted = extract_answer(text)
        normalized = normalize_answer(extracted) if extracted else None
        
        print(f"\nTest case {i+1}:")
        print(f"Text: {text}")
        print(f"Expected: {expected}")
        print(f"Extracted: {extracted}")
        print(f"Normalized: {normalized}")
        print(f"Result: {'✓' if normalized == expected else '✗'}")


if __name__ == "__main__":
    test_answer_extraction()