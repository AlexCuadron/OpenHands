# MATH-500 Benchmark

This benchmark evaluates LLM performance on the MATH-500 dataset, a subset of 500 problems from the MATH benchmark created by OpenAI in their "Let's Verify Step by Step" paper.

## Dataset

The MATH-500 dataset contains 500 math problems across various subjects:
- Algebra (124 problems)
- Intermediate Algebra (97 problems)
- Prealgebra (82 problems)
- Number Theory (62 problems)
- Precalculus (56 problems)
- Geometry (41 problems)
- Counting & Probability (38 problems)

Each problem includes:
- A problem statement
- A detailed solution
- The final answer
- Subject category
- Difficulty level (1-5)
- Unique ID

## Evaluation

The benchmark evaluates the LLM's ability to solve math problems with the help of a Python interpreter tool. The LLM is expected to:

1. Understand the math problem
2. Use the Python interpreter to help solve the problem when needed
3. Provide the final answer in the correct format

The evaluation compares the LLM's answer with the ground truth answer provided in the dataset.

## Usage

To run the benchmark:

```bash
cd /workspace/OpenHands
python -m evaluation.benchmarks.math500.run_infer --llm_config <llm_config> --agent_cls CodeActAgent --eval_output_dir <output_dir>
```

## Metrics

The benchmark reports the following metrics:
- Overall accuracy
- Accuracy by subject
- Accuracy by difficulty level