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

### Implementation Details

This benchmark uses a custom implementation of the `finish` tool that accepts an optional `answer` parameter. The answer is stored in both the `outputs` dictionary and the new `solution` parameter of the `AgentFinishAction` class.

The `solution` parameter is a more general way to provide a solution or answer to a task, and it can be used by other benchmarks as well. The benchmark will first check for a solution in the `solution` parameter, and if not found, it will fall back to the `outputs` dictionary for backward compatibility.

## Usage

### Docker Image

The benchmark uses a custom Docker image with pre-installed math libraries (numpy, matplotlib, sympy, scipy). The image will be built automatically when you run the benchmark for the first time.

To build the Docker image manually:

```bash
cd /workspace/OpenHands
./evaluation/benchmarks/math500/scripts/build_docker.sh
```

### Running the Benchmark

To run the benchmark:

```bash
cd /workspace/OpenHands
python -m evaluation.benchmarks.math500.run_infer --llm-config <llm_config> --agent-cls CodeActAgent --eval-output-dir <output_dir>
```

Or use the provided script:

```bash
./evaluation/benchmarks/math500/scripts/run_infer.sh <model_config> HEAD CodeActAgent <eval_limit> <num_workers>
```

## Metrics

The benchmark reports the following metrics:
- Overall accuracy
- Accuracy by subject
- Accuracy by difficulty level
