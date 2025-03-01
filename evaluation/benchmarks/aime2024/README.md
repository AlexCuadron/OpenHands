# AIME2024 Benchmark

This benchmark evaluates the performance of AI agents on problems from the American Invitational Mathematics Examination (AIME). The dataset is sourced from [AI-MO/aimo-validation-aime](https://huggingface.co/datasets/AI-MO/aimo-validation-aime) on Hugging Face.

## Dataset

The AIME is a challenging mathematics competition for high school students in the United States. The problems require advanced mathematical reasoning and problem-solving skills. The dataset contains 90 problems from various AIME competitions.

## Running the Benchmark

### Prerequisites

- Python 3.11+
- OpenHands installed
- Required Python packages: `datasets`, `pandas`, `matplotlib`

### Running a Single Example

To run a single example from the AIME2024 benchmark, you can use either positional or named arguments:

#### Using positional arguments (compatible with MATH500):

```bash
cd OpenHands
bash evaluation/benchmarks/aime2024/scripts/run_example.sh togetherDeepseek HEAD CodeActAgent 1 1 "0" "" ipython_only
```

This format follows: `<llm-config> <commit-hash> <agent-cls> <eval-limit> <num-workers> <eval-ids> <run-evaluation> <allowed-tools>`

#### Using named arguments:

```bash
cd OpenHands
bash evaluation/benchmarks/aime2024/scripts/run_example.sh --llm-config <your-llm-config>
```

This will run the first problem in the dataset.

### Running the Full Benchmark

To run the full AIME2024 benchmark:

#### Using positional arguments (compatible with MATH500):

```bash
cd OpenHands
bash evaluation/benchmarks/aime2024/scripts/run_infer.sh togetherDeepseek HEAD CodeActAgent 500 20 "" eval ipython_only
```

#### Using named arguments:

```bash
cd OpenHands
bash evaluation/benchmarks/aime2024/scripts/run_infer.sh --llm-config <your-llm-config> --eval-num-workers <num-workers>
```

### Options

#### Positional Arguments:
1. `MODEL_CONFIG`: LLM configuration to use (required)
2. `COMMIT_HASH`: Not used but kept for compatibility with MATH500
3. `AGENT`: Agent class to use (default: "CodeActAgent")
4. `EVAL_LIMIT`: Limit the number of examples to evaluate (default: 0 for full benchmark, 1 for example)
5. `NUM_WORKERS`: Number of workers for parallel evaluation (default: 1)
6. `EVAL_IDS`: Comma-separated list of example IDs to evaluate (default: "" for full benchmark, "0" for example)
7. `RUN_EVALUATION`: Set to "eval" to run evaluation after benchmark
8. `ALLOWED_TOOLS`: Tools allowed for the agent (default: "all")

#### Named Arguments:
- `--agent-cls`: Agent class to use (default: "CodeActAgent")
- `--llm-config`: LLM configuration to use (required)
- `--max-iterations`: Maximum number of iterations (default: 20)
- `--eval-note`: Note for the evaluation (default: "aime2024_benchmark")
- `--eval-output-dir`: Output directory (default: "./evaluation/results/aime2024")
- `--eval-num-workers`: Number of workers for parallel evaluation (default: 1)
- `--eval-n-limit`: Limit the number of examples to evaluate (default: 0, meaning all)
- `--eval-ids`: Comma-separated list of example IDs to evaluate (default: "", meaning all)
- `--allowed-tools`: Tools allowed for the agent (default: "all", options: "all", "ipython_only", "bash_only", "no_editor")

## Analyzing Results

To analyze the results of the benchmark:

```bash
python evaluation/benchmarks/aime2024/scripts/analyze_results.py --results-file <path-to-results-jsonl> --output-dir <output-directory>
```

This will generate:
- A summary of the results in JSON format
- Plots of the overall accuracy and accuracy by problem ID
- A detailed CSV file with the results for each problem

## Benchmark Details

The AIME2024 benchmark evaluates the agent's ability to:
1. Understand complex mathematical problems
2. Apply mathematical reasoning and problem-solving skills
3. Use tools (like Python libraries) to verify calculations and reasoning
4. Arrive at the correct numerical answer

AIME problems typically have integer answers, and the agent is evaluated based on whether it produces the exact correct answer.

## Example Problem

Here's an example problem from the dataset:

> Quadratic polynomials $P(x)$ and $Q(x)$ have leading coefficients $2$ and $-2,$ respectively. The graphs of both polynomials pass through the two points $(16,54)$ and $(20,53).$ Find $P(0) + Q(0).$

The correct answer is 116.