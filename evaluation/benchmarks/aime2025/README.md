# AIME2025 Benchmark

This benchmark evaluates the performance of AI agents on problems from the American Invitational Mathematics Examination (AIME) 2025. The dataset is sourced from [opencompass/AIME2025](https://huggingface.co/datasets/opencompass/AIME2025) on Hugging Face.

## Dataset

The AIME is a challenging mathematics competition for high school students in the United States. The problems require advanced mathematical reasoning and problem-solving skills. The dataset contains problems from the AIME 2025-I and AIME 2025-II competitions.

## Running the Benchmark

### Prerequisites

- Python 3.11+
- OpenHands installed
- Required Python packages: `datasets`, `pandas`, `matplotlib`

### Running a Single Example

To run a single example from the AIME2025 benchmark:

```bash
cd OpenHands
bash evaluation/benchmarks/aime2025/scripts/run_example.sh togetherDeepseek HEAD CodeActAgent 1 1 "0" "" ipython_only
```

This format follows: `<llm-config> <commit-hash> <agent-cls> <eval-limit> <num-workers> <eval-ids> <run-evaluation> <allowed-tools>`

This will run the first problem in the dataset.

### Running the Full Benchmark

To run the full AIME2025 benchmark:

```bash
cd OpenHands
bash evaluation/benchmarks/aime2025/scripts/run_infer.sh togetherDeepseek HEAD CodeActAgent 500 20 "" eval ipython_only
```

### Options

#### Positional Arguments:
1. `MODEL_CONFIG`: LLM configuration to use (required)
2. `COMMIT_HASH`: Git commit hash to use (optional)
3. `AGENT`: Agent class to use (default: "CodeActAgent")
4. `EVAL_LIMIT`: Limit the number of examples to evaluate (default: 0 for full benchmark, 1 for example)
5. `NUM_WORKERS`: Number of workers for parallel evaluation (default: 1)
6. `EVAL_IDS`: Comma-separated list of example IDs to evaluate (default: "" for full benchmark, "0" for example)
7. `RUN_EVALUATION`: Set to "eval" to run evaluation after benchmark
8. `ALLOWED_TOOLS`: Tools allowed for the agent (default: "all")
9. `USE_PREFIX`: Whether to use the prefix-based LLM approach (default: "true")

## Analyzing Results

There are three ways to analyze the results of the benchmark:

### 1. Using the eval_infer.sh script (recommended)

If you already have an output.jsonl file from a previous run, you can analyze it directly:

```bash
bash evaluation/benchmarks/aime2025/scripts/eval_infer.sh <path-to-output-jsonl> [output-directory]
```

Example:
```bash
bash evaluation/benchmarks/aime2025/scripts/eval_infer.sh ./evaluation/evaluation_outputs/AIME2025/CodeActAgent/v0.26.0/output.jsonl
```

### 2. Using the analyze_results.py script directly

```bash
poetry run python evaluation/benchmarks/aime2025/scripts/analyze_results.py <path-to-results-jsonl> --output-dir <output-directory>
```

### 3. Including "eval" in your benchmark run

Simply include "eval" in your command to automatically run the analysis after the benchmark:

```bash
bash evaluation/benchmarks/aime2025/scripts/run_infer.sh togetherDeepseek HEAD CodeActAgent 500 20 "" eval ipython_only
```

All methods will generate:
- A summary of the results in JSON format
- Plots of the overall accuracy and accuracy by problem ID
- A detailed CSV file with the results for each problem

## Benchmark Details

The AIME2025 benchmark evaluates the agent's ability to:
1. Understand complex mathematical problems
2. Apply mathematical reasoning and problem-solving skills
3. Use tools (like Python libraries) to verify calculations and reasoning
4. Arrive at the correct numerical answer

AIME problems typically have integer answers, and the agent is evaluated based on whether it produces the exact correct answer.

## Example Problem

Here's an example problem from the dataset:

> Find the sum of all integer bases $b>9$ for which $17_{b}$ is a divisor of $97_{b}$.

The correct answer is 70.

# Prefix-Based LLM Approach

This benchmark includes a special feature that uses a prefix-based LLM approach, where the assistant's previous responses and observations are combined into a growing narrative that's included as a prefix in subsequent turns.

## Running with Prefix-Based LLM

To run the benchmark with the prefix-based LLM approach (default):

```bash
./evaluation/benchmarks/aime2025/scripts/run_infer.sh limo HEAD CodeActAgent 1 1 "" eval ipython_only
```

To run the benchmark without the prefix-based LLM approach:

```bash
./evaluation/benchmarks/aime2025/scripts/run_infer.sh limo HEAD CodeActAgent 1 1 "" eval ipython_only false
```

## How Prefix-Based LLM Works

The prefix-based LLM approach is implemented in the `conditional_prefix_llm.py` module. It works by:

1. Detecting if we're running the AIME2025 benchmark
2. If so, using the PrefixLLM class instead of the standard LLM class
3. The PrefixLLM class transforms messages into a prefix-based format where the assistant's previous responses and observations are combined into a growing narrative that's included as a prefix in subsequent turns

This approach is particularly useful for models that support the `prefix` parameter (like DeepSeek) and for creating a more coherent conversation flow.

## Example

Original messages:
```json
[
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "Who won the world cup in 2022?"},
  {"role": "assistant", "content": "Let me check <tool>get_world_cup_winner(2022)</tool>"},
  {"role": "tool", "content": "Argentina"},
  {"role": "user", "content": "What was the score?"}
]
```

Transformed messages with prefix-based approach:
```json
[
  {
    "role": "user",
    "content": "You are a helpful assistant.\n\nWho won the world cup in 2022?"
  },
  {
    "role": "assistant",
    "content": "Let me check <tool>get_world_cup_winner(2022)</tool>\nObservation: Argentina",
    "prefix": true
  },
  {
    "role": "user",
    "content": "What was the score?"
  }
]
```
