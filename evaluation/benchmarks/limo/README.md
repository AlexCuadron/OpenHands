# LIMO Benchmark

This benchmark evaluates agent performance on the LIMO (Linguistic Math Olympiad) dataset, which contains challenging mathematical problems.

## Dataset

The LIMO dataset consists of 817 mathematical problems from various sources. Each problem has:
- A question
- A detailed solution
- An answer

The dataset is available on Hugging Face: [GAIR/LIMO](https://huggingface.co/datasets/GAIR/LIMO)

## Running the Benchmark

To run the benchmark, use the `run_infer.sh` script in the `scripts` directory:

```bash
./evaluation/benchmarks/limo/scripts/run_infer.sh <MODEL_CONFIG> HEAD CodeActAgent <EVAL_LIMIT> <NUM_WORKERS> <EVAL_IDS> eval <ALLOWED_TOOLS>
```

### Parameters:

- `MODEL_CONFIG`: The LLM configuration to use (e.g., `togetherDeepseek`)
- `EVAL_LIMIT`: Number of problems to evaluate (optional, default: all)
- `NUM_WORKERS`: Number of parallel workers (optional, default: 1)
- `EVAL_IDS`: Specific problem IDs to evaluate, comma-separated (optional)
- `ALLOWED_TOOLS`: Tools allowed for the agent (optional, default: "all")
  - Options: "all", "ipython_only", "bash_only", "no_editor"

### Example:

```bash
./evaluation/benchmarks/limo/scripts/run_infer.sh togetherDeepseek HEAD CodeActAgent 1 1 "" eval ipython_only
```

This will run the benchmark on 1 problem using the togetherDeepseek model with the CodeActAgent, allowing only the IPython tool.

## Evaluation

The benchmark evaluates the agent's ability to:
1. Understand complex mathematical problems
2. Reason through the solution step by step
3. Verify reasoning with code
4. Arrive at the correct answer

Results are saved in the `evaluation/evaluation_outputs` directory and include:
- Raw agent outputs
- Accuracy metrics
- Detailed analysis of correct and incorrect answers

To analyze results after running the benchmark, use:

```bash
python evaluation/benchmarks/limo/scripts/analyze_results.py <PATH_TO_OUTPUT_JSONL> --output-dir <OUTPUT_DIRECTORY>
```

## Implementation Details

The benchmark implementation includes:
- Custom instructions tailored for mathematical problem-solving
- Code verification requirements to ensure agents check their work
- Answer normalization to handle different formats
- Detailed logging and analysis tools

## Customization

You can customize the benchmark by modifying:
- `helper.py`: Instructions and agent response handling
- `run_infer.py`: Core evaluation logic
- `analyze_results.py`: Results analysis and visualization