# MATH-500 Benchmark

This benchmark evaluates the mathematical reasoning capabilities of language models using a subset of 500 problems from the MATH dataset, as curated by OpenAI for their "Let's Verify Step by Step" paper.

## Dataset

The MATH-500 dataset contains 500 problems across various mathematical subjects and difficulty levels. Each problem includes:

- A problem statement
- A detailed solution
- The correct answer
- Subject category (e.g., Algebra, Geometry, Calculus)
- Difficulty level (1-5, with 5 being the most difficult)

The dataset is available on Hugging Face: [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)

## Running the Benchmark

To run the benchmark, use the following command:

```bash
python -m evaluation.benchmarks.math500.run_infer --llm_config <llm_config> --agent_cls CodeActAgent --max_iterations 10 --eval_output_dir <output_dir>
```

Optional arguments:
- `--eval_n_limit <n>`: Limit evaluation to the first n instances
- `--eval_ids <id1,id2,...>`: Evaluate only specific instance IDs
- `--eval_num_workers <n>`: Number of parallel workers for evaluation
- `--eval_note <note>`: Add a note to the evaluation output directory name

## Evaluation Metrics

The benchmark evaluates models based on:

1. Accuracy: The percentage of problems for which the model provides the correct answer
2. Subject-wise accuracy: Performance across different mathematical subjects
3. Difficulty-level accuracy: Performance across different difficulty levels

## Implementation Details

The benchmark uses the OpenHands framework to:

1. Present each problem to the model
2. Extract the model's answer from its response
3. Compare the extracted answer with the reference answer
4. Log all interactions and results for analysis

The evaluation logs all LLM completions to enable detailed analysis of the model's reasoning process.