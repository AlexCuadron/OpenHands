# MATH Benchmark Evaluation

This folder contains evaluation harness for evaluating agents on the [MATH benchmark](https://github.com/hendrycks/math).

MATH is a dataset of 12,500 challenging competition mathematics problems. Each problem in MATH has a full step-by-step solution which can be used to teach models to generate answer derivations and explanations.

## Setup Environment and LLM Configuration

Please follow instruction [here](../../README.md#setup) to setup your local development environment and LLM.

## Start the evaluation

```bash
./evaluation/benchmarks/math/scripts/run_infer.sh [model_config] [git-version] [agent] [eval_limit] [eval-num-workers] [eval_ids] [run_evaluation]
```

- `model_config`, e.g. `eval_gpt4_1106_preview`, is the config group name for your LLM settings, as defined in your `config.toml`.
- `git-version`, e.g. `HEAD`, is the git commit hash of the OpenHands version you would like to evaluate. It could also be a release tag like `0.9.0`.
- `agent`, e.g. `CodeActAgent`, is the name of the agent for benchmarks, defaulting to `CodeActAgent`.
- `eval_limit`, e.g. `10`, limits the evaluation to the first `eval_limit` instances. By default, the script evaluates the entire test set.
- `eval-num-workers`: the number of workers to use for evaluation. Default: `1`.
- `eval_ids`, e.g. `"1,3,10"`, limits the evaluation to instances with the given IDs (comma separated).
- `run_evaluation`: set to `eval` to automatically run evaluation after the benchmark completes.

Following is the basic command to start the evaluation:

```bash
# Run benchmark without evaluation
./evaluation/benchmarks/math/scripts/run_infer.sh eval_gpt35_turbo HEAD CodeActAgent 10 1 "1,3,10"

# Run benchmark with automatic evaluation
./evaluation/benchmarks/math/scripts/run_infer.sh eval_gpt35_turbo HEAD CodeActAgent 10 1 "1,3,10" eval
```

## Summarize Results

```bash
poetry run python ./evaluation/benchmarks/math/scripts/summarize_results.py [path_to_output_jsonl_file]
```

Full example:

```bash
poetry run python ./evaluation/benchmarks/math/scripts/summarize_results.py evaluation/evaluation_outputs/outputs/MATH/CodeActAgent/gpt-4o-2024-05-13@20240620_maxiter_30_N_v1.9/output.jsonl
```