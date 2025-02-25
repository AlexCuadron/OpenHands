# Polyglot Aider Benchmark

This benchmark is based on the [Aider Polyglot Benchmark](https://github.com/Aider-AI/aider/tree/main/benchmark), which evaluates how effectively an agent can translate natural language coding requests into executable code that passes unit tests across multiple programming languages.

## Features

- Supports multiple programming languages (Python, JavaScript, Rust, Go, C++, Java)
- End-to-end evaluation of code editing capabilities
- Automated test execution and validation
- Parallel evaluation with multiple workers
- Detailed metrics and logging

## Usage

1. Make sure you have the required dependencies installed:
   ```bash
   pip install -e .[dev]
   ```

2. Run the benchmark using either style:

   **Old Style (Positional Arguments)**:
   ```bash
   ./scripts/run_infer.sh <model> <commit> <agent> <max_iters> <num_workers>
   ```
   Example:
   ```bash
   ./scripts/run_infer.sh 4ominiSky HEAD CodeActAgent 1000 1
   ```

   **New Style (Named Arguments)**:
   ```bash
   ./scripts/run_infer.sh \
       --agent-cls CodeActAgent \
       --llm-config configs/llm/gpt-4.yaml \
       --eval-output-dir eval_output \
       --eval-num-workers 10
   ```

### Command Line Arguments

**Old Style (Positional)**:
1. `model`: Model name (will look for configs/llm/{model}.yaml)
2. `commit`: Git commit or note to append to output directory
3. `agent`: Agent class name
4. `max_iters`: Maximum iterations per test
5. `num_workers`: Number of parallel workers

**New Style (Named)**:
- `--agent-cls`: The agent class to use (default: CodeActAgent)
- `--llm-config`: Path to the LLM configuration file (required)
- `--eval-output-dir`: Directory to store evaluation outputs (default: eval_output)
- `--eval-num-workers`: Number of parallel workers (default: 1)
- `--eval-n-limit`: Limit the number of test cases to run (-1 for all)
- `--eval-ids`: Comma-separated list of specific test IDs to run
- `--eval-note`: Optional note to append to the output directory name

## Output Format

The benchmark saves its results in the following structure:
```
eval_output/
├── PolyglotAiderBench/
│   ├── CodeActAgent/
│   │   ├── gpt-4_maxiter_10/
│   │   │   ├── infer_logs/
│   │   │   │   └── instance_*.log
│   │   │   ├── llm_completions/
│   │   │   │   └── instance_*/
│   │   │   └── output.jsonl
│   │   └── metadata.json
```

Each instance's results include:
- Test execution results
- LLM completions and costs
- Error tracking (syntax errors, timeouts, etc.)
- Full interaction history

## Supported Languages

The benchmark supports the following languages and test frameworks:
- Python: pytest
- JavaScript: npm test
- Rust: cargo test
- Go: go test
- C++: make test
- Java: Gradle test

## Docker Support

The benchmark runs in a Docker container to safely execute untrusted code. The container image includes all necessary language toolchains and test frameworks.