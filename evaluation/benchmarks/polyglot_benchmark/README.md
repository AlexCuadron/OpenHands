# Polyglot Benchmark

This benchmark is based on the [Aider Polyglot Benchmark](https://github.com/Aider-AI/polyglot-benchmark), which evaluates how effectively an agent can translate natural language coding requests into executable code that passes unit tests across multiple programming languages.

## Features

- Supports multiple programming languages (Python, JavaScript, Rust, Go, C++, Java)
- End-to-end evaluation of code editing capabilities
- Automated test execution and validation
- Parallel evaluation with multiple workers
- Detailed metrics and logging

## Setup

1. Clone the polyglot-benchmark repository:
   ```bash
   git clone https://github.com/Aider-AI/polyglot-benchmark.git /workspace/polyglot-benchmark
   ```

2. Build the Docker image for the benchmark:
   ```bash
   ./evaluation/benchmarks/polyglot_benchmark/scripts/build_docker.sh
   ```

## Usage

1. Make sure you have the required dependencies installed:
   ```bash
   pip install -e .[dev]
   ```

2. Run the benchmark:
   ```bash
   ./evaluation/benchmarks/polyglot_benchmark/scripts/run_infer.sh <model_config> <git-version> <agent> <eval_limit> <eval-num-workers> <eval_ids> <eval_languages>
   ```

### Command Line Arguments

- `model_config`: The LLM configuration to use (e.g., `eval_gpt4_1106_preview`)
- `git-version`: Git commit or note to append to output directory (e.g., `HEAD`)
- `agent`: Agent class name (e.g., `CodeActAgent`)
- `eval_limit`: Limit the number of examples to evaluate (default: `-1` for all)
- `eval-num-workers`: Number of parallel workers (default: `1`)
- `eval_ids`: Comma-separated list of specific test IDs to run (e.g., `"1,3,10"`)
- `eval_languages`: Comma-separated list of languages to test (e.g., `"python,javascript,rust"`)

### Environment Variables

You can also set the following environment variables:

```bash
export POLYGLOT_BENCHMARK_PATH="/path/to/polyglot-benchmark"  # Path to the polyglot-benchmark repository
export USE_UNIT_TESTS="true"  # Whether to run unit tests (default: true)
export NO_DOCKER="true"  # Skip Docker container creation and use local runtime (default: false)
export POLYGLOT_DOCKER_IMAGE="image:tag"  # Custom Docker image to use (default: ghcr.io/opendevin/eval-polyglot:v1.0.0)
export BUILD_LOCAL_DOCKER="true"  # Build a local Docker image if one doesn't exist (default: false)
```

### Docker Support

The benchmark uses Docker to create isolated environments for running code in different programming languages. There are two ways to use Docker with this benchmark:

#### Option 1: Build a Local Docker Image

You can build a local Docker image that contains all the necessary tools for the benchmark:

```bash
# Build the Docker image
./evaluation/benchmarks/polyglot_benchmark/scripts/build_local_docker.sh

# Run the benchmark with the local image
./evaluation/benchmarks/polyglot_benchmark/scripts/run_infer.sh eval_gpt4_1106_preview HEAD CodeActAgent 1 1
```

Alternatively, you can set the `BUILD_LOCAL_DOCKER` environment variable:

```bash
BUILD_LOCAL_DOCKER=true ./evaluation/benchmarks/polyglot_benchmark/scripts/run_infer.sh eval_gpt4_1106_preview HEAD CodeActAgent 1 1
```

#### Option 2: Use a Pre-built Docker Image

You can specify a custom Docker image to use:

```bash
POLYGLOT_DOCKER_IMAGE="your-custom-image:tag" ./evaluation/benchmarks/polyglot_benchmark/scripts/run_infer.sh eval_gpt4_1106_preview HEAD CodeActAgent 1 1
```

### Troubleshooting

#### Docker Issues

If you encounter Docker-related errors like:

```
Command 'docker buildx build ...' returned non-zero exit status 1
```

You can try the following solutions:

1. Build a local Docker image as described above.

2. Run with `NO_DOCKER=true` to use the local runtime instead:
   ```bash
   NO_DOCKER=true ./evaluation/benchmarks/polyglot_benchmark/scripts/run_infer.sh eval_gpt4_1106_preview HEAD CodeActAgent 1 1
   ```

3. Make sure Docker is installed and running:
   ```bash
   docker --version
   docker ps
   ```

4. Check if you have permission to use Docker:
   ```bash
   sudo usermod -aG docker $USER
   # Then log out and log back in
   ```

### Example

```bash
# Run evaluation on CodeActAgent for all Python instances with 2 workers
export POLYGLOT_BENCHMARK_PATH="/workspace/polyglot-benchmark"
./evaluation/benchmarks/polyglot_benchmark/scripts/run_infer.sh eval_gpt4_1106_preview HEAD CodeActAgent -1 2 "" "python"
```

## Summarize Results

After running the benchmark, you can summarize the results:

```bash
poetry run python ./evaluation/benchmarks/polyglot_benchmark/scripts/summarize_results.py <path_to_output_jsonl_file>
```

Example:

```bash
poetry run python ./evaluation/benchmarks/polyglot_benchmark/scripts/summarize_results.py evaluation/evaluation_outputs/outputs/PolyglotBenchmark/CodeActAgent/gpt-4-1106-preview_maxiter_30/output.jsonl
```

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