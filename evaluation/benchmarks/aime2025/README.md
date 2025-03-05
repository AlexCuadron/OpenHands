# AIME2025 Benchmark with Prefix-Based LLM

This benchmark is designed to evaluate the performance of the OpenHands agent on the AIME2025 dataset. It includes a special feature that uses a prefix-based LLM approach, where the assistant's previous responses and observations are combined into a growing narrative that's included as a prefix in subsequent turns.

## Running the Benchmark

To run the benchmark with the prefix-based LLM approach (default):

```bash
./evaluation/benchmarks/aime2025/scripts/run_infer.sh limo HEAD CodeActAgent 1 1 "" eval ipython_only
```

To run the benchmark without the prefix-based LLM approach:

```bash
./evaluation/benchmarks/aime2025/scripts/run_infer.sh limo HEAD CodeActAgent 1 1 "" eval ipython_only false
```

## Parameters

The `run_infer.sh` script accepts the following parameters:

1. `MODEL_CONFIG`: The model configuration to use (e.g., "limo")
2. `COMMIT_HASH`: The commit hash to use (e.g., "HEAD")
3. `AGENT`: The agent to use (e.g., "CodeActAgent")
4. `EVAL_LIMIT`: The number of examples to evaluate
5. `NUM_WORKERS`: The number of workers to use
6. `EVAL_IDS`: The IDs of the examples to evaluate
7. `RUN_EVALUATION`: Whether to run evaluation after the benchmark (e.g., "eval")
8. `ALLOWED_TOOLS`: The tools to allow (default: "all")
9. `USE_PREFIX`: Whether to use the prefix-based LLM approach (default: "true")

## Prefix-Based LLM Approach

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
  {"role": "assistant", "content": "Let me check <function>get_world_cup_winner(2022)</function>"},
  {"role": "function", "content": "Argentina"},
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
    "content": "Let me check <function>get_world_cup_winner(2022)</function>\nObservation: Argentina",
    "prefix": true
  },
  {
    "role": "user",
    "content": "What was the score?"
  }
]
```