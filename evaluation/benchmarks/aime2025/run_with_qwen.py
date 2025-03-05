"""Script to run AIME2025 benchmark with custom Qwen provider."""

import os
import sys
import argparse
from pathlib import Path

# Add the repository root to the Python path
repo_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(repo_root))

# Import the custom provider to register it
try:
    from openhands.custom_qwen_provider import custom_qwen_completion
    print("Successfully imported custom_qwen_provider")
except Exception as e:
    print(f"Error importing custom_qwen_provider: {e}")
    print("Continuing without custom provider...")

# Import the run_infer module
try:
    from evaluation.benchmarks.aime2025.run_infer import main as run_infer_main
    print("Successfully imported run_infer_main")
except Exception as e:
    print(f"Error importing run_infer_main: {e}")
    sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AIME2025 benchmark with custom Qwen provider")
    parser.add_argument("--dataset", type=str, default="aime2025-I", help="Dataset to use (aime2025-I or aime2025-II)")
    parser.add_argument("--output_dir", type=str, default="evaluation_outputs/aime2025_qwen", help="Output directory")
    parser.add_argument("--agent", type=str, default="CodeActAgent", help="Agent to use")
    parser.add_argument("--allowed_tools", type=str, default="ipython_only", help="Tools to allow (ipython_only, bash_only, no_editor, all)")
    parser.add_argument("--max_iterations", type=int, default=20, help="Maximum number of iterations")
    
    args = parser.parse_args()
    
    # Set environment variables for the benchmark
    os.environ["EVAL_LLM_MODEL"] = "hosted_vllm/AlexCuadron/DSR1-Qwen-14B-8a4e8f3a-checkpoint-64"
    os.environ["EVAL_LLM_TEMPERATURE"] = "0.0"
    os.environ["EVAL_LLM_API_KEY"] = "ddd"
    os.environ["EVAL_LLM_MAX_INPUT_TOKENS"] = "4096"
    os.environ["EVAL_LLM_MAX_OUTPUT_TOKENS"] = "4096"
    os.environ["EVAL_LLM_BASE_URL"] = "http://127.0.0.1:8001/v1/"
    os.environ["EVAL_LLM_CUSTOM_PROVIDER"] = "custom_qwen"
    
    # Set up the command line arguments for run_infer_main
    sys.argv = [
        sys.argv[0],
        "--dataset", args.dataset,
        "--output_dir", args.output_dir,
        "--agent", args.agent,
        "--allowed_tools", args.allowed_tools,
        "--max_iterations", str(args.max_iterations),
    ]
    
    # Run the benchmark
    try:
        run_infer_main()
    except Exception as e:
        print(f"Error running benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)