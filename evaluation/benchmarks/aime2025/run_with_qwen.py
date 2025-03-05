"""Script to run AIME2025 benchmark with custom Qwen provider."""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the repository root to the Python path
repo_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(repo_root))

logger.info("Setting up environment for Qwen model...")
# Set environment variables for the Qwen model
os.environ["EVAL_LLM_MODEL"] = "hosted_vllm/AlexCuadron/DSR1-Qwen-14B-8a4e8f3a-checkpoint-64"
os.environ["EVAL_LLM_TEMPERATURE"] = "0.0"
os.environ["EVAL_LLM_API_KEY"] = "ddd"
os.environ["EVAL_LLM_MAX_INPUT_TOKENS"] = "4096"
os.environ["EVAL_LLM_MAX_OUTPUT_TOKENS"] = "4096"
os.environ["EVAL_LLM_BASE_URL"] = "http://127.0.0.1:8001/v1/"
os.environ["EVAL_LLM_CUSTOM_PROVIDER"] = "custom_qwen"

# Import the custom provider to register it
try:
    from openhands.custom_qwen_provider import custom_qwen_completion
    logger.info("Successfully imported and registered custom_qwen_provider")
except Exception as e:
    logger.error(f"Error importing custom_qwen_provider: {e}")
    logger.warning("Continuing without custom provider...")

if __name__ == "__main__":
    logger.info(f"Running with arguments: {sys.argv}")
    
    # Import the run_infer module
    try:
        from evaluation.benchmarks.aime2025.run_infer import main as run_infer_main
        logger.info("Successfully imported run_infer_main")
        
        # Run the benchmark with the original arguments
        # We don't modify sys.argv, so all arguments passed to this script
        # will be passed directly to run_infer_main
        try:
            logger.info("Starting benchmark execution...")
            run_infer_main()
            logger.info("Benchmark execution completed successfully")
        except Exception as e:
            logger.error(f"Error running benchmark: {e}")
            import traceback
            logger.error(traceback.format_exc())
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error importing run_infer_main: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)