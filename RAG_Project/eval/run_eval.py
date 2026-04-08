"""Run offline RAG evaluation against golden dataset - Phase 3."""

import sys
import os
import logging
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline")
    parser.add_argument("--golden", default=None, help="Path to golden dataset JSON")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", default=None, choices=["basic", "advanced"],
                        help="Override pipeline mode (default: config.pipeline_mode)")
    parser.add_argument("--fail-on-threshold", action="store_true",
                        help="Exit with code 1 if any metric is below threshold (CI mode)")
    args = parser.parse_args()

    from src.config import get_config
    from src.main import build_pipeline
    from src.pipeline.evaluation import RAGEvaluator

    config = get_config()
    config.validate_for_runtime()
    golden_path = args.golden or config.golden_dataset_path

    pipeline = build_pipeline(config, mode=args.mode)
    evaluator = RAGEvaluator(pipeline, output_dir=config.eval_output_dir)

    scores = evaluator.evaluate(golden_path, top_k=args.top_k)
    logger.info(f"Final scores: {scores}")

    if args.fail_on_threshold:
        passed = evaluator.check_thresholds(scores)
        if not passed:
            logger.error("Evaluation failed — scores below threshold.")
            sys.exit(1)
        else:
            logger.info("All thresholds passed.")


if __name__ == "__main__":
    main()
