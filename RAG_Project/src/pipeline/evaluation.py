"""Offline evaluation pipeline using RAGAS - Phase 3."""

import json
import os
from typing import List, Dict
import logging
from ragas import evaluate as ragas_evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Evaluates the RAG pipeline against a golden QA dataset.
    Metrics: faithfulness, answer_relevancy, context_recall, context_precision.
    Phase 3: wire into CI to gate on quality thresholds.
    """

    # Minimum acceptable scores for CI gating (Phase 3)
    THRESHOLDS = {
        "faithfulness": 0.7,
        "answer_relevancy": 0.7,
        "context_recall": 0.5,
        "context_precision": 0.5,
    }

    def __init__(self, pipeline, output_dir: str = "./eval_results"):
        self.pipeline = pipeline
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def load_golden_dataset(self, path: str) -> List[Dict]:
        """Load golden QA pairs from JSON file."""
        with open(path) as f:
            return json.load(f)

    def run_pipeline_on_dataset(
        self, qa_pairs: List[Dict], top_k: int = 5
    ) -> List[Dict]:
        """Run the pipeline on all QA pairs and collect results."""
        results = []
        for i, item in enumerate(qa_pairs):
            logger.info(f"Evaluating {i+1}/{len(qa_pairs)}: {item['question'][:60]}...")
            output = self.pipeline.query(item["question"], top_k=top_k, check_faithfulness=True)
            results.append(
                {
                    "question": item["question"],
                    "answer": output["answer"],
                    "ground_truth": item["answer"],
                    "contexts": [c["text"] for c in output["chunks_used"]],
                    "refused": output["refused"],
                }
            )
        return results

    def evaluate(self, golden_dataset_path: str, top_k: int = 5) -> Dict:
        """
        Full eval: run pipeline + compute RAGAS metrics.
        Returns dict of metric_name -> score.
        """
        qa_pairs = self.load_golden_dataset(golden_dataset_path)
        results = self.run_pipeline_on_dataset(qa_pairs, top_k=top_k)

        dataset = EvaluationDataset(
            samples=[
                SingleTurnSample(
                    user_input=r["question"],
                    response=r["answer"],
                    reference=r["ground_truth"],
                    retrieved_contexts=r["contexts"],
                )
                for r in results
            ]
        )

        scores = ragas_evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        )
        score_dict = scores.to_pandas().select_dtypes(include="number").mean().to_dict()
        logger.info(f"Evaluation scores: {score_dict}")

        # Save results
        output_path = os.path.join(self.output_dir, "eval_results.json")
        with open(output_path, "w") as f:
            json.dump({"scores": score_dict, "details": results}, f, indent=2)
        logger.info(f"Results saved to {output_path}")

        return score_dict

    def check_thresholds(self, scores: Dict) -> bool:
        """
        Phase 3 CI gate: returns True if all metrics pass thresholds.
        Logs failures for each metric below threshold.
        """
        passed = True
        for metric, threshold in self.THRESHOLDS.items():
            score = scores.get(metric, 0.0)
            if score < threshold:
                logger.error(f"FAILED: {metric} = {score:.3f} < threshold {threshold}")
                passed = False
            else:
                logger.info(f"PASSED: {metric} = {score:.3f} >= threshold {threshold}")
        return passed
