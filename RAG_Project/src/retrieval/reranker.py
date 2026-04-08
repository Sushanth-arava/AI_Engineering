"""Cross-encoder reranker for precision improvement - Phase 2."""

from sentence_transformers import CrossEncoder
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Re-scores retrieved chunks using a cross-encoder model.
    Much more accurate than bi-encoder similarity alone.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        logger.info(f"Loading cross-encoder: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: List[Dict], top_k: int = 3) -> List[Dict]:
        """
        Re-score chunks against the query and return top_k by cross-encoder score.
        Each returned dict gains a 'rerank_score' field.
        """
        if not chunks:
            return []
        pairs = [(query, c["text"]) for c in chunks]
        scores = self.model.predict(pairs)
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)
        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]
