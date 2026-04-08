"""Hybrid retrieval: BM25 keyword search + vector semantic search.

BM25 index is held in memory. Suitable for corpora up to ~100k chunks.
For larger corpora, replace with a persistent keyword index (e.g. Elasticsearch).
"""

from rank_bm25 import BM25Okapi
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class HybridSearch:
    """
    Results are keyed by chunk ID (not text) so duplicate-text chunks are
    preserved as distinct entries throughout fusion and reranking.
    """

    def __init__(self, vector_store, bm25_weight: float = 0.4, vector_weight: float = 0.6):
        self.vector_store = vector_store
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self._bm25 = None
        self._corpus_chunks: List[Dict] = []

    def build_bm25_index(self, chunks: List[Dict]) -> None:
        
        self._corpus_chunks = chunks
        tokenized = [c["text"].lower().split() for c in chunks]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 index built with {len(chunks)} chunks.")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Hybrid search: fuse BM25 + vector scores, return top_k by combined score.

        Both scores are normalized to [0, 1] before fusion:
            fused = vector_weight * vector_score + bm25_weight * bm25_score
        """
        # Vector search — fetch more candidates than needed for fusion
        vector_results = self.vector_store.search(query, top_k=top_k * 2)

        if self._bm25 is None:
            logger.warning("BM25 index not built — falling back to vector-only search.")
            return vector_results[:top_k]

        # Build vector lookup keyed by chunk ID
        vector_by_id: Dict[str, Dict] = {r["id"]: r for r in vector_results}

        # BM25 scores for all corpus chunks.
        # BM25Okapi IDF can go negative for very common terms, producing negative
        # raw scores. Clamp to 0 before normalizing so fused scores stay in [0, 1].
        tokens = query.lower().split()
        bm25_raw = self._bm25.get_scores(tokens)
        bm25_clamped = [max(0.0, s) for s in bm25_raw]
        max_bm25 = max(bm25_clamped) if max(bm25_clamped) > 0 else 1.0
        bm25_by_id: Dict[str, float] = {
            self._corpus_chunks[i]["id"]: score / max_bm25
            for i, score in enumerate(bm25_clamped)
        }

        # Fuse over the union of candidate IDs
        all_ids = set(vector_by_id) | set(bm25_by_id)
        fused: Dict[str, float] = {}
        for chunk_id in all_ids:
            v_score = vector_by_id[chunk_id]["score"] if chunk_id in vector_by_id else 0.0
            b_score = bm25_by_id.get(chunk_id, 0.0)
            fused[chunk_id] = self.vector_weight * v_score + self.bm25_weight * b_score

        # Build result list; prefer the vector result dict (has score field) over raw corpus chunk
        id_to_chunk: Dict[str, Dict] = {c["id"]: c for c in self._corpus_chunks}
        id_to_chunk.update({r["id"]: r for r in vector_results})

        sorted_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)[:top_k]
        return [
            {**id_to_chunk[cid], "score": fused[cid]}
            for cid in sorted_ids
            if cid in id_to_chunk
        ]
