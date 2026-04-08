"""
RAG pipeline tests — two tiers:
# CI test trigger

  Unit tests (default, no API keys needed):
      pytest tests/

  Integration tests (require OPENAI_API_KEY + network):
      pytest tests/ -m integration
"""

import pytest
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def config():
    from src.config import get_config
    return get_config()


def make_sample_chunks(n=5):
    return [
        {
            "id": f"chunk_{i}",
            "text": f"Sample text for chunk {i}. This discusses topic {i}.",
            "metadata": {"original_id": f"doc_{i}", "source_id": f"doc_{i}"},
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Unit tests — DataProcessor (no external calls)
# ---------------------------------------------------------------------------

class TestDataProcessorUnit:
    def test_chunk_documents_produces_chunks(self):
        from src.ingestion import DataProcessor
        processor = DataProcessor(chunk_size=200, chunk_overlap=20)
        docs = [{"id": "1", "text": "A " * 300, "metadata": {"source_id": "1"}}]
        chunks = processor.chunk_documents(docs)
        assert len(chunks) > 1

    def test_chunk_has_required_fields(self):
        from src.ingestion import DataProcessor
        processor = DataProcessor(chunk_size=200, chunk_overlap=20)
        docs = [{"id": "1", "text": "Hello world. " * 50, "metadata": {"source_id": "1"}}]
        chunks = processor.chunk_documents(docs)
        for chunk in chunks:
            assert "id" in chunk
            assert "text" in chunk
            assert "metadata" in chunk

    def test_chunk_metadata_has_original_id(self):
        from src.ingestion import DataProcessor
        processor = DataProcessor(chunk_size=200, chunk_overlap=20)
        docs = [{"id": "42", "text": "Some text. " * 50, "metadata": {"source_id": "42"}}]
        chunks = processor.chunk_documents(docs)
        assert "original_id" in chunks[0]["metadata"]


# ---------------------------------------------------------------------------
# Unit tests — VectorStore (mocked ChromaDB + embeddings)
# ---------------------------------------------------------------------------

class TestVectorStoreUnit:
    def _make_vs(self, tmp_path):
        from src.retrieval import VectorStore
        with patch("src.retrieval.vector_store.OpenAIEmbeddings") as mock_emb, \
             patch("src.retrieval.vector_store.chromadb.PersistentClient") as mock_client:
            mock_emb.return_value.embed_documents.return_value = [[0.1] * 10] * 5
            mock_emb.return_value.embed_query.return_value = [0.1] * 10
            mock_collection = MagicMock()
            mock_collection.count.return_value = 5
            mock_collection.query.return_value = {
                "ids": [["chunk_0", "chunk_1"]],
                "documents": [["text 0", "text 1"]],
                "metadatas": [[{"original_id": "doc_0"}, {"original_id": "doc_1"}]],
                "distances": [[0.1, 0.2]],
            }
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            vs = VectorStore(persist_dir=str(tmp_path), collection_name="test")
            vs._mock_collection = mock_collection
            return vs

    def test_is_not_empty_after_add(self, tmp_path):
        vs = self._make_vs(tmp_path)
        assert not vs.is_empty()

    def test_search_returns_list(self, tmp_path):
        vs = self._make_vs(tmp_path)
        results = vs.search("test query", top_k=2)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_search_result_has_id(self, tmp_path):
        vs = self._make_vs(tmp_path)
        results = vs.search("test query", top_k=2)
        for r in results:
            assert "id" in r
            assert "text" in r
            assert "score" in r


# ---------------------------------------------------------------------------
# Unit tests — HybridSearch (fake VectorStore)
# ---------------------------------------------------------------------------

class TestHybridSearchUnit:
    def _make_hs(self, chunks):
        from src.retrieval import HybridSearch
        fake_vs = MagicMock()
        fake_vs.search.return_value = [
            {"id": c["id"], "text": c["text"], "metadata": c["metadata"], "score": 0.8}
            for c in chunks[:3]
        ]
        hs = HybridSearch(fake_vs, bm25_weight=0.4, vector_weight=0.6)
        hs.build_bm25_index(chunks)
        return hs

    def test_returns_results(self):
        chunks = make_sample_chunks(10)
        hs = self._make_hs(chunks)
        results = hs.search("topic 2", top_k=3)
        assert len(results) > 0

    def test_result_has_id(self):
        chunks = make_sample_chunks(10)
        hs = self._make_hs(chunks)
        results = hs.search("topic 1", top_k=3)
        for r in results:
            assert "id" in r

    def test_scores_are_non_negative(self):
        chunks = make_sample_chunks(10)
        hs = self._make_hs(chunks)
        results = hs.search("sample text", top_k=5)
        for r in results:
            assert r["score"] >= 0.0

    def test_falls_back_to_vector_if_no_bm25(self):
        from src.retrieval import HybridSearch
        fake_vs = MagicMock()
        fake_vs.search.return_value = [
            {"id": "a", "text": "hello", "metadata": {}, "score": 0.9}
        ]
        hs = HybridSearch(fake_vs)
        # _bm25 intentionally not built
        results = hs.search("hello", top_k=1)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Unit tests — RAGPipeline (mocked LLM + fake retriever)
# ---------------------------------------------------------------------------

class TestRAGPipelineUnit:
    def _make_pipeline(self):
        from src.pipeline import RAGPipeline
        fake_retriever = MagicMock()
        fake_retriever.search.return_value = [
            {
                "id": "chunk_0",
                "text": "Abraham Lincoln was the 16th president of the United States.",
                "metadata": {"original_id": "doc_0"},
                "score": 0.95,
            }
        ]
        with patch("src.pipeline.rag_pipeline.ChatOpenAI") as mock_llm_cls, \
             patch("src.pipeline.rag_pipeline._load_prompt_template", return_value="Q: {question}\nCtx: {context}\nA:"), \
             patch("src.pipeline.rag_pipeline._load_faithfulness_prompt_template", return_value="Faithful? {question} {answer} {context}"):
            mock_llm = MagicMock()
            mock_llm.invoke.return_value.content = "Lincoln was the 16th president."
            mock_llm_cls.return_value = mock_llm
            pipeline = RAGPipeline(retriever=fake_retriever)
            pipeline.llm = mock_llm
            # Patch the chain to return a fixed string
            pipeline.chain = MagicMock()
            pipeline.chain.invoke.return_value = "Lincoln was the 16th president."
        return pipeline, fake_retriever

    def test_query_returns_answer(self):
        pipeline, _ = self._make_pipeline()
        result = pipeline.query("Who was Lincoln?")
        assert "answer" in result
        assert len(result["answer"]) > 0

    def test_query_returns_citations(self):
        pipeline, _ = self._make_pipeline()
        result = pipeline.query("Who was Lincoln?")
        assert "citations" in result
        assert len(result["citations"]) > 0
        assert "source_id" in result["citations"][0]

    def test_faithfulness_fail_closed_on_exception(self):
        pipeline, _ = self._make_pipeline()
        pipeline.llm.invoke.side_effect = Exception("API down")
        # Should refuse rather than pass through
        is_faithful = pipeline._faithfulness_check("q", "a", "ctx")
        assert is_faithful is False

    def test_refused_when_faithfulness_fails(self):
        pipeline, _ = self._make_pipeline()
        pipeline.llm.invoke.return_value.content = "NO"
        result = pipeline.query("Who was Lincoln?", check_faithfulness=True)
        assert result["refused"] is True

    def test_no_chunks_returns_refusal(self):
        from src.pipeline import RAGPipeline
        fake_retriever = MagicMock()
        fake_retriever.search.return_value = []
        with patch("src.pipeline.rag_pipeline.ChatOpenAI"), \
             patch("src.pipeline.rag_pipeline._load_prompt_template", return_value="Q:{question} C:{context}"), \
             patch("src.pipeline.rag_pipeline._load_faithfulness_prompt_template", return_value="F:{question}{answer}{context}"):
            pipeline = RAGPipeline(retriever=fake_retriever)
        result = pipeline.query("Anything?")
        assert result["refused"] is True


# ---------------------------------------------------------------------------
# Integration tests — require OPENAI_API_KEY + HuggingFace network access
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestVectorStoreIntegration:
    @pytest.fixture(scope="class")
    def vs_with_data(self, config, tmp_path_factory):
        from src.ingestion import DataProcessor
        from src.retrieval import VectorStore
        from datasets import load_dataset
        processor = DataProcessor(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
        ds = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")
        split = ds[list(ds.keys())[0]]
        sample = [
            {"id": str(item["id"]), "text": item["passage"], "metadata": {"source_id": str(item["id"])}}
            for item in list(split)[:20]
        ]
        chunks = processor.chunk_documents(sample)
        persist_dir = str(tmp_path_factory.mktemp("chroma_int"))
        vs = VectorStore(persist_dir=persist_dir, collection_name="int_test")
        vs.add_chunks(chunks)
        return vs

    def test_search_returns_relevant_result(self, vs_with_data):
        results = vs_with_data.search("Abraham Lincoln president", top_k=3)
        assert len(results) > 0
        assert any("Lincoln" in r["text"] for r in results)

    def test_score_in_range(self, vs_with_data):
        results = vs_with_data.search("science", top_k=3)
        for r in results:
            assert 0.0 <= r["score"] <= 1.0


@pytest.mark.integration
class TestRAGPipelineIntegration:
    @pytest.fixture(scope="class")
    def pipeline(self, config, tmp_path_factory):
        from src.main import build_pipeline
        import dataclasses
        cfg = dataclasses.replace(
            config,
            chroma_persist_dir=str(tmp_path_factory.mktemp("chroma_full")),
            collection_name="int_pipeline",
        )
        return build_pipeline(cfg, mode="basic")

    def test_query_returns_answer(self, pipeline):
        result = pipeline.query("Who was Abraham Lincoln?")
        assert len(result["answer"]) > 0
        assert not result["refused"]

    def test_query_with_faithfulness(self, pipeline):
        result = pipeline.query("Who was Abraham Lincoln?", check_faithfulness=True)
        assert "answer" in result
        assert "refused" in result
