"""Pipeline builder — single entry point for API and eval."""

import logging

logger = logging.getLogger(__name__)


def build_pipeline(config, mode: str = None):
    """
    Build and return a RAGPipeline.

    Args:
        config: Config instance.
        mode: Override pipeline mode — "basic" or "advanced". If None, uses config.pipeline_mode.
              Pass explicitly from tests or CLI to avoid mutating global config.
    """
    resolved_mode = mode or config.pipeline_mode
    if resolved_mode == "basic":
        return _build_phase1(config)
    return _build_phase2(config)


def _build_phase1(config):
    """Phase 1: ChromaDB vector search only."""
    from src.ingestion import DataProcessor
    from src.retrieval import VectorStore
    from src.pipeline import RAGPipeline

    processor = DataProcessor(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    vs = VectorStore(
        persist_dir=config.chroma_persist_dir,
        collection_name=config.collection_name,
        embedding_model=config.embedding_model,
    )

    if vs.is_empty():
        logger.info("Vector store empty — ingesting dataset (basic mode)...")
        chunks = processor.process(config.dataset_name)
        vs.add_chunks(chunks)
        logger.info(f"Stored {vs.count()} chunks.")

    return RAGPipeline(
        retriever=vs,
        llm_model=config.llm_model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )


def _build_phase2(config):
    """Phase 2: BM25 + vector hybrid search, cross-encoder reranking."""
    from src.ingestion import DataProcessor
    from src.retrieval import VectorStore, HybridSearch, CrossEncoderReranker
    from src.pipeline import RAGPipeline

    processor = DataProcessor(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    vs = VectorStore(
        persist_dir=config.chroma_persist_dir,
        collection_name=config.collection_name,
        embedding_model=config.embedding_model,
    )

    chunks = None
    if vs.is_empty():
        logger.info("Vector store empty — ingesting dataset (advanced mode)...")
        chunks = processor.process(config.dataset_name)
        vs.add_chunks(chunks)
        logger.info(f"Stored {vs.count()} chunks.")

    if chunks is None:
        logger.info("Loading corpus for BM25 index...")
        docs = processor.load_corpus(config.dataset_name)
        chunks = processor.chunk_documents(docs)

    hs = HybridSearch(vs, bm25_weight=config.bm25_weight, vector_weight=config.vector_weight)
    hs.build_bm25_index(chunks)

    return RAGPipeline(
        retriever=hs,
        llm_model=config.llm_model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        reranker=CrossEncoderReranker(),
    )
