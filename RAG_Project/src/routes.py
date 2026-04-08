"""API route handlers."""

import logging

from fastapi import APIRouter, HTTPException, Request

from src.schemas import ChatRequest, ChatResponse, Citation, HealthResponse, IngestResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health(request: Request):
    # Reach into the pipeline's retriever to get a live count.
    # Works for both VectorStore (basic) and HybridSearch (advanced, wraps a VectorStore).
    pipeline = request.app.state.services.get("pipeline")
    if pipeline is None:
        return HealthResponse(status="ok", chunks_in_store=None)
    retriever = pipeline.retriever
    vs = getattr(retriever, "vector_store", retriever)  # HybridSearch.vector_store or VectorStore
    return HealthResponse(status="ok", chunks_in_store=vs.count())


@router.post("/chat", response_model=ChatResponse, tags=["RAG"])
def chat(req: ChatRequest, request: Request):
    """Ask a question. Returns an answer grounded in retrieved Wikipedia chunks."""
    pipeline = request.app.state.services.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")

    try:
        result = pipeline.query(
            req.question,
            top_k=req.top_k,
            check_faithfulness=req.check_faithfulness,
        )
    except Exception as exc:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=str(exc))

    return ChatResponse(
        question=req.question,
        answer=result["answer"],
        citations=[Citation(**c) for c in result["citations"]],
        refused=result["refused"],
    )


@router.post("/ingest", response_model=IngestResponse, tags=["RAG"])
def ingest(request: Request):
    """
    Full rebuild of the vector store and pipeline.

    Resets the ChromaDB collection, re-ingests the entire dataset, rebuilds
    the BM25 index (advanced mode), and replaces the live pipeline in state.
    This is NOT an incremental update — subsequent /chat calls will use the
    freshly ingested corpus immediately.
    """
    from src.retrieval import VectorStore
    from src.main import build_pipeline

    services = request.app.state.services
    config = services.get("config")
    if config is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")

    try:
        # Reset the collection on disk so build_pipeline() sees an empty store
        # and triggers a full ingest.
        vs = VectorStore(
            persist_dir=config.chroma_persist_dir,
            collection_name=config.collection_name,
            embedding_model=config.embedding_model,
        )
        vs.reset()

        # Rebuild the full pipeline (ingest + BM25 index if advanced mode).
        new_pipeline = build_pipeline(config)
        services["pipeline"] = new_pipeline

        # Report count from the new pipeline's retriever.
        retriever = new_pipeline.retriever
        live_vs = getattr(retriever, "vector_store", retriever)
        count = live_vs.count()
        logger.info(f"Full rebuild complete. {count} chunks stored.")
        return IngestResponse(message="Full rebuild complete.", chunks_stored=count)
    except Exception as exc:
        logger.exception("Ingestion error")
        raise HTTPException(status_code=500, detail=str(exc))
