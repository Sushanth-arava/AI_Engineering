import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_config
from src.main import build_pipeline
from src.routes import router

logger = logging.getLogger(__name__)


def build_app_state():
    config = get_config()
    config.validate_for_runtime()
    pipeline = build_pipeline(config)
    return {
        "config": config,
        "pipeline": pipeline,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.services = build_app_state()
    logger.info("RAG API ready.")
    try:
        yield
    finally:
        app.state.services.clear()


app = FastAPI(
    title="RAG Wikipedia Q&A API",
    description="Retrieval-Augmented Generation pipeline over Wikipedia.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
