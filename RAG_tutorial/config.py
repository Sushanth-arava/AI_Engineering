from typing import Literal, Optional
from pydantic import BaseModel, Field


class ChunkingConfig(BaseModel):
    strategy: Literal["recursive", "token", "markdown", "semantic"] = "recursive"
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=100, ge=0)
    separators: Optional[list[str]] = None
    semantic_threshold: Optional[float] = None


class RetrievalConfig(BaseModel):
    k: int = Field(default=3, gt=0)


class ModelConfig(BaseModel):
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o"


class StorageConfig(BaseModel):
    persist_directory: str = "db/chromaDB"


class RAGConfig(BaseModel):
    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    models: ModelConfig = ModelConfig()
    storage: StorageConfig = StorageConfig()


config = RAGConfig()
