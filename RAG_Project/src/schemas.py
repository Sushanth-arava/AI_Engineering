"""Pydantic request/response schemas."""

from typing import List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask.")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve.")
    check_faithfulness: bool = Field(
        True, description="Run a faithfulness check before returning the answer."
    )


class Citation(BaseModel):
    source_id: str
    text_snippet: str
    score: float


class ChatResponse(BaseModel):
    question: str
    answer: str
    citations: List[Citation]
    refused: bool


class IngestResponse(BaseModel):
    message: str
    chunks_stored: int


class HealthResponse(BaseModel):
    status: str
    chunks_in_store: Optional[int] = None
