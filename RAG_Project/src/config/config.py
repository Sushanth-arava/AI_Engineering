from dataclasses import dataclass, field
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    # API Keys
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))

    # Pipeline mode: "basic" (Phase 1) or "advanced" (Phase 2, default)
    pipeline_mode: str = "advanced"

    # Model settings
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-3.5-turbo"
    temperature: float = 0.0
    max_tokens: int = 512

    # Chunking
    chunk_size: int = 600
    chunk_overlap: int = 100

    # Retrieval
    top_k: int = 5
    rerank_top_k: int = 3

    # Vector store
    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "wikipedia_rag"

    # Dataset
    dataset_name: str = "rag-datasets/rag-mini-wikipedia"
    corpus_config: str = "text-corpus"
    qa_config: str = "question-answer"

    # Hybrid search weights (advanced mode)
    bm25_weight: float = 0.4
    vector_weight: float = 0.6

    # Evaluation
    eval_output_dir: str = "./eval_results"
    golden_dataset_path: str = "./eval/golden_dataset.json"

    def validate_for_runtime(self) -> None:
        """Call at API/eval startup when OpenAI-backed components will actually be constructed."""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file.")
        if self.pipeline_mode not in ("basic", "advanced"):
            raise ValueError(f"pipeline_mode must be 'basic' or 'advanced', got: {self.pipeline_mode!r}")


def get_config() -> Config:
    return Config()
