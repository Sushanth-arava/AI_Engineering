"""Document ingestion and chunking pipeline - Phase 1."""

from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """Loads rag-mini-wikipedia dataset and chunks passages for indexing."""

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def load_corpus(self, dataset_name: str = "rag-datasets/rag-mini-wikipedia") -> List[Dict]:
        
        logger.info(f"Loading corpus from {dataset_name}...")
        ds = load_dataset(dataset_name, "text-corpus")
        passages = ds["passages"] if "passages" in ds else ds[list(ds.keys())[0]]
        docs = []
        for item in passages:
            docs.append({
                "id": str(item["id"]),
                "text": item["passage"],
                "metadata": {"source_id": str(item["id"])},
            })
        logger.info(f"Loaded {len(docs)} passages.")
        return docs

    def load_qa_pairs(self, dataset_name: str = "rag-datasets/rag-mini-wikipedia") -> List[Dict]:
        
        ds = load_dataset(dataset_name, "question-answer")
        split = ds["test"] if "test" in ds else ds[list(ds.keys())[0]]
        qa_pairs = []
        for item in split:
            qa_pairs.append({
                "id": str(item["id"]),
                "question": item["question"],
                "answer": item["answer"],
            })
        logger.info(f"Loaded {len(qa_pairs)} QA pairs.")
        return qa_pairs

    def chunk_documents(self, docs: List[Dict]) -> List[Dict]:
        """
        Chunk documents into ~600-char segments with 100-char overlap (Phase 1).
        Returns list of dicts with: id, text, chunk_index, metadata.
        """
        chunks = []
        for doc in docs:
            splits = self.splitter.split_text(doc["text"])
            for i, chunk_text in enumerate(splits):
                chunks.append({
                    "id": f"{doc['id']}_chunk_{i}",
                    "text": chunk_text,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_index": i,
                        "original_id": doc["id"],
                    },
                })
        logger.info(f"Created {len(chunks)} chunks from {len(docs)} documents.")
        return chunks

    def process(self, dataset_name: str = "rag-datasets/rag-mini-wikipedia") -> List[Dict]:
        """Full ingestion pipeline: load → chunk."""
        docs = self.load_corpus(dataset_name)
        return self.chunk_documents(docs)
