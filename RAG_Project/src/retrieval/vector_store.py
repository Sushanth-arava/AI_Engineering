"""ChromaDB vector store for semantic retrieval."""

import chromadb
from langchain_openai import OpenAIEmbeddings
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class VectorStore:
   

    def __init__(self, persist_dir: str = "./chroma_db", collection_name: str = "wikipedia_rag",
                 embedding_model: str = "text-embedding-3-small"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embedder = OpenAIEmbeddings(model=embedding_model)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: List[Dict], batch_size: int = 100) -> None:
        
        logger.info(f"Upserting {len(chunks)} chunks into ChromaDB...")
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            ids = [c["id"] for c in batch]
            texts = [c["text"] for c in batch]
            metadatas = [c["metadata"] for c in batch]
            embeddings = self.embedder.embed_documents(texts)
            self.collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        logger.info("Upsert complete.")

    def reset(self) -> None:
        
        logger.info(f"Resetting collection '{self.collection_name}'...")
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection reset.")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        
        query_embedding = self.embedder.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for doc_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "id": doc_id,
                "text": doc,
                "metadata": meta,
                "score": 1 - dist, 
            })
        return hits

    def count(self) -> int:
        return self.collection.count()

    def is_empty(self) -> bool:
        return self.count() == 0
