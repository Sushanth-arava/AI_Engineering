"""RAG pipeline - Phases 1 and 2."""

import yaml
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

_PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "../../src/config/prompts.yaml")


def _load_prompt_template(prompts_path: str = _PROMPTS_PATH) -> str:
    """Load RAG prompt template from versioned YAML (Phase 2 prompt versioning)."""
    try:
        with open(prompts_path) as f:
            data = yaml.safe_load(f)
        return data["rag_prompt"]["template"]
    except Exception:
        return (
            "You are a helpful assistant. Answer the question using ONLY the provided context.\n"
            "If the context does not contain enough information, respond with:\n"
            "'I cannot answer this question based on the available information.'\n\n"
            "Question: {question}\n\nContext:\n{context}\n\nAnswer (cite the source passage id):"
        )


def _load_faithfulness_prompt_template(prompts_path: str = _PROMPTS_PATH) -> str:
    """Load faithfulness check prompt template from versioned YAML."""
    try:
        with open(prompts_path) as f:
            data = yaml.safe_load(f)
        return data["faithfulness_check_prompt"]["template"]
    except Exception:
        return (
            "Given the following question, answer, and source context, determine if the answer\n"
            "is supported by the context.\n\n"
            "Question: {question}\nAnswer: {answer}\nContext: {context}\n\n"
            "Is the answer faithfully supported by the context? Reply with only \"YES\" or \"NO\"."
        )


class RAGPipeline:
    """
    End-to-end RAG pipeline.
    Phase 1: basic vector retrieval + LLM answer + citation display.
    Phase 2: uses HybridSearch + CrossEncoderReranker if provided.
    """

    def __init__(self, retriever, llm_model: str = "gpt-3.5-turbo",
                 temperature: float = 0.0, max_tokens: int = 512,
                 reranker=None):
        """
        Args:
            retriever: VectorStore (basic) or HybridSearch (advanced).
            reranker: CrossEncoderReranker or None.
        """
        self.retriever = retriever
        self.reranker = reranker
        self.llm = ChatOpenAI(model=llm_model, temperature=temperature, max_tokens=max_tokens)
        prompt_template = _load_prompt_template()
        self.prompt = PromptTemplate.from_template(prompt_template)
        self.chain = self.prompt | self.llm | StrOutputParser()
        self._faithfulness_prompt_template = _load_faithfulness_prompt_template()

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant chunks, then optionally rerank (Phase 2)."""
        chunks = self.retriever.search(query, top_k=top_k)
        if self.reranker is not None:
            chunks = self.reranker.rerank(query, chunks, top_k=min(top_k, 3))
        return chunks

    def _format_context(self, chunks: List[Dict]) -> str:
        """Format chunks into context string with citation markers."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            source_id = chunk["metadata"].get("original_id", chunk["metadata"].get("source_id", f"chunk_{i}"))
            parts.append(f"[Source {i} | id={source_id}]\n{chunk['text']}")
        return "\n\n---\n\n".join(parts)

    def _faithfulness_check(self, question: str, answer: str, context: str) -> bool:
        """
        Phase 2: faithfulness check using versioned prompt from prompts.yaml.
        Uses LLM to verify the answer is grounded in the context.
        Returns True if answer is faithful, False if it should be refused.
        """
        check_prompt = self._faithfulness_prompt_template.format(
            question=question,
            answer=answer,
            context=context[:1000],
        )
        try:
            result = self.llm.invoke(check_prompt).content.strip().upper()
            return result.startswith("YES")
        except Exception:
            logger.exception("Faithfulness check failed — failing closed (refusing answer).")
            return False

    def query(self, question: str, top_k: int = 5, check_faithfulness: bool = False) -> Dict:
        """
        Full pipeline query.
        Returns: {answer, citations, chunks_used, refused}
        Citations = list of {source_id, text_snippet} for each supporting chunk.
        """
        chunks = self.retrieve(question, top_k=top_k)
        if not chunks:
            return {
                "answer": "I cannot answer this question based on the available information.",
                "citations": [],
                "chunks_used": [],
                "refused": True,
            }

        context = self._format_context(chunks)
        answer = self.chain.invoke({"question": question, "context": context})

        # Phase 2: faithfulness refusal
        refused = False
        if check_faithfulness:
            if not self._faithfulness_check(question, answer, context):
                answer = "I cannot answer this question based on the available information."
                refused = True

        # Citation display (Phase 1)
        citations = [
            {
                "source_id": c["metadata"].get("original_id", c["metadata"].get("source_id", "?")),
                "text_snippet": c["text"][:200] + ("..." if len(c["text"]) > 200 else ""),
                "score": round(c.get("rerank_score", c.get("score", 0.0)), 4),
            }
            for c in chunks
        ]

        return {
            "answer": answer,
            "citations": citations,
            "chunks_used": chunks,
            "refused": refused,
        }
