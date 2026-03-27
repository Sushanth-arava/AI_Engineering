# Retrieval-Augmented Generation (RAG)

- RAG is the process of optimizing the output of a large language model, so it references an authoritative knowledge base outside of its training data sources before generating a response.

- Large Language Models (LLMs) are trained on vast volumes of data and use billions of parameters to generate original output for tasks like answering questions, translating languages, and completing sentences.

- RAG extends the already powerful capabilities of LLMs to specific domains or an organization's internal knowledge base, all without the need to retrain the model. It is a cost-effective approach to improving LLM output so it remains relevant, accurate, and useful in various contexts.

---

## System Architecture

### 1. Knowledge-Base Construction (Ingestion Pipeline)

```
  Source Documents          Chunking             Embedding Model        Vector DB
  (~10M tokens)            (~5K tokens)
                                                  ┌─────────────┐
  ┌──────────────┐       ┌────────────┐           │  [1, 0, .1, │
  │    Source    │       │  ┌───────┐ │           │   0, 0]     │       ┌──────────┐
  │  Documents   │──────▶│  │ Chunk │ │──────────▶│  [1, 0, .1, │──────▶│ Vector   │
  │ (~10M tokens)│       │  ├───────┤ │   2000    │   0, 0]     │ 2000  │    DB    │
  └──────────────┘       │  │ Chunk │ │  chunks   │  [1, 0, .1, │ vecs  └──────────┘
                         │  ├───────┤ │           │   0, 0]     │
                         │  │ Chunk │ │           └─────────────┘
                         │  └───────┘ │                 ▲
                         └────────────┘           Embedding model
                          ~5K tokens/chunk        converts each chunk
                                                  into a dense vector
```

**Flow:**
1. **Source Documents** — Raw PDFs or text files (~10M tokens total) are loaded.
2. **Chunking** — Documents are split into smaller overlapping pieces (~5K tokens each), producing ~2000 chunks. This ensures each piece fits within the model's context window.
3. **Embedding Model** — Each chunk is passed through an embedding model (e.g. `text-embedding-3-small`) which converts the text into a dense numeric vector (e.g. `[1, 0, .1, 0, 0]`). Semantically similar text produces similar vectors.
4. **Vector DB** — All 2000 vectors (and their source chunks) are persisted in a vector database (ChromaDB), enabling fast similarity search at query time.

---

### 2. Retrieval Pipeline

```
                                              ┌───────────┐
                                         ┌──▶│   Chunk   │
  ┌───────┐   Embedding    ┌──────────┐  │   ├───────────┤    ┌─────────────────────┐
  │ Query │──────────────▶ │Retriever │──┼──▶│   Chunk   │───▶│  Chunk 1 Text       │
  └───────┘  [1, 0, .1,    └──────────┘  │   ├───────────┤    │  Chunk 2 Text       │
              0, 0]               ▲       └──▶│   Chunk   │    │  Chunk 3 Text       │
                                  │           └───────────┘    ├─────────────────────┤
                             cosine                            │  Question (prompt)  │
                            similarity                         └─────────────────────┘
                           search in                                     │
                           Vector DB                                     ▼
                                                                   LLM Response
```

**Flow:**
1. **Query** — The user submits a natural language question.
2. **Embedding** — The query is converted into a vector using the same embedding model used during ingestion.
3. **Retriever** — The query vector is compared against all stored vectors in the Vector DB using cosine similarity. The top-k most similar chunks are retrieved.
4. **Context + Prompt** — The retrieved chunks are assembled alongside the original question and passed as context to the LLM.
5. **LLM Response** — The LLM answers the question grounded in the retrieved context, reducing hallucination.

---

## Parts of a RAG

1. Knowledge based construction (ingestion pipeline)
2. Retrieval pipeline

### Ingestion Pipeline:

1. Chunking
2. Pass the chunks into an embedding model
3. Store them into vector databases

### Retrieval Pipeline:

1. Take query and convert it into vector embeddings
2. Search vector database for similar chunks
3. Return top-k chunks as context for the LLM
