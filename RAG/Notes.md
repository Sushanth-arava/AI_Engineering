# RAG Pipeline

This document covers the ingestion and retrieval pipelines for a Retrieval-Augmented Generation (RAG) system.

## File: `ingestion_pipeline.py`

### `load_documents(docs_path="Books")`

Loads all PDF files from the specified directory using LangChain's `DirectoryLoader` and `PyPDFLoader`.

- **Parameters:**
  - `docs_path` _(str)_: Path to the directory containing PDF files. Defaults to `"Books"`.
- **Returns:** A list of LangChain `Document` objects, one per PDF page.
- **Raises:** `FileNotFoundError` if the directory does not exist or contains no PDF files.
- **Side effects:** Prints a preview (source, content length, content snippet, metadata) for the first two loaded documents.

---

### `chunck_documents(documents, chunk_size=1000, chunk_overlap=0)`

Splits the loaded documents into smaller text chunks using LangChain's `CharacterTextSplitter`.

- **Parameters:**
  - `documents` _(list)_: List of `Document` objects returned by `load_documents`.
  - `chunk_size` _(int)_: Maximum number of characters per chunk. Defaults to `1000`.
  - `chunk_overlap` _(int)_: Number of characters to overlap between consecutive chunks. Defaults to `0`.
- **Returns:** A list of `Document` chunks ready for embedding.
- **Side effects:** Prints the source, length, and full content of the first two chunks, and a count of remaining chunks if more than five exist.

---

### `create_vector_store(chunks, persistant_directory="db/chromaDB")`

Embeds the document chunks using OpenAI's embedding model and stores them in a persistent ChromaDB vector store.

- **Parameters:**
  - `chunks` *(list)*: List of `Document` chunks returned by `chunck_documents`.
  - `persistant_directory` *(str)*: File path where ChromaDB will persist the vector store. Defaults to `"db/chromaDB"`.
- **Returns:** A `Chroma` vector store instance containing the embedded chunks.
- **Embedding model:** Uses `text-embedding-3-small` via `OpenAIEmbeddings` (requires `OPENAI_API_KEY` in the environment).
- **Similarity metric:** Configures the ChromaDB collection to use cosine similarity (`hsnw:space: cosine`).
- **Side effects:** Prints progress messages and the final storage path.

---

### `main()`

Entry point that orchestrates the full ingestion pipeline: loads PDFs from the `"Books"` directory, splits them into chunks, embeds the chunks, and stores them in ChromaDB.

---

## File: `retrieval_pipeline.py`

This script loads an existing ChromaDB vector store and retrieves the most relevant document chunks for a given query using semantic similarity search.

### Module-level setup

| Variable | Description |
|---|---|
| `persist_directory` | Path to the persisted ChromaDB store (`"db/chromaDB"`). |
| `embedding_model` | `OpenAIEmbeddings` using `text-embedding-3-small` to encode queries. |
| `db` | `Chroma` instance connected to the persisted store with cosine similarity. |
| `query` | The user query string to search against the vector store. |

### Retriever

```python
retriever = db.as_retriever(search_kwargs={"k": 3})
```

Wraps the `Chroma` store as a LangChain retriever that returns the top `k=3` most similar chunks for a query. An alternative threshold-based retriever is included as a comment:

```python
# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={"k": 5, "score_threshold": 0.3},
# )
```

This variant only returns chunks whose cosine similarity score meets or exceeds `0.3`.

### Query & output

Calls `retriever.invoke(query)` to fetch relevant documents, then prints each result with its source file, page number, and content.
