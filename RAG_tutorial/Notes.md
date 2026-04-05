# RAG Pipeline

This document covers all files in the RAG system.

---

## File: `config.py`

Centralised configuration using nested Pydantic models. A single `config` instance is imported by all other pipeline files.

### Classes

| Class | Description |
|---|---|
| `ChunkingConfig` | Controls text splitting strategy and parameters |
| `RetrievalConfig` | Controls how many docs are retrieved |
| `ModelConfig` | Embedding and LLM model names |
| `StorageConfig` | ChromaDB persistence path |
| `RAGConfig` | Top-level config composing all sub-configs |

### Defaults

| Config path | Default value |
|---|---|
| `config.chunking.strategy` | `"recursive"` |
| `config.chunking.chunk_size` | `1000` |
| `config.chunking.chunk_overlap` | `100` |
| `config.retrieval.k` | `3` |
| `config.models.embedding_model` | `"text-embedding-3-small"` |
| `config.models.llm_model` | `"gpt-4o"` |
| `config.storage.persist_directory` | `"db/chromaDB"` |

---

## File: `ingestion_pipeline.py`

Loads PDFs, splits them into chunks, embeds them, and stores them in ChromaDB.

### `load_documents(docs_path="Books")`

Loads all PDF files from the given directory using `DirectoryLoader` + `PyPDFLoader`.

- **Returns:** List of LangChain `Document` objects (one per PDF page).
- **Raises:** `FileNotFoundError` if the directory doesn't exist or has no PDFs.
- **Side effects:** Prints source, content length, preview, and metadata for the first two documents.

---

### `chunck_documents(documents, chunk_size, chunk_overlap)`

Splits documents into chunks using `CharacterTextSplitter`.

- **Defaults:** `chunk_size` and `chunk_overlap` from `config.chunking`.
- **Returns:** List of `Document` chunks.
- **Side effects:** Prints source, length, and content of the first two chunks; prints remaining count if more than five.

---

### `create_vector_store(chunks, persist_directory)`

Embeds chunks using OpenAI embeddings and stores them in a persistent ChromaDB collection.

- **Defaults:** `persist_directory` from `config.storage.persist_directory`.
- **Embedding model:** `config.models.embedding_model` (`text-embedding-3-small`).
- **Similarity metric:** Cosine (`hsnw:space: cosine`).
- **Returns:** A `Chroma` vector store instance.

---

### `main()`

Orchestrates the full ingestion pipeline: load → chunk → embed → store.

---

## File: `retrieval_pipeline.py`

Script that loads an existing ChromaDB vector store, retrieves relevant chunks for a hardcoded query, and generates an answer with `gpt-4o`.

### Module-level setup

| Variable | Description |
|---|---|
| `embedding_model` | `OpenAIEmbeddings` using `config.models.embedding_model` |
| `db` | `Chroma` connected to `config.storage.persist_directory` with cosine similarity |
| `query` | Hardcoded query string (`"What is celebrity problem?"`) |
| `retriever` | `db.as_retriever` returning top `config.retrieval.k` results |
| `model` | `ChatOpenAI` using `config.models.llm_model` |

### Flow

1. Retrieves top-k relevant docs via `retriever.invoke(query)`.
2. Builds a prompt combining the query and retrieved doc content.
3. Calls `model.invoke(messages)` with a system + human message.
4. Prints retrieved context (source, page, content) and the generated answer.

### Alternative retriever (commented out)

```python
# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={"k": 5, "score_threshold": 0.3},
# )
```

Returns chunks only if their cosine similarity score meets or exceeds `0.3`.

---

## File: `history_aware_generation.py`

Conversational RAG with multi-turn history. Rewrites follow-up questions into standalone queries before retrieval, and returns structured output.

### Pydantic models

| Model | Fields | Purpose |
|---|---|---|
| `RewrittenQuestion` | `rewritten_question: str` | Structured output for query rewriting |
| `RAGAnswer` | `answer: str`, `sources: list[str]` | Structured LLM answer with source filenames |
| `ChatTurn` | `user: str`, `assistant: str` | Single conversation turn |
| `ConversationState` | `turns: list[ChatTurn]` | Full conversation history; converts to LangChain messages via `to_langchain_messages()` |

### `ask_question(user_question)`

1. Converts conversation history to LangChain messages.
2. If history exists, uses `model.with_structured_output(RewrittenQuestion)` to rewrite the question into a standalone, searchable form.
3. Retrieves top `config.retrieval.k` docs using the (rewritten) question.
4. Builds a prompt with the original question + retrieved docs + history.
5. Uses `model.with_structured_output(RAGAnswer)` to get a typed answer with sources.
6. Appends the turn to `ConversationState`.
7. Prints and returns the `RAGAnswer`.

### `start_chat()`

REPL loop — prompts the user for input, calls `ask_question`, exits on `"quit"`.

---

## File: `recursive_text_splitter.py`

Demonstration script comparing `CharacterTextSplitter` vs `RecursiveCharacterTextSplitter` on a sample text.

### `CharacterTextSplitter`

- Splits on a single separator (` ` in this demo).
- Struggles with long paragraphs that have no natural separator boundaries.

### `RecursiveCharacterTextSplitter`

- Tries separators in order: `["\n\n", "\n", ". ", " ", ""]`.
- Falls back to the next separator if a chunk still exceeds `chunk_size`.
- Handles long paragraphs more gracefully.

### Demo text

Fake Tesla Q3 report with four sections, one of which is a very long paragraph with no newlines — designed to show where `CharacterTextSplitter` fails and `RecursiveCharacterTextSplitter` succeeds.

---

## File: `multi_query.py`

Multi-query RAG addresses a core weakness of single-query retrieval: a single phrasing of a question may miss relevant documents that would be retrieved under a different wording. The solution is to generate several semantically varied rewrites of the original query, retrieve documents for each, deduplicate, and answer from the merged result set.

### Why Multi-Query?

Vector similarity search is sensitive to phrasing. The same concept expressed differently can produce different embedding vectors and therefore hit different chunks in the store. Generating multiple query variations increases recall without changing the index.

### Module-level setup

| Variable | Description |
|---|---|
| `embedding_model` | `OpenAIEmbeddings` using `config.models.embedding_model` |
| `db` | `Chroma` connected to `config.storage.persist_directory` with cosine similarity |

### Pydantic models

| Model | Fields | Purpose |
|---|---|---|
| `QueryVariations` | `queries: List[str]` | Structured LLM output — a list of rewritten query strings |

### Multi-Query Flow

1. **Rewrite** — pass the original user question to an LLM with `model.with_structured_output(QueryVariations)`. The prompt instructs the model to produce N semantically distinct but equivalent phrasings.
2. **Retrieve per variation** — call `retriever.invoke(q)` for each query in `QueryVariations.queries`, collecting a list of `Document` lists.
3. **Deduplicate** — merge all result lists and remove duplicate chunks (typically by comparing `page_content` or document `id`).
4. **Generate** — build a prompt from the original question + deduplicated context and call the LLM for a final answer.

### Query Variation Strategies

| Strategy | Description |
|---|---|
| Paraphrase | Rephrase the question using different vocabulary |
| Perspective shift | Ask the same question from different angles (e.g., cause vs. effect) |
| Abstraction level | Generate both a general and a specific version of the question |
| Keyword expansion | Swap domain synonyms (e.g., "cost" → "price", "expenditure") |

### Tradeoffs

| Factor | Impact |
|---|---|
| More variations | Higher recall, more LLM calls, higher latency and cost |
| Fewer variations | Faster, cheaper, but may miss relevant chunks |
| Deduplication quality | Poor dedup inflates context size and degrades generation quality |

### Relationship to `history_aware_generation.py`

Both files rewrite the user question before retrieval. The difference is intent:

- `history_aware_generation.py` rewrites to resolve **conversational references** (e.g., "it", "that book") into a standalone query — one rewrite per turn.
- `multi_query.py` rewrites to **maximise retrieval coverage** — multiple variations from a single question, no conversation history required.

---

## File: `retrieval_methods.py`

Demonstrates and compares the three retrieval strategies supported by the ChromaDB `as_retriever` interface. Acts as a reference script — Methods 2 and 3 are commented out and can be swapped in place of Method 1.

### Method 1 — Top-k similarity (default)

```python
retriever = db.as_retriever(search_kwargs={"k": config.retrieval.k})
```

Returns the `k` most similar documents by cosine distance, unconditionally.

---

### Method 2 — Similarity with score threshold

```python
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0},
)
```

Returns up to `k` documents **only if** their similarity score meets or exceeds `score_threshold`. Documents below the threshold are dropped, so the result set may be smaller than `k` or empty.

| Parameter | Effect |
|---|---|
| `score_threshold` | Minimum cosine similarity a document must have to be included |
| `k` | Upper bound on returned documents |

Use when low-quality retrievals are worse than no retrieval (e.g., open-domain QA where a wrong answer is more harmful than "I don't know").

---

### Method 3 — Maximum Marginal Relevance (MMR)

```python
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,          # Final number of docs returned
        "fetch_k": 10,   # Initial candidate pool size
        "lambda_mult": 0.5,  # 0 = max diversity, 1 = max relevance
    },
)
```

MMR balances **relevance** (similarity to the query) and **diversity** (dissimilarity among selected documents). It iteratively selects documents that are both relevant to the query and not redundant with already-selected documents.

**Algorithm sketch:**

1. Fetch `fetch_k` candidates by similarity.
2. From the candidates, score each unselected document as:
   `MMR = λ · sim(doc, query) − (1 − λ) · max_sim(doc, already_selected)`
3. Add the highest-scoring document to the result set.
4. Repeat until `k` documents are selected.

| Parameter | Effect |
|---|---|
| `fetch_k` | Larger pool gives MMR more candidates to diversify from; should be `> k` |
| `lambda_mult` | Closer to `1.0` = prefer relevance; closer to `0.0` = prefer diversity |

Use when the top-k documents tend to be near-duplicate chunks covering the same passage, which wastes context window space and degrades generation quality.

---

## File: `semantic_text_splitter.py`

Demonstrates `SemanticChunker` from `langchain_experimental`, which splits text based on **embedding similarity between sentences** rather than character count or separator patterns.

### How `SemanticChunker` works

1. Splits the text into individual sentences.
2. Embeds each sentence using the provided embedding model.
3. Computes cosine similarity between consecutive sentence embeddings.
4. Identifies **breakpoints** — positions where similarity drops below a threshold — and cuts the text at those points.
5. Returns chunks where each chunk contains sentences that are semantically coherent with one another.

### Key parameters

| Parameter | Value in script | Description |
|---|---|---|
| `embeddings` | `OpenAIEmbeddings()` | Model used to embed sentences for similarity comparison |
| `breakpoint_threshold_type` | `"percentile"` | How the threshold is computed. `"percentile"` cuts at drops below the Nth percentile of all similarity scores; `"standard_deviation"` cuts at drops more than N std devs below the mean |
| `breakpoint_threshold_amount` | `70` | The percentile value — breakpoints are similarity drops below the 70th percentile |

### Comparison with character-based splitters

| Splitter | Split criterion | Preserves semantic coherence? | Cost |
|---|---|---|---|
| `CharacterTextSplitter` | Fixed character count + separator | No — can split mid-sentence or mid-topic | None |
| `RecursiveCharacterTextSplitter` | Hierarchical separators, falls back to smaller units | Partial — respects paragraph/sentence boundaries | None |
| `SemanticChunker` | Embedding similarity drop between sentences | Yes — chunks align with topic boundaries | Embedding API call per sentence |

### When to use

Use `SemanticChunker` when chunk coherence matters more than chunk-size predictability — for example, documents with abrupt topic changes (e.g., multi-section reports, legal contracts). The trade-off is higher ingestion cost (one embedding call per sentence) and non-uniform chunk sizes.

---

## Updated: `config.py` — additional `ChunkingConfig` fields

The `ChunkingConfig` model supports two additional optional fields not covered in the original documentation:

| Field | Type | Default | Description |
|---|---|---|---|
| `separators` | `Optional[list[str]]` | `None` | Custom separator list passed to `RecursiveCharacterTextSplitter`. If `None`, the splitter uses its built-in defaults (`["\n\n", "\n", ". ", " ", ""]`) |
| `semantic_threshold` | `Optional[float]` | `None` | Breakpoint threshold amount forwarded to `SemanticChunker` when `strategy = "semantic"` |

The `strategy` field also now documents four valid values:

| Value | Splitter used |
|---|---|
| `"recursive"` | `RecursiveCharacterTextSplitter` |
| `"token"` | Token-based splitter |
| `"markdown"` | Markdown-aware splitter |
| `"semantic"` | `SemanticChunker` (embedding-based) |

---

## Concept: Reciprocal Rank Fusion (RRF)

### Definition

Reciprocal Rank Fusion is a **rank aggregation algorithm** that merges multiple ranked lists of documents into a single unified ranking. It was introduced by Cormack et al. (2009) and is model-agnostic — it only requires the rank position of each document in each list, not raw scores or probabilities.

### Formula

For a document `d`, its RRF score across `n` ranked lists is:

```
RRF(d) = Σ  1 / (k + rank_i(d))
         i=1..n
```

Where:
- `rank_i(d)` is the 1-based position of document `d` in ranked list `i` (if `d` is absent from list `i`, it contributes 0 to the sum).
- `k` is a smoothing constant (commonly `k = 60`).
- The sum runs over all `n` ranked lists being fused.

### The role of `k`

`k` prevents very high scores from documents that appear at rank 1 in a single list. With `k = 60`:
- Rank 1 contributes `1/61 ≈ 0.0164`
- Rank 10 contributes `1/70 ≈ 0.0143`
- Rank 100 contributes `1/160 ≈ 0.0063`

A higher `k` flattens the score differences between ranks (more uniform weighting). A lower `k` amplifies the advantage of top-ranked documents.

### Step-by-step example

Given original query `Q`, an LLM generates 3 query variations. Each variation retrieves 5 documents from ChromaDB:

```
List 1 (variation 1): [Doc_A, Doc_B, Doc_C, Doc_D, Doc_E]
List 2 (variation 2): [Doc_C, Doc_A, Doc_F, Doc_B, Doc_G]
List 3 (variation 3): [Doc_B, Doc_C, Doc_A, Doc_H, Doc_D]
```

RRF scores with `k = 60`:

| Document | List 1 rank | List 2 rank | List 3 rank | RRF score |
|---|---|---|---|---|
| Doc_A | 1 → 1/61 | 2 → 1/62 | 3 → 1/63 | 0.0164 + 0.0161 + 0.0159 = **0.0484** |
| Doc_B | 2 → 1/62 | 4 → 1/64 | 1 → 1/61 | 0.0161 + 0.0156 + 0.0164 = **0.0481** |
| Doc_C | 3 → 1/63 | 1 → 1/61 | 2 → 1/62 | 0.0159 + 0.0164 + 0.0161 = **0.0484** |
| Doc_D | 4 → 1/64 | absent → 0 | 5 → 1/65 | 0.0156 + 0 + 0.0154 = **0.0310** |
| Doc_F | absent → 0 | 3 → 1/63 | absent → 0 | 0 + 0.0159 + 0 = **0.0159** |

Final re-ranked order: `[Doc_A ≈ Doc_C, Doc_B, Doc_D, Doc_F, ...]`

Documents appearing consistently across multiple lists (Doc_A, Doc_B, Doc_C) score higher than documents that rank well in only one list (Doc_F).

### When to use RRF

| Scenario | Why RRF helps |
|---|---|
| Multi-query retrieval | Fuses the ranked lists from each query variation into one coherent ranked list |
| Hybrid search (dense + sparse) | Combines BM25 keyword rankings with embedding similarity rankings without needing score normalisation |
| Ensemble retrievers | Merges results from multiple embedding models or retrieval strategies |
| Score incompatibility | Raw similarity scores from different retrieval methods are not directly comparable; ranks are |

### Relationship to multi-query RAG

In `multi_query.py`, multiple query variations each produce a ranked list of retrieved documents. Without RRF, the current implementation naively concatenates these lists and deduplicates by content. RRF is the principled upgrade to that deduplication step:

1. **Generate** N query variations (as `multi_query.py` already does).
2. **Retrieve** a ranked list of documents for each variation.
3. **Fuse** all ranked lists using RRF to produce a single ranked list.
4. **Truncate** to the top-k documents from the fused list.
5. **Generate** the final answer using those top-k documents as context.

The key advantage over simple deduplication: RRF **preserves rank information** — documents that appear near the top of multiple lists are surfaced above documents that appear only once, even if that single appearance was at rank 1.

### The Problem with k=0

When `k = 0`, the RRF formula collapses to `1 / rank_i(d)`, which makes score differences between consecutive ranks extremely steep:

| Rank | Score at k=0 | Score at k=60 | Penalty vs rank 1 (k=0) |
|---|---|---|---|
| 1 | 1.0000 | 0.01639 | — |
| 2 | 0.5000 | 0.01613 | 50% |
| 3 | 0.3333 | 0.01587 | 67% |
| 5 | 0.2000 | 0.01538 | 80% |
| 10 | 0.1000 | 0.01429 | 90% |

**Real-world issue:** Consider two chunks, Chunk X and Chunk Y, retrieved by a similarity search where their scores are nearly identical — say `0.95` vs `0.94` in Retrieval A. That `0.01` difference in cosine similarity is enough to place Chunk X at rank 1 and Chunk Y at rank 2. With `k = 0`, this translates directly into a `1.0` vs `0.5` RRF contribution — a **2× scoring gap** caused by a `1%` similarity difference.

This is problematic because small similarity score differences at the top of a ranked list are frequently noise, not signal. Embedding distances are approximate; a `0.01` gap does not reliably mean one document is meaningfully more relevant than the other.

The smoothing constant `k` (typically `60`) exists precisely to dampen this effect. By adding `k` to every denominator, the score curve flattens near the top of the ranking:

- With `k = 60`, rank 1 scores `1/61 ≈ 0.0164` and rank 2 scores `1/62 ≈ 0.0161` — a difference of `~1.5%` rather than `50%`.
- A document must be consistently absent from, or ranked very low across, multiple lists before RRF significantly discounts it.

In short, `k = 0` rewards a single strong retrieval hit far too aggressively. The default `k = 60` ensures that **cross-list consistency** drives the final ranking, not random noise in individual similarity scores.

---

## File: `hybrid_search.py`

Demonstrates a hybrid retrieval system that combines vector semantic search and keyword-based (BM25) search using an `EnsembleRetriever`. This script shows how different retrieval methods can be weighted and merged to leverage the strengths of both dense and sparse retrieval.

### Sample Data

A list of 18 document chunks covering tech companies (Microsoft, Tesla, Google, SpaceX, NVIDIA, Apple) and programming languages (Python, Java). The chunks are intentionally repetitive and polysemous to demonstrate retrieval challenges:

- **Repetition:** "Tesla" appears multiple times across different documents with varying context.
- **Polysemy:** Terms like "Python", "Java", "Orange", and "Apple" refer to multiple distinct concepts (language vs. animal, programming vs. beverage, fruit vs. company).

Each chunk is converted to a LangChain `Document` with metadata: `{"source": f"chunk {i}"}`.

### Retrievers

#### **Vector Retriever (Semantic)**

```python
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_configuration={"hnsw:space": "cosine"},
)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
```

- **Model:** `text-embedding-3-small` from OpenAI.
- **Similarity metric:** Cosine distance (via HNSW).
- **Returns:** Top 2 documents by semantic similarity.
- **Strength:** Captures meaning and conceptual relevance (e.g., "space exploration company" matches documents about SpaceX).
- **Weakness:** Sensitive to synonym mismatch; may miss exact keyword matches if the embedding space clusters differently.

#### **BM25 Retriever (Keyword-based)**

```python
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 3
```

- **Algorithm:** BM25 (Okapi BM25), a probabilistic ranking function used in information retrieval.
- **Returns:** Top 3 documents by keyword frequency and inverse document frequency.
- **Strength:** Excels at exact and partial keyword matching (e.g., "Tesla Cybertruck" → all documents mentioning both terms).
- **Weakness:** Ignores semantic meaning; "purchase cost" and "acquisition price" are treated as completely different queries.

### Hybrid Retriever (Ensemble)

```python
hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5]
)
```

- **Strategy:** Combines results from both retrievers using equal weights (50% semantic, 50% keyword).
- **Rank aggregation:** LangChain's `EnsembleRetriever` uses Reciprocal Rank Fusion (RRF) internally to merge and re-rank results.
- **Benefit:** Documents that rank well in both semantic and keyword searches are surfaced first, leveraging complementary retrieval signals.

### Example Query

```python
test_query = "electric vehicle manufacturing"
retrieved_chunks = hybrid_retriever.invoke(test_query)
```

**Expected behavior:**
- **Vector retriever** finds documents semantically related to electric vehicles and manufacturing.
- **BM25 retriever** finds documents explicitly containing "electric", "vehicle", or "manufacturing".
- **Hybrid result** combines both, prioritizing documents that satisfy both signals.

### When to use hybrid search

| Scenario | Why hybrid helps |
|---|---|
| Mixed terminology | Documents use both technical jargon and plain language; vector search finds semantic matches, BM25 finds exact keywords |
| Named entities | Proper nouns (company names, people) are better matched by BM25; vector embeddings may conflate similar names |
| Short queries | Keyword search is robust for short phrases; vector search may be noisy without enough context |
| Polysemous terms | "Python" (language vs. snake): BM25 returns all "Python" mentions; vector search disambiguates based on context |
| Balanced precision/recall | Neither method dominates; hybrid avoids missing relevant documents that one method alone would fail to retrieve |

### Tuning the hybrid system

| Parameter | Adjustment | Effect |
|---|---|---|
| `vector_retriever.search_kwargs["k"]` | Increase from 2 | Vector retriever returns more candidates; increases recall but may add noise |
| `bm25_retriever.k` | Increase from 3 | BM25 returns more candidates; helps with keyword diversity but increases candidate pool |
| `weights` | Change from `[0.5, 0.5]` | Shift towards vector (`[0.7, 0.3]`) or keyword (`[0.3, 0.7]`) depending on query type |
| Embedding model | Switch to `text-embedding-3-large` | Richer embeddings → better semantic matching, but higher cost and latency |
| Similarity metric | Change from `cosine` | Try `"l2"` (Euclidean) or `"ip"` (inner product) to test distance measures |

### Relationship to other files

- **`retrieval_pipeline.py`:** Uses a single retriever (vector-only). This file extends that with a second retriever and ensemble.
- **`multi_query.py`:** Generates multiple query variations; hybrid search could be applied per variation for even higher recall.
- **`retrieval_methods.py`:** Showcases alternatives like MMR and similarity thresholds. Those can be used with the vector retriever to further tune retrieval quality.
- **`reciprocal_rank_fusion.py`:** Explains the rank aggregation algorithm underlying `EnsembleRetriever`.
