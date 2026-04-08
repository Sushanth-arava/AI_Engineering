# RAG Wikipedia Q&A

A Retrieval-Augmented Generation (RAG) pipeline that answers questions grounded in Wikipedia. Built in three progressive phases — from basic vector search to hybrid retrieval with reranking, faithfulness enforcement, and a CI quality gate.

---

## Table of Contents

- [Architecture](#architecture)
- [Phase 1 — Basic RAG](#phase-1--basic-rag)
- [Phase 2 — Advanced RAG](#phase-2--advanced-rag)
- [Phase 3 — Evaluation & CI Gate](#phase-3--evaluation--ci-gate)
- [Algorithms](#algorithms)
- [Project Structure](#project-structure)
- [Setup & Running](#setup--running)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Prompt Versioning](#prompt-versioning)

---

## Architecture

```
POST /chat
     │
     ▼
[FastAPI routes]
     │
     ▼
[RAGPipeline]
     │
     ├── retrieve(query, top_k)
     │     ├── basic mode    → VectorStore (ChromaDB cosine similarity)
     │     └── advanced mode → HybridSearch (BM25 + vector fusion)
     │                              └── CrossEncoderReranker
     │
     ├── LangChain chain
     │     rag_prompt (prompts.yaml) → ChatOpenAI → StrOutputParser
     │
     └── _faithfulness_check()        [optional, fails closed]
           faithfulness_check_prompt → ChatOpenAI → YES / NO
```

The API defaults to **advanced mode** (`pipeline_mode = "advanced"`). Switch to `"basic"` for faster startup without BM25/reranking.

---

## Phase 1 — Basic RAG

**Goal:** Ingest Wikipedia, embed chunks, answer questions with citations.

### Ingestion

1. Load `rag-datasets/rag-mini-wikipedia` from HuggingFace
2. Split into chunks (default: 600 chars, 100 overlap) via `DataProcessor`
3. Embed each chunk with OpenAI `text-embedding-3-small`
4. Persist embeddings to **ChromaDB** on local disk

### Query

1. Embed the incoming question using the same model
2. Cosine similarity search → top-k chunks from ChromaDB
3. Format chunks as numbered context with `[Source N | id=X]` citation markers
4. LangChain chain: `rag_prompt | ChatOpenAI(gpt-3.5-turbo) | StrOutputParser`
5. Return answer + citations

---

## Phase 2 — Advanced RAG

Adds three improvements on top of Phase 1: hybrid search, cross-encoder reranking, and a faithfulness gate.

### Hybrid Search (BM25 + Vector Fusion)

| Signal | Strength | Weakness |
|---|---|---|
| **BM25** (keyword) | Exact term matching, named entities | Misses synonyms, paraphrases |
| **Vector** (semantic) | Understands meaning, handles paraphrases | Can miss exact keyword matches |

Both scores are normalized to `[0, 1]` before fusion:

```
fused_score = 0.6 × vector_score + 0.4 × bm25_score
```

Weights are configurable via `bm25_weight` and `vector_weight` in config.

### Cross-Encoder Reranking

After hybrid search returns candidates, a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-scores each `(query, chunk)` pair jointly.

Bi-encoders encode query and document independently — the cross-encoder sees both together, producing significantly more accurate relevance scores. The trade-off is speed (scores can't be pre-computed), so it runs only on the top-k candidates (default: 5 in, top 3 out).

### Faithfulness Gate

Every response optionally passes through a second LLM call before being returned:

```
"Is the answer supported by the context? Reply YES or NO."
```

- **YES** → answer is returned as-is
- **NO** → answer is replaced with a refusal and `refused: true` is set
- **Exception** → also refuses (`refused: true`). The gate **fails closed** — a broken check is never silently treated as a pass.

This prevents hallucination: the system declines rather than inventing a plausible-sounding answer.

---

## Phase 3 — Evaluation & CI Gate

### Golden Dataset

`eval/golden_dataset.json` contains 60 manually verified question–answer pairs spanning science, history, geography, literature, and technology. These serve as the fixed measuring stick against which every code change is evaluated.

### RAGAS Metrics

The offline evaluator (`eval/run_eval.py`) runs the full pipeline on all 60 pairs and computes four metrics using [RAGAS](https://github.com/explodinggradients/ragas):

| Metric | What it measures | Threshold |
|---|---|---|
| **Faithfulness** | Are claims in the answer supported by retrieved chunks? | ≥ 0.70 |
| **Answer Relevancy** | Does the answer actually address the question? | ≥ 0.70 |
| **Context Recall** | Did retrieval find the chunks needed to answer? | ≥ 0.50 |
| **Context Precision** | Are retrieved chunks actually relevant (low noise)? | ≥ 0.50 |

### CI Pipeline

The workflow (`.github/workflows/eval.yml`) runs three jobs:

| Job | Trigger | What it does |
|---|---|---|
| `unit-tests` | Every PR and push | Mocked tests — no API key needed, fast (~30s) |
| `evaluate` | After unit-tests pass | Live pipeline vs. 60 golden pairs, fails if below threshold |
| `integration-tests` | Push to `main` only | Real OpenAI + HuggingFace calls (`@pytest.mark.integration`) |

```
PR opened
  └── unit-tests
        └── evaluate (OPENAI_API_KEY required, ~5–10 min)
              └── any RAGAS metric < threshold → exit(1) → PR blocked
```

To enforce the quality gate as a hard merge block:
> GitHub → Settings → Branches → Branch protection → `main` → "Require status checks to pass" → select `unit-tests` and `evaluate`

---

## Algorithms

### BM25 (Okapi BM25)

A probabilistic ranking function based on term frequency, inverse document frequency, and length normalization:

```
score(q, d) = Σ IDF(tᵢ) × [f(tᵢ, d) × (k₁ + 1)] / [f(tᵢ, d) + k₁ × (1 - b + b × |d|/avgdl)]
```

- `f(tᵢ, d)` — term frequency in document
- `IDF(tᵢ)` — inverse document frequency (penalizes common terms)
- `k₁`, `b` — saturation and length normalization parameters
- `avgdl` — average document length across the corpus

Implementation: `rank-bm25` library, index built in memory from all corpus chunks.

### Cosine Similarity (Vector Search)

Chunks and queries are embedded into a high-dimensional space. Retrieval finds chunks whose direction is closest to the query embedding:

```
similarity(q, d) = (q · d) / (‖q‖ × ‖d‖)
```

Implementation: ChromaDB with `text-embedding-3-small` (1536-dim vectors).

### Score Fusion

BM25 scores are normalized by the max score in the result set. Vector scores from ChromaDB are already in `[0, 1]`. Fusion is a weighted linear combination:

```python
fused = 0.6 × vector_score + 0.4 × (bm25_score / max_bm25_score)
```

### Cross-Encoder Reranking

`ms-marco-MiniLM-L-6-v2` is a BERT-based model fine-tuned on MS MARCO passage ranking. It concatenates the query and passage and outputs a single relevance score. Because it attends to both sequences jointly, it captures fine-grained interactions that bi-encoders miss — at the cost of running per-candidate at query time.

---

## Project Structure

```
RAG_Project/
├── src/
│   ├── api.py                  # FastAPI app + lifespan state management
│   ├── routes.py               # /chat, /ingest, /health endpoints
│   ├── schemas.py              # Pydantic request/response models
│   ├── main.py                 # build_pipeline() factory (basic / advanced)
│   ├── config/
│   │   ├── config.py           # Config dataclass + validate_for_runtime()
│   │   └── prompts.yaml        # Versioned prompt templates
│   ├── ingestion/
│   │   └── data_processor.py   # HuggingFace dataset loading + chunking
│   ├── retrieval/
│   │   ├── vector_store.py     # ChromaDB wrapper (upsert, reset, search)
│   │   ├── hybrid_search.py    # BM25 + vector score fusion
│   │   └── reranker.py         # CrossEncoder reranker
│   └── pipeline/
│       ├── rag_pipeline.py     # retrieve → generate → faithfulness check
│       └── evaluation.py       # RAGAS evaluator + threshold checker
├── eval/
│   ├── golden_dataset.json     # 60 verified Q&A pairs
│   └── run_eval.py             # Offline eval script (CI entry point)
├── tests/
│   └── test_retrieval.py
├── .github/
│   └── workflows/
│       └── eval.yml            # CI pipeline with quality gate
├── .env.example
└── requirements.txt
```

---

## Setup & Running

### Prerequisites

- Python 3.11+
- OpenAI API key

### Install

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env and set: OPENAI_API_KEY=sk-...
```

### Run the API

```bash
uvicorn src.api:app --reload
```

On first startup, the Wikipedia dataset is ingested into ChromaDB (one-time, ~1–2 min). Subsequent starts load from disk instantly.

### Run Tests

```bash
# Unit tests only — no API key needed
pytest tests/ -m "not integration"

# Integration tests — requires OPENAI_API_KEY
pytest tests/ -m integration
```

### Run Evaluation Locally

```bash
# Print scores
python eval/run_eval.py

# Force a pipeline mode
python eval/run_eval.py --mode basic
python eval/run_eval.py --mode advanced

# CI mode — exits with code 1 if any metric is below threshold
python eval/run_eval.py --fail-on-threshold --top-k 5
```

Results are written to `eval_results/eval_results.json`.

---

## API Reference

### `POST /chat`

Ask a question. Returns an answer grounded in retrieved Wikipedia chunks.

**Request:**
```json
{
  "question": "Who discovered penicillin?",
  "top_k": 5,
  "check_faithfulness": true
}
```

**Response:**
```json
{
  "question": "Who discovered penicillin?",
  "answer": "Penicillin was discovered by Alexander Fleming in 1928. [Source 1]",
  "citations": [
    {
      "source_id": "wiki_1042",
      "text_snippet": "Alexander Fleming observed that a mold contaminating...",
      "score": 0.9134
    }
  ],
  "refused": false
}
```

When `refused: true`, the faithfulness gate determined the answer was not grounded in the retrieved context.

---

### `GET /health`

```json
{ "status": "ok", "chunks_in_store": 8431 }
```

---

### `POST /ingest`

Rebuilds the vector store from scratch. Useful after changing chunking settings.

```json
{ "message": "Full rebuild complete.", "chunks_stored": 8431 }
```

---

## Configuration

All settings are in `src/config/config.py`:

| Setting | Default | Description |
|---|---|---|
| `pipeline_mode` | `"advanced"` | `"basic"` (vector only) or `"advanced"` (hybrid + reranker) |
| `llm_model` | `gpt-3.5-turbo` | OpenAI model for answer generation |
| `embedding_model` | `text-embedding-3-small` | OpenAI embedding model |
| `temperature` | `0.0` | LLM temperature |
| `max_tokens` | `512` | Max tokens in LLM response |
| `chunk_size` | `600` | Characters per chunk |
| `chunk_overlap` | `100` | Overlap between consecutive chunks |
| `top_k` | `5` | Chunks retrieved per query |
| `rerank_top_k` | `3` | Chunks kept after reranking (advanced mode) |
| `bm25_weight` | `0.4` | BM25 contribution to fused score |
| `vector_weight` | `0.6` | Vector similarity contribution to fused score |
| `chroma_persist_dir` | `./chroma_db` | ChromaDB storage path |
| `golden_dataset_path` | `./eval/golden_dataset.json` | Path to golden Q&A pairs |
| `eval_output_dir` | `./eval_results` | Where eval results are written |

---

## Prompt Versioning

All LLM prompts live in `src/config/prompts.yaml`. The file is loaded at startup — updating a prompt requires only a YAML edit and redeploy, no code change.

```yaml
version: "1.0"

rag_prompt:
  name: "basic_rag"
  template: |
    ...answer using ONLY the provided context...

faithfulness_check_prompt:
  name: "faithfulness_check"
  template: |
    ...Is the answer faithfully supported by the context? YES or NO...
```

Both prompts have inline fallback defaults in `rag_pipeline.py` in case the YAML is missing.
