import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_chroma import Chroma
from dotenv import load_dotenv
from config import config
from pydantic import BaseModel
from typing import List
from collections import defaultdict


load_dotenv()

peristent_directory = "db/chroma_db"
embedding_model = OpenAIEmbeddings(model=config.models.embedding_model)
llm = ChatOpenAI(model="gpt-4o", temperature=0)


# **: Pydantic model for structured output


class QueryVariations(BaseModel):
    queries: List[str]


db = Chroma(
    persist_directory=config.storage.persist_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)

original_query = "What is consistent hasing?"
print(f"Original query: {original_query}")


llm_with_tools = llm.with_structured_output(QueryVariations)

prompt = f"""Generate 3 different variations of this query that would help retrieve relevant documents:

Original query: {original_query}

Return 3 alternative queries that rephrase or approach the same question from different angles."""

response = llm_with_tools.invoke(prompt)

query_variations = response.queries

for i, variation in enumerate(query_variations, 1):
    print(f"Variation {i}: {variation}")


""" Use the variations to get results """
retriever = db.as_retriever(search_kwargs={"k": 5})
retrieved_results = []

for i, variation in enumerate(query_variations, 1):
    print(f"Results for query variation {i} : {variation}")
    docs = retriever.invoke(variation)
    retrieved_results.append(docs)
    print(f"Retrieved {len(docs)} documents \n")

    for j, doc in enumerate(docs, 1):
        print(f"Document {j}:")
        print(f"Source {doc.metadata.get('source')}")
        print(f"{doc.page_content[:200]}...\m")

    print("-" * 20)


""" Implementing Reciprocal Rank Fusion """


def reciprocal_rank_fusion(chunk_lists, k=60, verbose=True):

    if verbose:
        print("\n" + "=" * 60)
        print("APPLYING RECIPROCAL RANK FUSION")
        print("=" * 60)
        print(f"\nUsing k={k}")
        print("Calculating RRF scores...\n")

    # Data structures for RRF calculation
    rrf_scores = defaultdict(float)  # Will store: {chunk_content: rrf_score}
    all_unique_chunks = {}  # Will store: {chunk_content: actual_chunk_object}

    # For verbose output - track chunk IDs
    chunk_id_map = {}
    chunk_counter = 1

    # Go through each retrieval result
    for query_idx, chunks in enumerate(chunk_lists, 1):
        if verbose:
            print(f"Processing Query {query_idx} results:")

        # Go through each chunk in this query's results
        for position, chunk in enumerate(chunks, 1):  # position is 1-indexed
            # Use chunk content as unique identifier
            chunk_content = chunk.page_content

            # Assign a simple ID if we haven't seen this chunk before
            if chunk_content not in chunk_id_map:
                chunk_id_map[chunk_content] = f"Chunk_{chunk_counter}"
                chunk_counter += 1

            chunk_id = chunk_id_map[chunk_content]

            # Store the chunk object (in case we haven't seen it before)
            all_unique_chunks[chunk_content] = chunk

            # Calculate position score: 1/(k + position)
            position_score = 1 / (k + position)

            # Add to RRF score
            rrf_scores[chunk_content] += position_score

            if verbose:
                print(
                    f"  Position {position}: {chunk_id} +{position_score:.4f} (running total: {rrf_scores[chunk_content]:.4f})"
                )
                print(f"    Preview: {chunk_content[:80]}...")

        if verbose:
            print()

    # Sort chunks by RRF score (highest first)
    sorted_chunks = sorted(
        [
            (all_unique_chunks[chunk_content], score)
            for chunk_content, score in rrf_scores.items()
        ],
        key=lambda x: x[1],  # Sort by RRF score
        reverse=True,  # Highest scores first
    )

    if verbose:
        print(
            f"✅ RRF Complete! Processed {len(sorted_chunks)} unique chunks from {len(chunk_lists)} queries."
        )

    return sorted_chunks


fused_results = reciprocal_rank_fusion(retrieved_results, k=60, verbose=True)


print("\n" + "=" * 60)
print("FINAL RRF RANKING")
print("=" * 60)

print(f"\nTop {min(10, len(fused_results))} documents after RRF fusion:\n")

for rank, (doc, rrf_score) in enumerate(fused_results[:10], 1):
    print(f"🏆 RANK {rank} (RRF Score: {rrf_score:.4f})")
    print(f"{doc.page_content[:200]}...")
    print("-" * 50)

print(
    f"\n✅ RRF Complete! Fused {len(fused_results)} unique documents from {len(query_variations)} query variations."
)
print("\n💡 Key benefits:")
print("   • Documents appearing in multiple queries get boosted scores")
print("   • Higher positions contribute more to the final score")
print("   • Balanced fusion using k=60 for gentle position penalties")
