from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────────────────────────
# SETUP: Create our sample company data
# ──────────────────────────────────────────────────────────────────

chunks = [
    "Microsoft acquired GitHub for 7.5 billion dollars in 2018.",
    "Tesla Cybertruck production ramp begins in 2024.",
    "Google is a large technology company with global operations.",
    "Tesla reported strong quarterly results. Tesla continues to lead in electric vehicles. Tesla announced new manufacturing facilities.",
    "SpaceX develops Starship rockets for Mars missions.",
    "The tech giant acquired the code repository platform for software development.",
    "NVIDIA designs Starship architecture for their new GPUs.",
    "Tesla Tesla Tesla financial quarterly results improved significantly.",
    "Cybertruck reservations exceeded company expectations.",
    "Microsoft is a large technology company with global operations.",
    "Apple announced new iPhone features for developers.",
    "The apple orchard harvest was excellent this year.",
    "Python programming language is widely used in AI.",
    "The python snake can grow up to 20 feet long.",
    "Java coffee beans are imported from Indonesia.",
    "Java programming requires understanding of object-oriented concepts.",
    "Orange juice sales increased during winter months.",
    "Orange County reported new housing developments.",
]
# TODO: Loop through all the chunks and to convert them into langchain documents

documents = [
    Document(page_content=chunk, metadata={"source": f"chunk {i}"})
    for i, chunk in enumerate(chunks)
]

# print("Sample data")

# for i, chunk in enumerate(documents):
#     print(f" {i}: {chunk}")

# **: Vector retriever


# print("Setting up vector search retriever....")
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_configuration={"hnsw:space": "cosine"},
)

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


# test_query = "space exploration company"

# print(f"Testing, {test_query}")

# test_docs = vector_retriever.invoke(test_query)

# for docs in test_docs:
#     print(f"{docs}")


# ** ────────────────────────────────────
# ** BM25 Retriever
# ** ────────────────────────────────────

# print("setting up bm25 retriever...")

bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 3

# # test_query = "Tesla"
# test_query = "Tesla Cybertruck"

# print(f"Testing... {test_query}")

# test_docs = bm25_retriever.invoke(test_query)

# for docs in test_docs[:2]:
#     print(f"Found: {docs.page_content}")

# ** ──────────────────────────────────────────────────────────────────
# ** Hybrid Retriver
# ** ──────────────────────────────────────────────────────────────────

print("Setting up Hybrid retriver..")

hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5]
)


# test_query = "purchase cost 7.5 billion"
test_query = "electric vehicle manufacturing"
retrieved_chunks = hybrid_retriever.invoke(test_query)

for i, doc in enumerate(retrieved_chunks):
    print(f"{i}, {doc.page_content}")
