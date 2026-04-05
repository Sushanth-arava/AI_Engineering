import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_chroma import Chroma
from dotenv import load_dotenv
from config import config

load_dotenv()

embedding_model = OpenAIEmbeddings(model=config.models.embedding_model)

db = Chroma(
    persist_directory=config.storage.persist_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)

query = "What is load balancing?"

# **: Method 1

retriever = db.as_retriever(search_kwargs={"k": config.retrieval.k})

# **: Method 2 => Similarity with score threshold
# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={"k": 3, "score_threshold": 0},
# )

# **: Method 3 => Maximum Marginal Relevance(MMR)
# retriever = db.as_retriever(
#     search_type="mmr",
#     search_kwargs={
#         "k": 3,  # Final number of docs
#         "fetch_k": 10,  # Initial pool to select from
#         "lambda_mult": 0.5,  # 0=max diversity, 1=max relevance
#     },
# )


docs = retriever.invoke(query)

print(f"Retrieved {len(docs)} documents of threshold:0.3 \n")

for i, doc in enumerate(docs, 1):
    print(f"Document {i}: ")
    print(f"{doc.page_content}\n")

print("-" * 60)
