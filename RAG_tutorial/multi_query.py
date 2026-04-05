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
