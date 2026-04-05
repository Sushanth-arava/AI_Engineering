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

query = "What is a potato?"

# **: Method 1 => Get k similar

retriever = db.as_retriever(search_kwargs={"k": config.retrieval.k})
# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={"k": 5, "score_threshold": 0.3},
# )
relavant_docs = retriever.invoke(query)


combined_input = f"""Based on the knowledge from the documents,answer this query: {query}


Documents: {chr(10).join([f"- {doc.page_content}" for doc in relavant_docs])}

Please provide a clear, detailed and helpful answer using only the information from these documents. If you can't find any relavent information, please say 'I couldn't find any info from these docs'

"""
model = ChatOpenAI(model=config.models.llm_model)
messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content=combined_input),
]

print(f"User query:{query}")

print("--context--")

for i, doc in enumerate(relavant_docs, 1):
    source = doc.metadata.get("source", "Unknown")
    page = doc.metadata.get("page", "?")
    print(f"Document {i} | Source: {source} | Page: {page}\n {doc.page_content}\n")


result = model.invoke(messages)

print("-----Generated Response-----")

print("Content only: ")
print(result.content)
