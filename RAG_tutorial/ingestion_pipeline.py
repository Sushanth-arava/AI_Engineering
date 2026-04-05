import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from config import config

load_dotenv()


""" Load all the documents """

def load_documents(docs_path="Books"):
    """Load all PDF documents from Books."""
    print("Loading all documents....")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"The directory {docs_path} does not exist. Please create it and add your files."
        )

    loader = DirectoryLoader(
        path=docs_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(
            f"No .pdf files found in {docs_path}. Please add your PDF documents."
        )

    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i + 1}:")
        print(f"  Source: {doc.metadata.get('source')}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  Metadata: {doc.metadata}")

    return documents


""" Chunck the documents """

def chunck_documents(documents, chunk_size=config.chunking.chunk_size, chunk_overlap=config.chunking.chunk_overlap):

    print("Splitting the documents into chunks...")

    text_splitter= CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks=text_splitter.split_documents(documents)

    if chunks:

        for i, chunk in enumerate(chunks[:2]):
            print(f"\n ----chunk{i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)

        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")

    return chunks
    

""" store the embedding in the vector database """

def create_vector_store(chunks, persist_directory=config.storage.persist_directory):

    """ create and persiste chromaDB vector store """

    print("Creating embeddings and storing in chroma DB")

    embedding_model=OpenAIEmbeddings(model=config.models.embedding_model)

    print("----creating a vector store---")

    vector_store=Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hsnw:space":"cosine"}
    )
    print(f"Vector is created and stored in {persist_directory}")

    return vector_store

    
    

def main():
    print("Main")
    docs_path = "Books"
    documents = load_documents(docs_path)
    chunks = chunck_documents(documents)
    embeddings=create_vector_store(chunks)
    print(embeddings)

if __name__ == "__main__":
    main()