""" Vector store indexer for the RAG system """


from langchain_qdrant import QdrantVectorStore
from typing import List
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance,VectorParams

def create_and_populate_vector_store(
    docs: List[Document], 
    collection_name: str = "rag_collection", 
    embedding: OpenAIEmbeddings = None) -> QdrantVectorStore:
    """ Create and populate a vector store with the documents """
    
    if not docs:
        raise ValueError("No documents found to index into the vector store")
      
    if embedding is None:
        embedding = OpenAIEmbeddings(model="text-embedding-3-small")
        
    client = QdrantClient(":memory:")
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding,
    )
    vector_store.add_documents(documents=docs)
    print(f"Successfully created and populated vector store with {len(docs)} documents")
    return vector_store


if __name__ == "__main__":
    from src.rag.loader import load_documents
    from src.rag.chunker import chunk_documents
    
    print("Loading documents...")
    docs = load_documents("data")
    
    print("Chunking documents...")
    chunks = chunk_documents(docs)
    
    print("Creating and populating vector store...")
    vector_store = create_and_populate_vector_store(chunks)
    print(vector_store)
    print(f"Successfully created and populated vector store with {len(chunks)} indexed chunks")