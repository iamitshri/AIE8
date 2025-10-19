""" Retriever for querying the vector store """


from langchain_qdrant import QdrantVectorStore


def create_retriever(vector_store: QdrantVectorStore, k: int = 5):
    """ Create a retriever for querying the vector store """
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    print(f"Successfully created a retriever with {k} results")
    return retriever

def retrieve_documents(question: str, retriever):
    """ Retrieve documents from the vector store """
    retrieved_docs = retriever.invoke(question)
    print(f"Successfully retrieved {len(retrieved_docs)} documents from the vector store for the question: {question}")
    return retrieved_docs

if __name__ == "__main__":
    from src.rag.indexer import create_and_populate_vector_store
    from src.rag.loader import load_documents
    from src.rag.chunker import chunk_documents
    
    print("Loading documents...")
    docs = load_documents("data")
    
    print("Chunking documents...")
    chunks = chunk_documents(docs)
    
    print("Creating vector store...")
    vector_store = create_and_populate_vector_store(chunks)
    
    print("Creating retriever...")
    retriever = create_retriever(vector_store, k=5)
    
    print("Testing retrieval...")
    question = "What is the main purpose of the document?"
    results = retrieve_documents(retriever, question)
    
    print(f"\n📄 Retrieved {len(results)} documents:")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc.page_content[:200]}...")