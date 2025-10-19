"""RAG Graph builder using LangGraph"""

from langgraph.graph import START, StateGraph
from src.agents.state import State
from src.agents.nodes import create_retrieve_node, create_generate_node


def create_rag_graph(retriever, llm):
    """
    Build and compile the RAG graph.
    
    Args:
        retriever: The retriever instance
        llm: The LLM instance
        
    Returns:
        Compiled LangGraph ready to invoke
        
    Example:
        >>> retriever = create_retriever(vector_store)
        >>> llm = create_llm()
        >>> graph = create_rag_graph(retriever, llm)
        >>> result = graph.invoke({"question": "What is AI?"})
        >>> print(result["response"])
    """
    # Step 1: Create nodes using the factory functions
    retrieve_node = create_retrieve_node(retriever)
    generate_node = create_generate_node(llm)
    
    # Step 2: Build the graph
    graph = StateGraph(State).add_sequence([retrieve_node, generate_node])

    # Step 3: Add starting edge
    graph.add_edge(START, "retrieve_node")
    
    # Step 4: Compile and return
    compiled_graph = graph.compile()
    print("Successfully compiled RAG graph")
    return compiled_graph


if __name__ == "__main__":
    """Test the graph with the full RAG pipeline"""
    from src.rag.loader import load_documents
    from src.rag.chunker import chunk_documents
    from src.rag.indexer import create_and_populate_vector_store
    from src.rag.retriever import create_retriever
    from src.rag.generator import create_llm
    
    print("Initializing RAG system...")
    
    # Initialize RAG components
    docs = load_documents("data")
    chunks = chunk_documents(docs)
    vector_store = create_and_populate_vector_store(chunks)
    retriever = create_retriever(vector_store, k=5)
    llm = create_llm()
    
    print("\nBuilding graph...")
    graph = create_rag_graph(retriever, llm)
    
    print("\nTesting graph with question...")
    test_question = "How do people use AI?"
    result = graph.invoke({"question": test_question})
    
    print("\n" + "="*80)
    print("QUESTION:", test_question)
    print("="*80)
    print("ANSWER:", result["response"])
    print("="*80)
    print(f"\nContext had {len(result['context'])} documents")
