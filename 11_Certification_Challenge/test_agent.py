"""Test script for the agent system with tools"""

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# RAG system imports
from src.rag.loader import load_documents
from src.rag.chunker import chunk_documents
from src.rag.indexer import create_and_populate_vector_store
from src.rag.retriever import create_retriever
from src.rag.generator import create_llm

# Simple RAG graph
from src.agents.graph import create_rag_graph

# Agent system imports
from src.agents.tools import create_all_tools
from src.agents.agent_graph import create_agent_graph
from src.config.settings import settings


def initialize_rag_system():
    """Initialize the RAG system and return RAG graph"""
    print("=" * 80)
    print("INITIALIZING RAG SYSTEM")
    print("=" * 80)
    
    print("\n1. Loading documents...")
    docs = load_documents("data")
    
    print("\n2. Chunking documents...")
    chunks = chunk_documents(docs)
    
    print("\n3. Creating vector store...")
    vector_store = create_and_populate_vector_store(chunks)
    
    print("\n4. Creating retriever...")
    retriever = create_retriever(vector_store, k=5)
    
    print("\n5. Creating LLM...")
    llm = create_llm()
    
    print("\n6. Building RAG graph...")
    rag_graph = create_rag_graph(retriever, llm)
    
    print("\n✅ RAG system initialized!\n")
    return rag_graph


def create_agent(rag_graph):
    """Create the agent with all tools"""
    print("=" * 80)
    print("CREATING AGENT")
    print("=" * 80)
    
    print("\n1. Creating tools (RAG, Tavily, Arxiv)...")
    tools = create_all_tools(rag_graph)
    print(f"   Created {len(tools)} tools:")
    for tool in tools:
        print(f"   - {tool.name}: {tool.description}")
    
    print("\n2. Creating model with tools...")
    model = ChatOpenAI(model=settings.LLM_MODEL, temperature=0)
    model_with_tools = model.bind_tools(tools)
    print(f"   Model: {settings.LLM_MODEL}")
    
    print("\n3. Building agent graph...")
    agent_graph = create_agent_graph(model_with_tools, tools)
    
    print("\n✅ Agent created!\n")
    return agent_graph


def test_agent_query(agent_graph, question):
    """Test the agent with a specific question"""
    print("=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)
    
    # Create input with HumanMessage
    inputs = {"messages": [HumanMessage(content=question)]}
    
    # Invoke agent
    print("\nAgent thinking...\n")
    result = agent_graph.invoke(inputs)
    
    # Extract final answer
    final_answer = result["messages"][-1].content
    
    print("\n" + "=" * 80)
    print("FINAL ANSWER:")
    print("=" * 80)
    print(final_answer)
    print("=" * 80)
    
    return result


def main():
    """Main test function"""
    print("\n" + "🚀" * 40)
    print("TESTING AGENT SYSTEM WITH TOOLS")
    print("🚀" * 40 + "\n")
    
    # Initialize RAG system
    rag_graph = initialize_rag_system()
    
    # Create agent
    agent_graph = create_agent(rag_graph)
    
    # Test questions
    test_questions = [
        # Should use RAG tool (internal documents)
        "How do people use AI according to our internal documents?",
        
        # Could use web search (current events)
        "What are the latest AI developments this week?",
        
        # Could use arxiv (research papers)
        "Find recent research papers on large language models",
    ]
    
    print("\n" + "🧪" * 40)
    print("RUNNING TESTS")
    print("🧪" * 40 + "\n")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/{len(test_questions)}")
        print(f"{'='*80}\n")
        
        try:
            result = test_agent_query(agent_graph, question)
            print("\n✅ Test passed!\n")
        except Exception as e:
            print(f"\n❌ Test failed: {e}\n")
            import traceback
            traceback.print_exc()
    
    print("\n" + "🎉" * 40)
    print("ALL TESTS COMPLETED")
    print("🎉" * 40 + "\n")


if __name__ == "__main__":
    main()

