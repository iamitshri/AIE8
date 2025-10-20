"""Tools for the agent"""

from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
import json


def create_rag_tool(rag_graph):
    """
    Factory function to create a RAG tool with a specific graph instance.
    
    Args:
        rag_graph: Compiled RAG graph instance
        
    Returns:
        RAG tool that uses the provided graph
    """
    @tool
    def ai_rag_tool(question: str):
        """
        Use this RAG tool to answer questions based on internal knowledge documents.
        Input should be a fully formed question.
        """
        print(f"RAG tool called with question: {question}")
        response = rag_graph.invoke({"question": question})
        
        # Return BOTH the answer AND the retrieved contexts
        result = {
            "answer": response["response"],
            "contexts": [doc.page_content for doc in response["context"]]
        }
        return json.dumps(result)
    
    return ai_rag_tool


def create_tavily_tool():
    """Create a Tavily search tool for web search"""
    return TavilySearchResults(max_results=5)


def create_arxiv_tool():
    """Create an arXiv search tool for academic papers"""
    return ArxivQueryRun()


def create_all_tools(rag_graph):
    """
    Create all tools for the agent.
    
    Args:
        rag_graph: Compiled RAG graph instance
        
    Returns:
        List of all tools [rag_tool, tavily_tool, arxiv_tool]
    """
    rag_tool = create_rag_tool(rag_graph)
    tavily_tool = create_tavily_tool()
    arxiv_tool = create_arxiv_tool()
    
    return [rag_tool, tavily_tool, arxiv_tool]