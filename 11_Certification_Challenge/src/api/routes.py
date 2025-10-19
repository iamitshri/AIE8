"""API routes for the AI agent"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from langchain_core.messages import HumanMessage
import src.app_state as app_state
router = APIRouter(prefix="/api", tags=["agent"])


class QueryRequest(BaseModel):
    """Request model for asking questions"""
    question: str
    use_agent: bool = True  # If False, use simple RAG only


class QueryResponse(BaseModel):
    """Response model for answers"""
    answer: str
    tool_calls: List[str] = []
    context_count: int = 0


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Main endpoint to query the AI system.
    
    Args:
        request: Question and settings
        
    Returns:
        Answer and metadata
    """
    if request.use_agent:
        # Use agent with tools
        if app_state.AGENT_GRAPH is None:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        try:
            # Invoke agent
            inputs = {"messages": [HumanMessage(content=request.question)]}
            result = app_state.AGENT_GRAPH.invoke(inputs)
            
            # Extract answer from last message
            final_message = result["messages"][-1]
            answer = final_message.content
            
            # Track which tools were called
            tool_calls = []
            for msg in result["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tool_calls.extend([tc["name"] for tc in msg.tool_calls])
            
            # Get context count if available
            context_count = len(result.get("context", []))
            
            return QueryResponse(
                answer=answer,
                tool_calls=tool_calls,
                context_count=context_count
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")
    
    else:
        # Use simple RAG only
        if app_state.RAG_GRAPH is None:
            raise HTTPException(status_code=503, detail="RAG system not initialized")
        
        try:
            # Invoke simple RAG
            result = app_state.RAG_GRAPH.invoke({"question": request.question})
            
            return QueryResponse(
                answer=result["response"],
                tool_calls=["rag_graph"],
                context_count=len(result.get("context", []))
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"RAG error: {str(e)}")


@router.get("/tools")
async def get_tools():
    """Get list of available tools"""
    if app_state.AGENT_GRAPH is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    # Return tool information
    return {
        "tools": [
            {
                "name": "ai_rag_tool",
                "description": "Answer questions based on internal documents"
            },
            {
                "name": "tavily_search",
                "description": "Search the web for current information"
            },
            {
                "name": "arxiv",
                "description": "Search academic papers on arxiv.org"
            }
        ]
    }


@router.get("/stats")
async def get_stats():
    """Get system statistics"""
    return {
        "status": "operational",
        "endpoints": {
            "query": "/api/query",
            "tools": "/api/tools",
            "stats": "/api/stats"
        }
    }

