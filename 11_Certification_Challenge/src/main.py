"""FastAPI application with agent system"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import initialization functions
from src.rag.loader import load_documents
from src.rag.chunker import chunk_documents
from src.rag.indexer import create_and_populate_vector_store
from src.rag.retriever import create_retriever
from src.rag.generator import create_llm
from src.agents.graph import create_rag_graph
from src.agents.tools import create_all_tools
from src.agents.agent_graph import create_agent_graph
from src.config.settings import settings
from langchain_openai import ChatOpenAI
import src.app_state as app_state
 


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager - runs on startup and shutdown.
    
    This initializes the RAG system and agent ONCE when the server starts,
    not on every request.
    """
    
    print("\n" + "="*80)
    print("🚀 SERVER STARTING - INITIALIZING AI SYSTEMS")
    print("="*80)
    
    try:
        # Initialize RAG system
        print("\n1️⃣ Loading and indexing documents...")
        docs = load_documents("data")
        chunks = chunk_documents(docs)
        vector_store = create_and_populate_vector_store(chunks)
        retriever = create_retriever(vector_store, k=5)
        llm = create_llm()
        
        print("\n2️⃣ Building RAG graph...")
        app_state.RAG_GRAPH = create_rag_graph(retriever, llm)
        
        print("\n3️⃣ Creating agent with tools...")
        tools = create_all_tools(app_state.RAG_GRAPH)
        model = ChatOpenAI(model=settings.LLM_MODEL, temperature=0)
        model_with_tools = model.bind_tools(tools)
        app_state.AGENT_GRAPH = create_agent_graph(model_with_tools, tools)
        
        print("\n" + "="*80)
        print("✅ AI SYSTEMS READY - SERVER IS LIVE")
        print("="*80 + "\n")
        
        yield  # Server runs here
        
    except Exception as e:
        print(f"\n❌ ERROR DURING STARTUP: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # Cleanup on shutdown (if needed)
        print("\n👋 Server shutting down...")


# Create FastAPI app with lifespan
app = FastAPI(
    title="AI Agent API",
    description="RAG-powered AI agent with tool calling",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include API routes
from src.api import routes
app.include_router(routes.router)


@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {
        "status": "healthy",
        "message": "AI Agent API is running",
        "rag_initialized": app_state.RAG_GRAPH is not None,
        "agent_initialized": app_state.AGENT_GRAPH is not None
    }   


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "systems": {
            "rag_graph": "ready" if app_state.RAG_GRAPH is not None else "not initialized",
            "agent_graph": "ready" if app_state.AGENT_GRAPH is not None else "not initialized"
        },
        "model": settings.LLM_MODEL
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host="localhost",
        port=8000,
        reload=True  # Auto-reload on code changes
    )

