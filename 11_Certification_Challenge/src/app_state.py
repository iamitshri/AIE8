""" Shared application state for the FastAPI application """


# Global state - initialized in main.py lifespan, used in routes.py
RAG_GRAPH = None
AGENT_GRAPH = None