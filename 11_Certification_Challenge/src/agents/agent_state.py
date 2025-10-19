"""State schema for the agentic loop"""

from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages
from langchain_core.documents import Document


class AgentState(TypedDict):
    """
    State for the agentic loop with tool calling.
    
    Attributes:
        messages: Conversation history with add_messages reducer
        context: Retrieved documents from RAG tool (when used)
    """
    messages: Annotated[list, add_messages]
    context: List[Document]