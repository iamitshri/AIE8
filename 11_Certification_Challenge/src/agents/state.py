"""State schema for the RAG graph"""

from typing import List
from typing_extensions import TypedDict
from langchain_core.documents import Document


class State(TypedDict):
    """
    State that flows through the RAG graph nodes.
    
    Attributes:
        question (str): The user's input question
        context (List[Document]): Retrieved documents from vector store
        response (str): Generated answer from LLM
    """
    question: str
    context: List[Document]
    response: str