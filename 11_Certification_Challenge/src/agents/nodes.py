""" Nodes for the RAG graph """

from src.rag.retriever import retrieve_documents
from src.agents.state import State
from src.rag.generator import generate_answer

def create_retrieve_node(retriever):
    """ Create a retrieve node for the RAG graph """
    def retrieve_node(state: State):
        """ Retrieve documents from the vector store """
        question = state["question"]
        print(f"Retrieving documents for: {question}")
        context = retrieve_documents(question, retriever)
        print(f"Retrieved {len(context)} documents")
        return {"context": context}
    return retrieve_node


def create_generate_node(llm):
    """ Create a generate node for the RAG graph """
    def generate_node(state: State):
        """ Generate an answer from the context """
        context = state["context"]
        question = state["question"]
        print(f"Generating answer for: {question}")
        answer = generate_answer(question, context, llm)
        print(f"Generated answer ({len(answer)} chars)")
        return {"response": answer}
    return generate_node