""" Generator for RAG: Combines context and question to generate answers. """

from typing import List
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from src.config.settings import settings


RAG_PROMPT = """\
You are a helpful assistant who can answer questions based on the following context only:
If you cannot answer the question based on the context - you must say "I don't know".

### Context:
{context}

### Question:
{question}
"""

def create_llm(model: str = None, temperature: float = None):
    """ 
    Create an LLM instance
    
    Args: 
        model: str - Model name (defaults to settings.LLM_MODEL = "gpt-4o-mini")
        temperature: float - Temperature (defaults to settings.LLM_TEMPERATURE = 0.0)
        
    Returns:
        ChatOpenAI: An instance of the ChatOpenAI class
    """
    model = model or settings.LLM_MODEL
    temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
    print(f"Creating LLM with model={model}, temperature={temperature}")
    return ChatOpenAI(model=model, temperature=temperature)

def create_prompt(context: str, question: str):
    """ Create a prompt for the RAG system """
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    messages = prompt.format_messages(context=context, question=question)
    print(f"Successfully created a prompt for the RAG system with {len(messages)} messages")
    return messages


def generate_answer(question: str, context:List[Document],llm=None)-> str:
    """ 
    Generate an answer based on the context and question
    
    Args:
        question: str
        context: List[Document]
        llm: ChatOpenAI = None
        
    Returns:
        str: The generated answer
    """

    if not context:
        raise ValueError("No context provided to generate an answer")
    if not question:
        raise ValueError("No question provided to generate an answer")
    if not llm:
        llm = create_llm()
    docs_content = "\n\n".join(doc.page_content for doc in context)
    print(f"Successfully created the context content for the RAG system with {len(docs_content)} characters")
    messages = create_prompt(docs_content, question)
    response = llm.invoke(messages)
    if not response:
        raise ValueError("No response generated from the LLM")
    return response.content

if __name__ == "__main__":
    from src.rag.retriever import retrieve_documents, create_retriever
    from src.rag.indexer import create_and_populate_vector_store
    from src.rag.loader import load_documents
    from src.rag.chunker import chunk_documents
    
    print("🚀 Testing Full RAG Pipeline with Generation\n")
    
    print("1️⃣ Loading documents...")
    docs = load_documents("data")
    
    print("\n2️⃣ Chunking documents...")
    chunks = chunk_documents(docs)
    
    print("\n3️⃣ Creating vector store...")
    vector_store = create_and_populate_vector_store(chunks)
    
    print("\n4️⃣ Creating retriever...")
    retriever = create_retriever(vector_store, k=5)
    
    print("\n5️⃣ Creating LLM...")
    llm = create_llm()
    print("✅ LLM created successfully")
    
    print("\n6️⃣ Testing RAG generation...")
    question = "How do people use AI?"
    print(f"Question: {question}\n")
    
    # Retrieve relevant context
    context_docs = retrieve_documents(question, retriever)
    
    # Generate answer
    answer = generate_answer(question, context_docs, llm)
    
    print("\n" + "="*50)
    print("🎯 GENERATED ANSWER:")
    print("="*50)
    print(answer)
    print("="*50)
    
    print(f"\n✅ Full RAG pipeline completed successfully!")
    print(f"   - Retrieved {len(context_docs)} documents")
    print(f"   - Generated answer with {len(answer)} characters")
