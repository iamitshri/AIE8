from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.rag.loader import load_documents

def chunk_documents(docs: List[Document], chunk_size: int = 750, chunk_overlap: int = 100) -> List[Document]:
    """ Chunk documents into smaller chunks """
    if not docs:
        raise ValueError("No documents found to chunk")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(docs)
    if not split_docs:
        raise ValueError(f"Text splitter failed to create chunks from the documents found in {len(docs)} documents"
                         f"please check if the documents are valid."
                         f"please check if the chunk size and overlap are appropriate."
                         )
    print(f"Successfully chunked {len(docs)} documents into {len(split_docs)} chunks")
    return split_docs

if __name__ == "__main__":
    docs = load_documents("data")
    split_docs = chunk_documents(docs, chunk_size=750, chunk_overlap=100)
    print(split_docs)