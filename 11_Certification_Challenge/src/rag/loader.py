""" Document loader for the PDF files """

from pathlib import Path
from typing import List

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document


def load_documents(data_dir:str ="data") -> List[Document]:
    """ Load documents from the data directory """
    path=Path(data_dir)
    if not path.exists():
        raise ValueError(f"Directory {data_dir} does not exist")
    if not path.is_dir():
        raise ValueError(f"Path {data_dir} is not a directory")
    loader = DirectoryLoader(str(data_dir), glob="**/*.pdf", loader_cls=PyMuPDFLoader)
    docs = loader.load()
    if not docs:
        raise ValueError(f"No documents found in {data_dir}"
                         f"Please check if the directory contains PDF files"
                         f"and if the files are readable."
                         )
    print(f"Loaded {len(docs)} documents from {data_dir}")
    return docs

if __name__ == "__main__":
    docs = load_documents("data")
    print(docs)