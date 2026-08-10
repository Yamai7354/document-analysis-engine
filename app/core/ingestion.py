import os
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)


def ingest_document(file_path: str) -> List[Document]:
    """
    Ingests a document based on its extension and returns a list of generic LangChain Documents.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    elif ext == ".md":
        loader = UnstructuredMarkdownLoader(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    return loader.load()


def ingest_directory(dir_path: str) -> List[Document]:
    """
    Ingests all supported documents in a given directory.
    """
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    all_docs = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            ext = Path(file).suffix.lower()
            if ext in [".pdf", ".txt", ".md"]:
                file_path = os.path.join(root, file)
                try:
                    docs = ingest_document(file_path)
                    all_docs.extend(docs)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

    return all_docs
