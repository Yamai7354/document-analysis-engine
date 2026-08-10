import os
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from app.core.embeddings import get_embedder
from config import CONFIG

PERSIST_DIRECTORY = os.path.join(os.getcwd(), "models", "chroma_db")
_DEFAULT_TOP_K = CONFIG.get("retrieval", {}).get("top_k", 4)


def get_vectorstore(collection_name: str = "documents"):
    """
    Gets or creates a Chroma vector store instance.
    """
    embedder = get_embedder()
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedder,
        persist_directory=PERSIST_DIRECTORY,
    )
    return vectorstore


def add_documents_to_store(
    documents: List[Document], collection_name: str = "documents"
):
    """
    Adds chunked documents to the Chroma vector store.
    """
    vectorstore = get_vectorstore(collection_name)
    vectorstore.add_documents(documents)
    return vectorstore


def search_documents(
    query: str, collection_name: str = "documents", k: int = _DEFAULT_TOP_K
):
    """
    Searches the vector store for the top k most similar documents to the query.
    """
    vectorstore = get_vectorstore(collection_name)
    return vectorstore.similarity_search(query, k=k)


def get_chunk_counts_by_source(collection_name: str = "documents") -> dict:
    """
    Returns {source_filename: chunk_count} for everything stored in the
    vector DB. Used for the document list and stats/chart tools.
    """
    vectorstore = get_vectorstore(collection_name)
    data = vectorstore.get(include=["metadatas"])
    counts: dict = {}
    for meta in data.get("metadatas", []) or []:
        source = os.path.basename((meta or {}).get("source", "unknown"))
        counts[source] = counts.get(source, 0) + 1
    return counts
