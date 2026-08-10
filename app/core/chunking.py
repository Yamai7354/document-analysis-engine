from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CONFIG

_CHUNKING_CFG = CONFIG.get("chunking", {})


def chunk_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """
    Splits a list of standard LangChain Documents into smaller chunks while preserving some overlap.
    """
    chunk_size = chunk_size or _CHUNKING_CFG.get("chunk_size", 1000)
    chunk_overlap = chunk_overlap or _CHUNKING_CFG.get("chunk_overlap", 200)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    return text_splitter.split_documents(documents)
