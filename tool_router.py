"""
Tool registry: the real capabilities the agent can call — RAG search over
the uploaded documents, document listing/stats, and chart generation —
exposed as LangChain tools so the LLM can invoke them via function calling.

Tools are built per-collection via `make_tools(collection_name)` so callers
can scope them to a single shared knowledge base (the default, used by the
CLI/API for normal single-operator use) or to a per-visitor collection (used
by the public demo, so visitors don't see each other's uploads).
"""
import os
import re
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from langchain_core.documents import Document
from langchain_core.tools import BaseTool, tool

from app.core.vectorstore import search_documents, get_chunk_counts_by_source
from config import CONFIG

CHARTS_DIR = CONFIG.get("charts", {}).get("output_dir", "charts")
DEFAULT_COLLECTION = "documents"


def format_search_results(docs: List[Document]) -> str:
    if not docs:
        return "No relevant passages found in the knowledge base."
    lines = []
    for i, doc in enumerate(docs):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        lines.append(f"[{i + 1}] ({source}) {doc.page_content}")
    return "\n\n".join(lines)


def _safe_chart_filename(collection_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", collection_name)
    return f"chunks_per_document_{slug}.png"


def make_tools(
    collection_name: str = DEFAULT_COLLECTION,
) -> Tuple[List[BaseTool], Dict[str, BaseTool]]:
    """
    Builds a fresh set of tools bound to `collection_name`. Cheap to call —
    no I/O happens until a tool is actually invoked.
    """

    @tool
    def search_knowledge_base(query: str) -> str:
        """Search the uploaded documents for passages relevant to `query`.
        Use this before answering any question about document content — do
        not answer from general knowledge. Returns the most relevant
        excerpts along with their source filenames."""
        docs = search_documents(query, collection_name=collection_name)
        return format_search_results(docs)

    @tool
    def list_documents() -> str:
        """List every document currently in the knowledge base, with how
        many chunks each was split into."""
        counts = get_chunk_counts_by_source(collection_name=collection_name)
        if not counts:
            return "No documents have been uploaded yet."
        return "\n".join(f"- {name}: {n} chunks" for name, n in counts.items())

    @tool
    def get_document_stats() -> str:
        """Get aggregate statistics about the knowledge base: number of
        documents and total chunks stored."""
        counts = get_chunk_counts_by_source(collection_name=collection_name)
        total_docs = len(counts)
        total_chunks = sum(counts.values())
        return f"{total_docs} document(s), {total_chunks} chunk(s) total."

    @tool
    def generate_chart() -> str:
        """Generate a bar chart showing how many chunks each uploaded
        document contributed to the knowledge base, save it as a PNG, and
        return the filename. Use this when the user asks for a chart,
        graph, or visualization of the documents."""
        counts = get_chunk_counts_by_source(collection_name=collection_name)
        if not counts:
            return "No documents to chart yet — upload something first."

        os.makedirs(CHARTS_DIR, exist_ok=True)
        names = list(counts.keys())
        values = list(counts.values())

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(names, values, color="#2F6FED")
        ax.set_ylabel("Chunks")
        ax.set_title("Chunks per document")
        ax.tick_params(axis="x", rotation=30)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        fig.tight_layout()

        filename = _safe_chart_filename(collection_name)
        filepath = os.path.join(CHARTS_DIR, filename)
        fig.savefig(filepath, dpi=150)
        plt.close(fig)

        return f"Chart saved as {filename}"

    tools = [search_knowledge_base, list_documents, get_document_stats, generate_chart]
    return tools, {t.name: t for t in tools}


def search_knowledge_base_raw(
    query: str, collection_name: str = DEFAULT_COLLECTION
) -> List[Document]:
    """The underlying search call, exposed separately so callers (agent.py)
    can access the structured Documents, not just the LLM-facing string."""
    return search_documents(query, collection_name=collection_name)


# Default tools bound to the single shared collection — used by the CLI, the
# FastAPI backend, and anywhere else that isn't the per-visitor public demo.
TOOLS, TOOLS_BY_NAME = make_tools(DEFAULT_COLLECTION)
