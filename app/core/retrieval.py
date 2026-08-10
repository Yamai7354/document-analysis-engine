from typing import List, Dict, Any

from llm import get_chat_llm

_llm = None

ANSWER_PROMPT = """You are a helpful assistant answering questions using only the \
provided document excerpts. If the excerpts don't contain the answer, say so \
plainly rather than guessing.

Question: {query}

Document excerpts:
{context}

Answer:"""


def get_llm():
    global _llm
    if _llm is None:
        _llm = get_chat_llm()
    return _llm


def format_context(docs: List[Any]) -> str:
    """
    Formats retrieved documents into a single context string.
    """
    context_str = ""
    for i, doc in enumerate(docs):
        source = getattr(doc, "metadata", {}).get("source", "unknown")
        context_str += f"--- Source {i + 1}: {source} ---\n{doc.page_content}\n\n"
    return context_str


def generate_answer(query: str, retrieved_docs: List[Any]) -> Dict[str, Any]:
    """
    Generates an answer to `query` grounded in `retrieved_docs`, via the
    configured LLM. Short-circuits (no LLM call) when there are no documents.
    """
    if not retrieved_docs:
        return {
            "answer": (
                "I could not find any relevant information in the knowledge "
                "base. Try uploading a document first."
            ),
            "source_documents": [],
        }

    context = format_context(retrieved_docs)
    prompt = ANSWER_PROMPT.format(query=query, context=context)
    response = get_llm().invoke(prompt)

    return {
        "answer": response.content,
        "source_documents": [doc.page_content for doc in retrieved_docs],
    }
