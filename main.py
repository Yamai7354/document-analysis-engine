"""
CLI entry point: chat with the document analysis agent from the terminal,
without needing the FastAPI/Streamlit stack running. Useful for quick
testing of ingestion + the agent loop.

Usage:
    python main.py
    python main.py --ingest path/to/file.pdf
"""
import argparse
import uuid

from agent import run_agent


def ingest(path: str) -> None:
    from app.core.ingestion import ingest_document
    from app.core.chunking import chunk_documents
    from app.core.vectorstore import add_documents_to_store

    docs = ingest_document(path)
    chunks = chunk_documents(docs)
    add_documents_to_store(chunks)
    print(f"Ingested {path}: {len(chunks)} chunks added.")


def chat_loop() -> None:
    session_id = str(uuid.uuid4())
    print("Document Analysis Engine — CLI chat (type 'exit' to quit)")
    print(f"Session: {session_id}\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        result = run_agent(query, session_id=session_id)
        for step in result["tool_calls"]:
            print(f"  [tool: {step['tool']}] args={step['args']}")
        print(f"Agent: {result['answer']}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Document Analysis Engine CLI")
    parser.add_argument("--ingest", metavar="FILE", help="Ingest a document, then exit")
    args = parser.parse_args()

    if args.ingest:
        ingest(args.ingest)
        return

    chat_loop()


if __name__ == "__main__":
    main()
