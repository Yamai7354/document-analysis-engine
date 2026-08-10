"""
Public demo entrypoint (deployed on Streamlit Community Cloud).

Differs from app/frontend/app.py in three ways, all needed for a demo that
strangers can use:
  1. No separate FastAPI backend — calls the agent/ingestion functions
     in-process, so there's only one service to deploy.
  2. Each visitor gets their own private Chroma collection, so uploads and
     chat history never leak between visitors.
  3. Hard caps on messages and uploads per visitor, so one visitor can't
     exhaust the shared free-tier LLM quota for everyone else.
"""
import os
import tempfile
import uuid

import streamlit as st

from app.core.ingestion import ingest_document
from app.core.chunking import chunk_documents
from app.core.vectorstore import add_documents_to_store, get_chunk_counts_by_source
from agent import run_agent
from tool_router import make_tools

MAX_TURNS_PER_SESSION = 15
MAX_FILES_PER_SESSION = 3
MAX_FILE_SIZE_MB = 5

st.set_page_config(page_title="Document Analysis Engine — Demo", page_icon="📚", layout="wide")


def _new_session():
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.collection_name = f"demo_{st.session_state.session_id}"
    st.session_state.messages = []
    st.session_state.turn_count = 0
    st.session_state.upload_count = 0


if "session_id" not in st.session_state:
    _new_session()

st.title("📚 Document Analysis Engine — Live Demo")
st.info(
    "This is a public demo of a RAG chatbot + agent built for AI freelance "
    "client work. Your session is private (nobody else sees your uploads or "
    "chat), but it's backed by a **free, rate-limited AI model**, so replies "
    "may be slower or less sharp than a paid model — and it's capped at "
    f"**{MAX_TURNS_PER_SESSION} messages** and **{MAX_FILES_PER_SESSION} files** "
    "per visitor to keep the demo available for everyone. "
    "[See the project / get in touch](mailto:Randy.johnson7354@outlook.com)."
)

with st.sidebar:
    st.header("📄 Upload a document")
    remaining_uploads = MAX_FILES_PER_SESSION - st.session_state.upload_count
    st.caption(f"{remaining_uploads} upload(s) left this session")

    uploaded_files = st.file_uploader(
        "PDF, TXT, or Markdown",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        disabled=remaining_uploads <= 0,
    )

    if st.button("Process documents", disabled=remaining_uploads <= 0):
        if not uploaded_files:
            st.warning("Please choose at least one file.")
        else:
            for file in uploaded_files:
                if st.session_state.upload_count >= MAX_FILES_PER_SESSION:
                    st.warning("Upload limit reached for this session.")
                    break
                size_mb = file.size / (1024 * 1024)
                if size_mb > MAX_FILE_SIZE_MB:
                    st.error(f"{file.name} is {size_mb:.1f}MB — max is {MAX_FILE_SIZE_MB}MB.")
                    continue
                with st.spinner(f"Processing {file.name}..."):
                    try:
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            path = os.path.join(tmp_dir, file.name)
                            with open(path, "wb") as f:
                                f.write(file.getbuffer())
                            docs = ingest_document(path)
                            chunks = chunk_documents(docs)
                            add_documents_to_store(
                                chunks, collection_name=st.session_state.collection_name
                            )
                        st.session_state.upload_count += 1
                        st.success(f"Processed {file.name}")
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {e}")

    st.divider()
    st.header("📚 Your knowledge base")
    counts = get_chunk_counts_by_source(collection_name=st.session_state.collection_name)
    if counts:
        for name, n in counts.items():
            st.caption(f"**{name}** — {n} chunks")
    else:
        st.caption("No documents uploaded yet.")

    if st.button("📊 Generate chart", disabled=not counts):
        tools, tools_by_name = make_tools(st.session_state.collection_name)
        message = tools_by_name["generate_chart"].invoke({})
        filename = message.replace("Chart saved as ", "")
        from tool_router import CHARTS_DIR
        chart_path = os.path.join(CHARTS_DIR, filename)
        if os.path.exists(chart_path):
            st.image(chart_path)

    st.divider()
    st.caption(f"{MAX_TURNS_PER_SESSION - st.session_state.turn_count} message(s) left")
    if st.button("🔄 Start a fresh session"):
        _new_session()
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("tool_calls"):
            with st.expander("Agent steps"):
                for step in message["tool_calls"]:
                    st.markdown(f"**{step['tool']}**`({step['args']})`")
                    st.code(step["result"], language=None)

if st.session_state.turn_count >= MAX_TURNS_PER_SESSION:
    st.warning(
        f"You've reached the {MAX_TURNS_PER_SESSION}-message limit for this demo "
        "session. Click **Start a fresh session** in the sidebar to continue."
    )
elif prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = run_agent(
                    prompt,
                    session_id=st.session_state.session_id,
                    collection_name=st.session_state.collection_name,
                )
                answer = result["answer"]
                sources = result.get("sources", [])
                tool_calls = result.get("tool_calls", [])

                st.markdown(answer)

                if sources:
                    with st.expander("View Sources"):
                        for i, source in enumerate(sources):
                            st.markdown(f"**Source {i + 1}:**\n{source}\n---")

                if tool_calls:
                    with st.expander("Agent steps"):
                        for step in tool_calls:
                            st.markdown(f"**{step['tool']}**`({step['args']})`")
                            st.code(step["result"], language=None)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "tool_calls": tool_calls}
                )
                st.session_state.turn_count += 1
                st.rerun()
            except Exception as e:
                st.error(f"Something went wrong: {e}")
