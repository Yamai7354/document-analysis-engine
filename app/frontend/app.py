import os
import uuid

import requests
import streamlit as st

API_URL = os.environ.get("DOC_ENGINE_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Document Analysis Engine", page_icon="📚", layout="wide")

st.title("📚 Document Analysis Engine")
st.markdown("Upload documents and ask questions about them!")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar: document upload + knowledge base overview
with st.sidebar:
    st.header("📄 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or Markdown files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if st.button("Process Documents"):
        if uploaded_files:
            for file in uploaded_files:
                with st.spinner(f"Processing {file.name}..."):
                    files = {"file": (file.name, file, file.type)}
                    try:
                        response = requests.post(f"{API_URL}/upload", files=files)
                        if response.status_code == 200:
                            st.success(f"Successfully processed {file.name}")
                        else:
                            st.error(f"Error processing {file.name}: {response.text}")
                    except Exception as e:
                        st.error(f"Connection error: {e}")
        else:
            st.warning("Please upload at least one file.")

    st.divider()
    st.header("📚 Knowledge Base")

    if st.button("Refresh"):
        st.rerun()

    try:
        docs_response = requests.get(f"{API_URL}/documents", timeout=5)
        if docs_response.status_code == 200:
            documents = docs_response.json().get("documents", [])
            if documents:
                for doc in documents:
                    st.caption(f"**{doc['name']}** — {doc['chunks']} chunks")
            else:
                st.caption("No documents uploaded yet.")
        else:
            st.caption("Could not load document list.")
    except Exception:
        st.caption("Backend not reachable — is the API running?")

    if st.button("📊 Generate chart"):
        try:
            chart_response = requests.post(f"{API_URL}/charts/generate", timeout=30)
            if chart_response.status_code == 200:
                filename = chart_response.json().get("message", "")
                st.session_state["last_chart"] = filename.replace(
                    "Chart saved as ", ""
                )
            else:
                st.error(f"Error generating chart: {chart_response.text}")
        except Exception as e:
            st.error(f"Connection error: {e}")

    if st.session_state.get("last_chart"):
        try:
            img_response = requests.get(
                f"{API_URL}/charts/{st.session_state['last_chart']}", timeout=10
            )
            if img_response.status_code == 200:
                st.image(img_response.content)
        except Exception:
            pass

    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("tool_calls"):
            with st.expander("Agent steps"):
                for step in message["tool_calls"]:
                    st.markdown(f"**{step['tool']}**`({step['args']})`")
                    st.code(step["result"], language=None)

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={"query": prompt, "session_id": st.session_state.session_id},
                    timeout=60,
                )
                if response.status_code == 200:
                    result = response.json()
                    answer = result.get("answer", "No answer provided.")
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
                        {
                            "role": "assistant",
                            "content": answer,
                            "tool_calls": tool_calls,
                        }
                    )
                else:
                    st.error(f"Error connecting to API: {response.text}")
            except Exception as e:
                st.error(f"Error connecting to API: {e}. Is the backend running?")
