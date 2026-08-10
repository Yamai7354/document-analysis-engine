# Document Analysis Engine

A chatbot-driven knowledge base: upload documents, ask questions about them,
and get grounded answers with cited sources — plus an agent layer that can
look up document stats and generate charts on request.

## Architecture

- **Ingestion → Chunking → Embedding → Vector Store**: PDF/TXT/MD files are
  loaded, split into overlapping chunks, embedded locally with a
  HuggingFace sentence-transformer, and stored in a Chroma vector DB
  (`app/core/ingestion.py`, `chunking.py`, `embeddings.py`, `vectorstore.py`).
- **Agent loop** (`agent.py`): on each chat turn, loads conversation history
  from SQLite, lets the LLM decide whether to call tools, executes any
  tool calls, and returns a grounded final answer.
- **Tools** (`tool_router.py`): `search_knowledge_base` (RAG search over
  uploaded docs), `list_documents`, `get_document_stats`, and
  `generate_chart` (chunks-per-document bar chart, saved to `charts/`).
- **Planner** (`planner.py`): wraps the LLM with OpenAI-style function
  calling so it can choose which tool(s) to call.
- **Memory** (`memory.py`): SQLite-backed conversation history, scoped per
  session, so the agent remembers earlier turns in a conversation.
- **API** (`app/api/main.py`): FastAPI backend — `/upload`, `/chat`,
  `/documents`, `/charts/generate`, `/charts/{filename}`.
- **Frontend** (`app/frontend/app.py`): Streamlit chat UI with document
  upload, a live knowledge-base panel, chart generation, and an "agent
  steps" view showing which tools were called for transparency.
- **CLI** (`main.py`): terminal chat loop for quick testing without the
  frontend — `python main.py`, or `python main.py --ingest file.pdf`.

The LLM provider is swappable via `config.yaml` (`llm.provider:
openai|anthropic` + `llm.model`) — both `llm.py` and the underlying
LangChain integrations are already wired for either.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (The `venv/` folder in this repo is a stale artifact from another
   machine — ignore it, or delete and recreate it locally.)

2. Copy `.env.example` to `.env` and add an API key for whichever provider
   `config.yaml` is set to (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`).

3. Run the backend:
   ```bash
   uvicorn app.api.main:app --reload
   ```

4. In another terminal, run the frontend:
   ```bash
   streamlit run app/frontend/app.py
   ```

5. Open the Streamlit URL, upload a document, and start asking questions.

## Testing

```bash
pytest tests/ -v
```

Most of the pipeline (ingestion, chunking, embedding, vector search, chart
generation) is tested without any LLM calls. The one test that exercises
the full agent loop (`test_chat_requires_working_llm`) needs a funded API
key for the configured provider — it skips cleanly instead of failing if
that provider has no credits.
