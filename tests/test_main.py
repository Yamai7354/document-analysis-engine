import pytest
from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "message": "Document Analysis Engine API is running",
    }


def test_documents_endpoint_responds():
    response = client.get("/documents")
    assert response.status_code == 200
    assert "documents" in response.json()


def test_chat_requires_working_llm():
    """
    Exercises the full agent loop end to end. Requires a funded API key for
    whichever provider is set in config.yaml (llm.provider) — skips rather
    than failing if that provider has no credits, so a billing issue
    doesn't read as a code bug in CI.
    """
    response = client.post("/chat", json={"query": "hello"})
    no_llm_markers = ("credit", "not found", "401", "403", "api key", "api_key")
    if response.status_code == 500 and any(
        m in response.text.lower() for m in no_llm_markers
    ):
        pytest.skip("Configured LLM provider has no usable credentials for this run")
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "tool_calls" in data
