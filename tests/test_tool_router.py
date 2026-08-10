from langchain_core.documents import Document

import tool_router
from tool_router import TOOLS_BY_NAME, make_tools


def test_format_search_results_empty():
    assert "No relevant" in tool_router.format_search_results([])


def test_format_search_results_with_docs():
    docs = [Document(page_content="hello world", metadata={"source": "data/foo.txt"})]
    result = tool_router.format_search_results(docs)
    assert "foo.txt" in result
    assert "hello world" in result


def test_list_documents_empty(monkeypatch):
    monkeypatch.setattr(tool_router, "get_chunk_counts_by_source", lambda **kw: {})
    assert "No documents" in TOOLS_BY_NAME["list_documents"].invoke({})


def test_list_documents_with_data(monkeypatch):
    monkeypatch.setattr(
        tool_router, "get_chunk_counts_by_source",
        lambda **kw: {"a.txt": 3, "b.pdf": 5},
    )
    result = TOOLS_BY_NAME["list_documents"].invoke({})
    assert "a.txt: 3 chunks" in result
    assert "b.pdf: 5 chunks" in result


def test_get_document_stats(monkeypatch):
    monkeypatch.setattr(
        tool_router, "get_chunk_counts_by_source",
        lambda **kw: {"a.txt": 3, "b.pdf": 5},
    )
    result = TOOLS_BY_NAME["get_document_stats"].invoke({})
    assert "2 document(s)" in result
    assert "8 chunk(s)" in result


def test_generate_chart_no_documents(monkeypatch):
    monkeypatch.setattr(tool_router, "get_chunk_counts_by_source", lambda **kw: {})
    result = TOOLS_BY_NAME["generate_chart"].invoke({})
    assert "No documents to chart" in result


def test_generate_chart_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        tool_router, "get_chunk_counts_by_source",
        lambda **kw: {"a.txt": 3, "b.pdf": 5},
    )
    monkeypatch.setattr(tool_router, "CHARTS_DIR", str(tmp_path))
    result = TOOLS_BY_NAME["generate_chart"].invoke({})
    assert "chunks_per_document" in result
    assert list(tmp_path.glob("chunks_per_document*.png"))


def test_make_tools_scopes_calls_to_their_collection(monkeypatch):
    """
    Regression test for visitor isolation in the public demo: two tool sets
    built for different collection_names must query their own collection,
    not leak into each other.
    """
    seen = []

    def fake(**kwargs):
        seen.append(kwargs.get("collection_name"))
        return {}

    monkeypatch.setattr(tool_router, "get_chunk_counts_by_source", fake)

    _, by_name_a = make_tools("session-a")
    _, by_name_b = make_tools("session-b")
    by_name_a["list_documents"].invoke({})
    by_name_b["list_documents"].invoke({})

    assert seen == ["session-a", "session-b"]
