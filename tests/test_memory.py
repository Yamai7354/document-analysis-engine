import os

from memory import ConversationMemory


def _mem(tmp_path):
    return ConversationMemory(db_path=os.path.join(tmp_path, "test_memory.db"))


def test_add_and_get_history(tmp_path):
    mem = _mem(tmp_path)
    mem.add_message("session-1", "user", "hello")
    mem.add_message("session-1", "assistant", "hi there")
    mem.add_message("session-2", "user", "unrelated")

    history = mem.get_history("session-1")
    assert history == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_history_is_session_scoped(tmp_path):
    mem = _mem(tmp_path)
    mem.add_message("a", "user", "msg-a")
    mem.add_message("b", "user", "msg-b")

    assert len(mem.get_history("a")) == 1
    assert len(mem.get_history("b")) == 1


def test_clear(tmp_path):
    mem = _mem(tmp_path)
    mem.add_message("s", "user", "hi")
    mem.clear("s")
    assert mem.get_history("s") == []


def test_history_respects_limit_and_order(tmp_path):
    mem = _mem(tmp_path)
    for i in range(5):
        mem.add_message("s", "user", f"msg-{i}")

    history = mem.get_history("s", limit=2)
    assert [h["content"] for h in history] == ["msg-3", "msg-4"]
