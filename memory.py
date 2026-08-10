"""
Conversation memory: per-session chat history, persisted to SQLite so it
survives process restarts.
"""
import os
import sqlite3
from datetime import datetime, timezone
from typing import List, Dict

from config import CONFIG

DB_PATH = CONFIG.get("memory", {}).get("db_path", "models/agent_memory.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,   -- 'user' | 'assistant' | 'tool'
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
"""


class ConversationMemory:
    """SQLite-backed conversation history, scoped by session_id."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        conn.executescript(_SCHEMA)
        conn.commit()
        conn.close()

    def add_message(self, session_id: str, role: str, content: str) -> None:
        conn = self._connect()
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?)",
            (session_id, role, content, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        conn.close()

    def get_history(self, session_id: str, limit: int = 20) -> List[Dict[str, str]]:
        """Returns the last `limit` messages for a session, oldest first."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        conn.close()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def clear(self, session_id: str) -> None:
        conn = self._connect()
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()


_default_memory = None


def get_memory() -> ConversationMemory:
    global _default_memory
    if _default_memory is None:
        _default_memory = ConversationMemory()
    return _default_memory
