"""Long-term memory — SQLite-backed persistence for cross-session recall.

Stores:
  - User facts/preferences as key-value pairs with category tags
  - Conversation summaries for session-level recall
  - Full-text search via simple LIKE queries (no embedding dependency)

Thread-safe: uses a single write lock.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "agent_long_term.db",
)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT UNIQUE NOT NULL,
    content TEXT NOT NULL,
    category TEXT DEFAULT 'general',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS conversation_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_input TEXT NOT NULL,
    agent_response TEXT NOT NULL,
    summary TEXT DEFAULT '',
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(key);
CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
CREATE INDEX IF NOT EXISTS idx_summaries_created ON conversation_summaries(created_at);
"""


class LongTermMemory:
    """SQLite-backed long-term memory.

    Usage:
        ltm = LongTermMemory()
        ltm.remember("user_name", "Alice", category="user_profile")
        facts = ltm.recall("user_name")
        ltm.summarize_conversation("Hello", "Hi there!", "Greeting exchange")
        history = ltm.recent_summaries(limit=10)
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or _DB_PATH
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Create tables and indexes if they don't exist."""
        with self._lock:
            try:
                conn = sqlite3.connect(self._db_path)
                conn.executescript(_SCHEMA_SQL)
                conn.commit()
                conn.close()
            except sqlite3.Error as exc:
                logger.error("Failed to initialize long-term memory DB: %s", exc)

    # ---- Key-Value memories ----

    def remember(self, key: str, content: str, category: str = "general") -> None:
        """Store a key-value fact, overwriting if key already exists."""
        now = time.time()
        with self._lock:
            try:
                conn = sqlite3.connect(self._db_path)
                conn.execute(
                    """INSERT INTO memories (key, content, category, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(key) DO UPDATE SET
                       content=excluded.content, category=excluded.category,
                       updated_at=excluded.updated_at""",
                    (key, content, category, now, now),
                )
                conn.commit()
                conn.close()
            except sqlite3.Error as exc:
                logger.error("Failed to store memory '%s': %s", key, exc)

    def recall(self, key: str) -> str | None:
        """Retrieve a specific fact by key."""
        with self._lock:
            try:
                conn = sqlite3.connect(self._db_path)
                row = conn.execute(
                    "SELECT content FROM memories WHERE key=?", (key,)
                ).fetchone()
                conn.close()
                return row[0] if row else None
            except sqlite3.Error as exc:
                logger.error("Failed to recall memory '%s': %s", key, exc)
                return None

    def search(self, query: str, category: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        """Search memories by keyword (LIKE match on key and content)."""
        with self._lock:
            try:
                conn = sqlite3.connect(self._db_path)
                like = f"%{query}%"
                if category:
                    rows = conn.execute(
                        """SELECT key, content, category, created_at
                           FROM memories
                           WHERE category=? AND (key LIKE ? OR content LIKE ?)
                           ORDER BY updated_at DESC LIMIT ?""",
                        (category, like, like, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """SELECT key, content, category, created_at
                           FROM memories
                           WHERE key LIKE ? OR content LIKE ?
                           ORDER BY updated_at DESC LIMIT ?""",
                        (like, like, limit),
                    ).fetchall()
                conn.close()
                return [
                    {"key": r[0], "content": r[1], "category": r[2], "created_at": r[3]}
                    for r in rows
                ]
            except sqlite3.Error as exc:
                logger.error("Failed to search memories: %s", exc)
                return []

    def list_all(self, category: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """List all stored facts, optionally filtered by category."""
        with self._lock:
            try:
                conn = sqlite3.connect(self._db_path)
                if category:
                    rows = conn.execute(
                        """SELECT key, content, category, created_at
                           FROM memories WHERE category=?
                           ORDER BY updated_at DESC LIMIT ?""",
                        (category, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """SELECT key, content, category, created_at
                           FROM memories ORDER BY updated_at DESC LIMIT ?""",
                        (limit,),
                    ).fetchall()
                conn.close()
                return [
                    {"key": r[0], "content": r[1], "category": r[2], "created_at": r[3]}
                    for r in rows
                ]
            except sqlite3.Error as exc:
                logger.error("Failed to list memories: %s", exc)
                return []

    def forget(self, key: str) -> bool:
        """Delete a specific memory. Returns True if something was deleted."""
        with self._lock:
            try:
                conn = sqlite3.connect(self._db_path)
                cur = conn.execute("DELETE FROM memories WHERE key=?", (key,))
                deleted = cur.rowcount > 0
                conn.commit()
                conn.close()
                return deleted
            except sqlite3.Error as exc:
                logger.error("Failed to forget memory '%s': %s", key, exc)
                return False

    # ---- Conversation summaries ----

    def summarize_conversation(
        self, user_input: str, agent_response: str, summary: str = "",
    ) -> None:
        """Store a conversation summary for cross-session recall."""
        now = time.time()
        with self._lock:
            try:
                conn = sqlite3.connect(self._db_path)
                conn.execute(
                    """INSERT INTO conversation_summaries
                       (user_input, agent_response, summary, created_at)
                       VALUES (?, ?, ?, ?)""",
                    (user_input, agent_response, summary, now),
                )
                conn.commit()
                conn.close()
            except sqlite3.Error as exc:
                logger.error("Failed to store conversation summary: %s", exc)

    def recent_summaries(self, limit: int = 10) -> list[dict[str, Any]]:
        """Retrieve recent conversation summaries."""
        with self._lock:
            try:
                conn = sqlite3.connect(self._db_path)
                rows = conn.execute(
                    """SELECT user_input, agent_response, summary, created_at
                       FROM conversation_summaries
                       ORDER BY created_at DESC LIMIT ?""",
                    (limit,),
                ).fetchall()
                conn.close()
                return [
                    {"user_input": r[0], "agent_response": r[1],
                     "summary": r[2], "created_at": r[3]}
                    for r in rows
                ]
            except sqlite3.Error as exc:
                logger.error("Failed to retrieve summaries: %s", exc)
                return []

    # ---- Maintenance ----

    def clear_all(self) -> str:
        """Clear all long-term memories. Returns status message."""
        with self._lock:
            try:
                conn = sqlite3.connect(self._db_path)
                mem_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                sum_count = conn.execute(
                    "SELECT COUNT(*) FROM conversation_summaries"
                ).fetchone()[0]
                conn.execute("DELETE FROM memories")
                conn.execute("DELETE FROM conversation_summaries")
                conn.commit()
                conn.close()
                msg = f"已清除 {mem_count} 条记忆和 {sum_count} 条对话摘要"
                logger.info("Long-term memory cleared: %s", msg)
                return msg
            except sqlite3.Error as exc:
                logger.error("Failed to clear long-term memory: %s", exc)
                return f"清除失败: {exc}"

    @property
    def stats(self) -> dict[str, int]:
        """Return count of stored memories and summaries."""
        with self._lock:
            try:
                conn = sqlite3.connect(self._db_path)
                mem = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                sums = conn.execute(
                    "SELECT COUNT(*) FROM conversation_summaries"
                ).fetchone()[0]
                conn.close()
                return {"memories": mem, "summaries": sums}
            except sqlite3.Error:
                return {"memories": 0, "summaries": 0}
