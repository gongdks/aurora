"""Long-term memory — SQLite-backed persistence for cross-session recall.

Stores:
  - User facts/preferences as key-value pairs with category tags
  - Conversation summaries for session-level recall
  - Vector embeddings for semantic similarity search
  - Full-text search via LIKE queries as fallback

Vector store backends:
  - sqlite (default): SQLite BLOB storage, zero dependency
  - memory: fast in-memory storage (volatile)

Thread-safe: uses a single write lock with a shared SQLite connection.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import sqlite3
import struct
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any

from agent.config import settings
from agent.llm.embeddings import cosine_similarity, get_embedding_provider
from agent.memory.vector_store import BaseVectorStore, get_vector_store

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
    embedding BLOB,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS conversation_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_input TEXT NOT NULL,
    agent_response TEXT NOT NULL,
    summary TEXT DEFAULT '',
    embedding BLOB,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(key);
CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
CREATE INDEX IF NOT EXISTS idx_summaries_created ON conversation_summaries(created_at);
"""


class LongTermMemory:
    """SQLite-backed long-term memory with shared connection.

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
        self._conn = sqlite3.connect(
            self._db_path, check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._init_db()
        self._vector_store: BaseVectorStore | None = None
        self._search_cache: OrderedDict[str, tuple[list[dict[str, Any]], float]] = OrderedDict()
        self._search_cache_max: int = 128
        self._search_cache_ttl: float = 30.0

    @property
    def vs(self) -> BaseVectorStore | None:
        if self._vector_store is None:
            try:
                self._vector_store = get_vector_store()
            except Exception as exc:
                logger.warning("Vector store unavailable: %s", exc)
                self._vector_store = None
        return self._vector_store

    @contextmanager
    def _cursor(self):
        """Context manager yielding a cursor, auto-committing on success."""
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    def _init_db(self) -> None:
        """Create tables, indexes, and migrate schema if needed."""
        with self._lock:
            try:
                self._conn.executescript(_SCHEMA_SQL)
                self._conn.commit()
                self._migrate_schema()
            except sqlite3.Error as exc:
                logger.error("Failed to initialize long-term memory DB: %s", exc)

    def _migrate_schema(self) -> None:
        """Add missing columns to existing tables."""
        migrations = [
            ("memories", "embedding", "ALTER TABLE memories ADD COLUMN embedding BLOB"),
            ("conversation_summaries", "embedding", "ALTER TABLE conversation_summaries ADD COLUMN embedding BLOB"),
        ]
        for table, column, sql in migrations:
            try:
                cols = [row[1] for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()]
                if column not in cols:
                    self._conn.execute(sql)
                    self._conn.commit()
                    logger.info("Migrated: added %s column to %s table", column, table)
            except sqlite3.Error as exc:
                logger.warning("Migration for %s.%s failed: %s", table, column, exc)

    def close(self) -> None:
        """Close the shared connection."""
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass

    # ---- Key-Value memories ----

    @staticmethod
    def _encode_embedding(vec: list[float]) -> bytes:
        """Encode a float vector as a BLOB for SQLite storage."""
        return struct.pack(f"{len(vec)}f", *vec)

    @staticmethod
    def _decode_embedding(blob: bytes | None) -> list[float] | None:
        """Decode a BLOB back to a float vector."""
        if not blob:
            return None
        count = len(blob) // 4
        return list(struct.unpack(f"{count}f", blob))

    def remember(self, key: str, content: str, category: str = "general") -> None:
        """Store a key-value fact with embedding for semantic search."""
        now = time.time()
        embedding_blob = None
        vec: list[float] | None = None
        try:
            ep = get_embedding_provider()
            vec = ep.embed(f"{key} {content}")
            if vec:
                embedding_blob = self._encode_embedding(vec)
        except Exception as exc:
            logger.warning("Failed to generate embedding for '%s': %s", key, exc)

        vs = self.vs
        if vs and vec:
            vs.upsert(
                id=f"mem_{key}",
                vector=vec,
                metadata={"key": key, "content": content, "category": category, "type": "memory"},
            )

        with self._lock:
            try:
                with self._cursor() as cur:
                    cur.execute(
                        """INSERT INTO memories (key, content, category, embedding, created_at, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?)
                           ON CONFLICT(key) DO UPDATE SET
                           content=excluded.content, category=excluded.category,
                           embedding=COALESCE(excluded.embedding, memories.embedding),
                           updated_at=excluded.updated_at""",
                        (key, content, category, embedding_blob, now, now),
                    )
                self._invalidate_search_cache()
            except sqlite3.Error as exc:
                logger.error("Failed to store memory '%s': %s", key, exc)

    def recall(self, key: str) -> str | None:
        """Retrieve a specific fact by key."""
        with self._lock:
            try:
                with self._cursor() as cur:
                    row = cur.execute(
                        "SELECT content FROM memories WHERE key=?", (key,)
                    ).fetchone()
                    return row["content"] if row else None
            except sqlite3.Error as exc:
                logger.error("Failed to recall memory '%s': %s", key, exc)
                return None

    def search(self, query: str, category: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        """Search memories — tries semantic search first, falls back to LIKE."""
        if not query.strip():
            return []
        semantic_results = self.semantic_search(query, category=category, limit=limit)
        if semantic_results:
            return semantic_results
        return self._keyword_search(query, category=category, limit=limit)

    def keyword_search(self, query: str, category: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        """Search memories by keyword only (no embedding)."""
        return self._keyword_search(query, category=category, limit=limit)

    def semantic_search(self, query: str, category: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        """Search memories by semantic similarity (cosine distance).

        Uses vector store (SQLite BLOB or InMemory) for semantic search.
        Results are cached with short TTL for repeated queries.
        """
        cache_key = hashlib.md5(f"{query}:{category or ''}:{limit}".encode()).hexdigest()
        now = time.time()

        cached = self._search_cache.get(cache_key)
        if cached is not None:
            results, expire_at = cached
            if now < expire_at:
                self._search_cache.move_to_end(cache_key)
                return results
            del self._search_cache[cache_key]

        try:
            ep = get_embedding_provider()
            query_vec = ep.embed(query)
            if not query_vec:
                return []
        except Exception:
            return []

        vs = self.vs
        if vs:
            filters: dict[str, Any] | None = {"type": "memory"}
            if category:
                filters["category"] = category
            try:
                results = vs.search(query_vec, top_k=limit * 2, filters=filters)
                items: list[dict[str, Any]] = []
                for r in results:
                    meta = r["metadata"]
                    items.append({
                        "key": meta.get("key", r["id"]),
                        "content": meta.get("content", ""),
                        "category": meta.get("category", "general"),
                        "created_at": meta.get("created_at", 0),
                        "score": r["score"],
                    })
                items.sort(key=lambda x: x.get("score", 0), reverse=True)
                items = items[:limit]
                self._search_cache[cache_key] = (items, now + self._search_cache_ttl)
                self._search_cache.move_to_end(cache_key)
                self._evict_search_cache()
                return items
            except Exception as exc:
                logger.warning("Vector store search failed, falling back: %s", exc)

        items = self._semantic_search_sqlite(query_vec, category, limit)
        self._search_cache[cache_key] = (items, now + self._search_cache_ttl)
        self._search_cache.move_to_end(cache_key)
        self._evict_search_cache()
        return items

    def _evict_search_cache(self) -> None:
        """Evict expired and overflow entries from search cache."""
        now = time.time()
        expired = [k for k, (_, exp) in self._search_cache.items() if exp <= now]
        for k in expired:
            del self._search_cache[k]
        while len(self._search_cache) > self._search_cache_max:
            self._search_cache.popitem(last=False)

    def _invalidate_search_cache(self) -> None:
        """Clear all cached search results (called after writes)."""
        self._search_cache.clear()

    def _semantic_search_sqlite(self, query_vec: list[float], category: str | None, limit: int) -> list[dict[str, Any]]:
        """Fallback: scan all embeddings from SQLite BLOB and compute cosine similarity."""
        with self._lock:
            try:
                with self._cursor() as cur:
                    if category:
                        rows = cur.execute(
                            """SELECT key, content, category, embedding, created_at
                               FROM memories WHERE category=?""",
                            (category,),
                        ).fetchall()
                    else:
                        rows = cur.execute(
                            """SELECT key, content, category, embedding, created_at
                               FROM memories""",
                        ).fetchall()
            except sqlite3.Error as exc:
                logger.error("Failed to semantic search memories: %s", exc)
                return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            emb = self._decode_embedding(row["embedding"])
            if emb:
                sim = cosine_similarity(query_vec, emb)
                if sim > 0.05:
                    scored.append((sim, {
                        "key": row["key"],
                        "content": row["content"],
                        "category": row["category"],
                        "created_at": row["created_at"],
                        "score": sim,
                    }))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:limit]]

    def _keyword_search(self, query: str, category: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        """Fallback keyword search using LIKE."""
        with self._lock:
            try:
                with self._cursor() as cur:
                    like = f"%{query}%"
                    if category:
                        rows = cur.execute(
                            """SELECT key, content, category, created_at
                               FROM memories
                               WHERE category=? AND (key LIKE ? OR content LIKE ?)
                               ORDER BY updated_at DESC LIMIT ?""",
                            (category, like, like, limit),
                        ).fetchall()
                    else:
                        rows = cur.execute(
                            """SELECT key, content, category, created_at
                               FROM memories
                               WHERE key LIKE ? OR content LIKE ?
                               ORDER BY updated_at DESC LIMIT ?""",
                            (like, like, limit),
                        ).fetchall()
                    return [
                        {"key": r["key"], "content": r["content"],
                         "category": r["category"], "created_at": r["created_at"]}
                        for r in rows
                    ]
            except sqlite3.Error as exc:
                logger.error("Failed to keyword search memories: %s", exc)
                return []

    def list_all(self, category: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """List all stored facts, optionally filtered by category."""
        with self._lock:
            try:
                with self._cursor() as cur:
                    if category:
                        rows = cur.execute(
                            """SELECT key, content, category, created_at
                               FROM memories WHERE category=?
                               ORDER BY updated_at DESC LIMIT ?""",
                            (category, limit),
                        ).fetchall()
                    else:
                        rows = cur.execute(
                            """SELECT key, content, category, created_at
                               FROM memories ORDER BY updated_at DESC LIMIT ?""",
                            (limit,),
                        ).fetchall()
                    return [
                        {"key": r["key"], "content": r["content"],
                         "category": r["category"], "created_at": r["created_at"]}
                        for r in rows
                    ]
            except sqlite3.Error as exc:
                logger.error("Failed to list memories: %s", exc)
                return []

    def forget(self, key: str) -> bool:
        """Delete a specific memory. Returns True if something was deleted."""
        vs = self.vs
        if vs:
            try:
                vs.delete(f"mem_{key}")
            except Exception:
                pass
        with self._lock:
            try:
                with self._cursor() as cur:
                    cur.execute("DELETE FROM memories WHERE key=?", (key,))
                    return cur.rowcount > 0
            except sqlite3.Error as exc:
                logger.error("Failed to forget memory '%s': %s", key, exc)
                return False

    # ---- Conversation summaries ----

    def summarize_conversation(
        self, user_input: str, agent_response: str, summary: str = "",
    ) -> None:
        """Store a conversation summary with embedding for cross-session recall."""
        now = time.time()
        embedding_blob = None
        try:
            ep = get_embedding_provider()
            vec = ep.embed(f"{user_input} {summary} {agent_response[:200]}")
            if vec:
                embedding_blob = self._encode_embedding(vec)
        except Exception:
            pass

        with self._lock:
            try:
                with self._cursor() as cur:
                    cur.execute(
                        """INSERT INTO conversation_summaries
                           (user_input, agent_response, summary, embedding, created_at)
                           VALUES (?, ?, ?, ?, ?)""",
                        (user_input, agent_response, summary, embedding_blob, now),
                    )
                self._invalidate_search_cache()
            except sqlite3.Error as exc:
                logger.error("Failed to store conversation summary: %s", exc)

    def recent_summaries(self, limit: int = 10) -> list[dict[str, Any]]:
        """Retrieve recent conversation summaries."""
        with self._lock:
            try:
                with self._cursor() as cur:
                    rows = cur.execute(
                        """SELECT user_input, agent_response, summary, created_at
                           FROM conversation_summaries
                           ORDER BY created_at DESC LIMIT ?""",
                        (limit,),
                    ).fetchall()
                    return [
                        {"user_input": r["user_input"], "agent_response": r["agent_response"],
                         "summary": r["summary"], "created_at": r["created_at"]}
                        for r in rows
                    ]
            except sqlite3.Error as exc:
                logger.error("Failed to retrieve summaries: %s", exc)
                return []

    # ---- Maintenance ----

    def clear_all(self) -> str:
        """Clear all long-term memories. Returns status message."""
        vs = self.vs
        if vs:
            try:
                vs.clear()
            except Exception:
                pass
        with self._lock:
            try:
                with self._cursor() as cur:
                    mem_count = cur.execute(
                        "SELECT COUNT(*) FROM memories"
                    ).fetchone()[0]
                    sum_count = cur.execute(
                        "SELECT COUNT(*) FROM conversation_summaries"
                    ).fetchone()[0]
                    cur.execute("DELETE FROM memories")
                    cur.execute("DELETE FROM conversation_summaries")
                msg = f"已清除 {mem_count} 条记忆和 {sum_count} 条对话摘要"
                logger.info("Long-term memory cleared: %s", msg)
                return msg
            except sqlite3.Error as exc:
                logger.error("Failed to clear long-term memory: %s", exc)
                return f"清除失败: {exc}"

    @property
    def stats(self) -> dict[str, int]:
        """Return count of stored memories and summaries."""
        base: dict[str, int] = {"memories": 0, "summaries": 0}
        with self._lock:
            try:
                with self._cursor() as cur:
                    base["memories"] = cur.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                    base["summaries"] = cur.execute(
                        "SELECT COUNT(*) FROM conversation_summaries"
                    ).fetchone()[0]
            except sqlite3.Error:
                pass
        vs = self.vs
        if vs:
            try:
                base["vectors"] = vs.count()
            except Exception:
                base["vectors"] = 0
        return base