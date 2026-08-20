"""Vector store abstraction — SQLite BLOB + InMemory.

Provides a unified interface for vector operations (upsert, semantic search,
delete, clear) with two backends:

  - SQLiteVectorStore: persistent SQLite BLOB storage, zero-dependency
  - InMemoryVectorStore: fast in-memory storage (lost on restart)

Usage:
    from agent.memory.vector_store import get_vector_store

    vs = get_vector_store()
    vs.upsert("mem_1", vec, metadata={"key": "user_name", "category": "profile"})
    results = vs.search(query_vec, top_k=5)
"""

from __future__ import annotations

import json
import logging
import math
import os
import struct
import sqlite3
import threading
import time
from typing import Any

from agent.config import settings

logger = logging.getLogger(__name__)

_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "agent_vector.db",
)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS vectors (
    id TEXT PRIMARY KEY,
    vector BLOB NOT NULL,
    metadata_json TEXT DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_vectors_metadata ON vectors(metadata_json);
"""


class BaseVectorStore:
    """Abstract interface for vector stores."""

    def upsert(
        self, id: str, vector: list[float], metadata: dict[str, Any] | None = None
    ) -> None:
        raise NotImplementedError

    def search(
        self, query_vector: list[float], top_k: int = 10, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    def delete(self, id: str) -> None:
        raise NotImplementedError

    def clear(self) -> int:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def close(self) -> None:
        pass


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_b = math.sqrt(sum(x * x for x in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryVectorStore(BaseVectorStore):
    """Fast in-memory vector store — lost on restart, but ultra-fast.

    Suitable for hot-path lookups and temporary caches.
    Thread-safe via a read-write lock pattern.
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}
        self._vectors: dict[str, list[float]] = {}
        self._lock = threading.RLock()

    def upsert(
        self, id: str, vector: list[float], metadata: dict[str, Any] | None = None
    ) -> None:
        now = time.time()
        with self._lock:
            self._vectors[id] = list(vector)
            self._store[id] = {
                "metadata": metadata or {},
                "created_at": self._store.get(id, {}).get("created_at", now),
                "updated_at": now,
            }

    def search(
        self, query_vector: list[float], top_k: int = 10, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        with self._lock:
            items: list[tuple[float, dict[str, Any]]] = []
            for vid, vec in self._vectors.items():
                if len(vec) != len(query_vector):
                    continue
                meta = self._store[vid]["metadata"]
                if filters and not all(meta.get(k) == v for k, v in filters.items()):
                    continue
                sim = _cosine_similarity(query_vector, vec)
                if sim > 0.01:
                    items.append((sim, {"id": vid, "score": sim, "metadata": meta}))

        items.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in items[:top_k]]

    def delete(self, id: str) -> None:
        with self._lock:
            self._vectors.pop(id, None)
            self._store.pop(id, None)

    def clear(self) -> int:
        with self._lock:
            n = len(self._vectors)
            self._vectors.clear()
            self._store.clear()
            return n

    def count(self) -> int:
        with self._lock:
            return len(self._vectors)


class SQLiteVectorStore(BaseVectorStore):
    """SQLite + BLOB vector store — persistent, zero-dependency.

    Suitable for datasets up to ~10k vectors. Cosine similarity computed
    in Python. Thread-safe with a single write lock.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or _DB_PATH
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            self._conn.executescript(_SCHEMA_SQL)
            self._conn.commit()

    @staticmethod
    def _encode(vec: list[float]) -> bytes:
        return struct.pack(f"{len(vec)}f", *vec)

    @staticmethod
    def _decode(blob: bytes) -> list[float]:
        count = len(blob) // 4
        return list(struct.unpack(f"{count}f", blob))

    def upsert(
        self, id: str, vector: list[float], metadata: dict[str, Any] | None = None
    ) -> None:
        now = time.time()
        vec_blob = self._encode(vector)
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        with self._lock:
            try:
                self._conn.execute(
                    """INSERT INTO vectors (id, vector, metadata_json, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(id) DO UPDATE SET
                       vector=excluded.vector,
                       metadata_json=excluded.metadata_json,
                       updated_at=excluded.updated_at""",
                    (id, vec_blob, meta_json, now, now),
                )
                self._conn.commit()
            except sqlite3.Error as exc:
                logger.error("Failed to upsert vector '%s': %s", id, exc)

    def search(
        self, query_vector: list[float], top_k: int = 10, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT id, vector, metadata_json FROM vectors"
                ).fetchall()
        except sqlite3.Error as exc:
            logger.error("Failed to search vectors: %s", exc)
            return []

        qvec = query_vector
        norm_q = math.sqrt(sum(x * x for x in qvec))
        if norm_q == 0:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            vec = self._decode(row["vector"])
            if len(vec) != len(qvec):
                continue
            sim = _cosine_similarity(qvec, vec)
            if sim > 0.01:
                meta = json.loads(row["metadata_json"] or "{}")
                if filters:
                    if not all(meta.get(k) == v for k, v in filters.items()):
                        continue
                scored.append((sim, {"id": row["id"], "score": sim, "metadata": meta}))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:top_k]]

    def delete(self, id: str) -> None:
        with self._lock:
            try:
                self._conn.execute("DELETE FROM vectors WHERE id=?", (id,))
                self._conn.commit()
            except sqlite3.Error as exc:
                logger.error("Failed to delete vector '%s': %s", id, exc)

    def clear(self) -> int:
        with self._lock:
            try:
                count = self._conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
                self._conn.execute("DELETE FROM vectors")
                self._conn.commit()
                return count
            except sqlite3.Error:
                return 0

    def count(self) -> int:
        with self._lock:
            try:
                return self._conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
            except sqlite3.Error:
                return 0

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass


_vector_store: BaseVectorStore | None = None


def get_vector_store() -> BaseVectorStore:
    """Get or create the singleton vector store.

    Uses SQLite BLOB for persistent storage (default) or InMemory for
    fast-but-volatile storage, controlled by config.VECTOR_STORE.
    """
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    backend = settings.VECTOR_STORE.strip().lower()

    if backend == "memory":
        _vector_store = InMemoryVectorStore()
        logger.info("Vector store: InMemory (fast, volatile)")
    else:
        _vector_store = SQLiteVectorStore()
        logger.info("Vector store: SQLite BLOB (persistent, zero-dependency)")

    return _vector_store


def reset_vector_store() -> None:
    """Force recreation of the vector store (e.g. after config change)."""
    global _vector_store
    if _vector_store is not None:
        _vector_store.close()
    _vector_store = None


def get_storage_info() -> dict[str, Any]:
    """Get current storage architecture information.

    Shows the layered storage model:
      - SQLite: always-on structured data (facts, skills, configs)
      - Vector: selected vector search backend (SQLite BLOB or InMemory)
      - Cache: in-memory LRU caches
    """
    vs = _vector_store
    backend_type = "none"
    backend_detail = ""
    if vs is not None:
        backend_type = "in_memory" if isinstance(vs, InMemoryVectorStore) else "sqlite_blob"
        backend_detail = f" (count={vs.count()})"

    long_term_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "agent_long_term.db",
    )
    skill_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "agent_skills", "skills.db",
    )

    return {
        "structured_storage": {
            "type": "SQLite",
            "databases": [
                {"path": long_term_path, "purpose": "facts, preferences, conversation summaries"},
                {"path": skill_path, "purpose": "learned skills, tool sequences"},
            ],
        },
        "vector_storage": {
            "type": backend_type,
            "detail": backend_detail.strip(),
            "config": settings.VECTOR_STORE,
        },
        "cache_layers": {
            "llm_response": "LRU+TTL (global LLMCache)",
            "embedding": "LRU+TTL (per-provider cache)",
            "search_results": "LRU+TTL (LongTermMemory._search_cache)",
            "tool_descriptions": "GraphOrchestrator._tools_desc",
        },
        "architecture": "layered",
    }