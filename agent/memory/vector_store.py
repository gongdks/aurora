"""Vector store abstraction — SQLite BLOB or ChromaDB.

Provides a unified interface for vector operations (upsert, semantic search,
delete, clear). Default backend is SQLite BLOB (zero-dependency), with Chroma
as a drop-in upgrade for larger-scale semantic search.

Usage:
    from agent.memory.vector_store import get_vector_store

    vs = get_vector_store()  # auto-selects backend based on config
    vs.upsert("mem_1", vec, metadata={"key": "user_name", "category": "profile"})
    results = vs.search(query_vec, top_k=5)
"""

from __future__ import annotations

import json
import logging
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


class SQLiteVectorStore(BaseVectorStore):
    """SQLite + BLOB vector store — zero-dependency fallback.

    Suitable for small-to-medium datasets (up to ~10k vectors).
    Semantic search scans all vectors and computes cosine similarity in Python.
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
            rows = self._conn.execute(
                "SELECT id, vector, metadata_json FROM vectors"
            ).fetchall()
        except sqlite3.Error as exc:
            logger.error("Failed to search vectors: %s", exc)
            return []

        import math

        qvec = query_vector
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            vec = self._decode(row["vector"])
            if len(vec) != len(qvec):
                continue
            dot = sum(x * y for x, y in zip(qvec, vec))
            norm_q = math.sqrt(sum(x * x for x in qvec))
            norm_v = math.sqrt(sum(x * x for x in vec))
            if norm_q == 0 or norm_v == 0:
                continue
            sim = dot / (norm_q * norm_v)
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


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB vector store — professional-grade vector search.

    Requires `chromadb` package: pip install chromadb
    Supports metadata filtering, HNSW indexing, and persistent storage.
    """

    def __init__(self, persist_dir: str | None = None) -> None:
        self._persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self._client = None
        self._collection = None
        self._init_client()

    def _init_client(self) -> None:
        try:
            import chromadb
            from chromadb.config import Settings

            self._client = chromadb.PersistentClient(
                path=self._persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name="agent_memories",
                metadata={"hnsw:space": "cosine"},
            )
            count = self._collection.count()
            logger.info("[Chroma] Connected | collection=agent_memories | count=%d", count)
        except ImportError:
            logger.warning("chromadb not installed. Run: pip install chromadb")
            raise
        except Exception as exc:
            logger.error("Failed to initialize ChromaDB: %s", exc)
            raise

    def upsert(
        self, id: str, vector: list[float], metadata: dict[str, Any] | None = None
    ) -> None:
        try:
            self._collection.upsert(
                ids=[id],
                embeddings=[vector],
                metadatas=[metadata or {}],
            )
        except Exception as exc:
            logger.error("Chroma upsert failed for '%s': %s", id, exc)

    def search(
        self, query_vector: list[float], top_k: int = 10, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        try:
            kwargs: dict[str, Any] = {
                "query_embeddings": [query_vector],
                "n_results": min(top_k, self._collection.count() or 1),
                "include": ["metadatas", "distances"],
            }
            if filters:
                kwargs["where"] = filters
            results = self._collection.query(**kwargs)

            items: list[dict[str, Any]] = []
            if results and results["ids"]:
                for i, rid in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i] if results["distances"] else 0.0
                    score = 1.0 - distance if distance else 0.0
                    items.append({
                        "id": rid,
                        "score": score,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    })
            return items
        except Exception as exc:
            logger.error("Chroma search failed: %s", exc)
            return []

    def delete(self, id: str) -> None:
        try:
            self._collection.delete(ids=[id])
        except Exception as exc:
            logger.error("Chroma delete failed for '%s': %s", id, exc)

    def clear(self) -> int:
        try:
            count = self._collection.count()
            self._client.delete_collection("agent_memories")
            self._collection = self._client.get_or_create_collection(
                name="agent_memories",
                metadata={"hnsw:space": "cosine"},
            )
            return count
        except Exception:
            return 0

    def count(self) -> int:
        try:
            return self._collection.count()
        except Exception:
            return 0

    def close(self) -> None:
        pass


_vector_store: BaseVectorStore | None = None


def get_vector_store() -> BaseVectorStore:
    """Get or create the singleton vector store based on config."""
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    backend = settings.VECTOR_STORE.strip().lower()
    if backend == "chroma":
        try:
            _vector_store = ChromaVectorStore()
            logger.info("Vector store backend: ChromaDB")
            return _vector_store
        except Exception as exc:
            logger.warning("ChromaDB init failed (%s), falling back to SQLite", exc)

    _vector_store = SQLiteVectorStore()
    logger.info("Vector store backend: SQLite BLOB")
    return _vector_store


def reset_vector_store() -> None:
    """Force recreation of the vector store (e.g. after config change)."""
    global _vector_store
    if _vector_store is not None:
        _vector_store.close()
    _vector_store = None