"""Skill store — dynamic skill accumulation from successful executions.

Learns reusable skills from successful tool executions:
  1. Monitor execution results
  2. Extract successful tool-use patterns
  3. Generate skill definitions (triggers + step sequences)
  4. Register as runtime-available tools
  5. Persist to SQLite for cross-session reuse

A skill is a reusable, parameterized tool sequence that has been
verified to work successfully for a specific type of task.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_SKILLS_DIR = "./agent_skills"


@dataclass
class SkillDefinition:
    """A reusable skill learned from successful executions."""

    name: str
    description: str
    trigger_patterns: list[str] = field(default_factory=list)
    tool_sequence: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.5
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    version: int = 1
    created_at: float = 0.0
    updated_at: float = 0.0
    skill_id: str = ""

    def __post_init__(self) -> None:
        if not self.skill_id:
            self.skill_id = hashlib.md5(self.name.encode("utf-8")).hexdigest()[:12]
        if not self.created_at:
            self.created_at = time.time()
        if not self.updated_at:
            self.updated_at = self.created_at

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / max(total, 1)

    @property
    def is_reliable(self) -> bool:
        return self.success_rate >= 0.6 and self.usage_count >= 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "trigger_patterns": self.trigger_patterns,
            "tool_sequence": self.tool_sequence,
            "confidence": self.confidence,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillDefinition:
        return cls(
            skill_id=data.get("skill_id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            trigger_patterns=data.get("trigger_patterns", []),
            tool_sequence=data.get("tool_sequence", []),
            confidence=data.get("confidence", 0.5),
            usage_count=data.get("usage_count", 0),
            success_count=data.get("success_count", 0),
            failure_count=data.get("failure_count", 0),
            version=data.get("version", 1),
            created_at=data.get("created_at", 0),
            updated_at=data.get("updated_at", 0),
        )


class SkillStore:
    """SQLite-backed persistent skill storage with thread safety."""

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            os.makedirs(_DEFAULT_SKILLS_DIR, exist_ok=True)
            db_path = os.path.join(_DEFAULT_SKILLS_DIR, "skills.db")
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self._cache: dict[str, SkillDefinition] = {}

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS skills (
                    skill_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    trigger_patterns TEXT,
                    tool_sequence TEXT,
                    confidence REAL DEFAULT 0.5,
                    usage_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    version INTEGER DEFAULT 1,
                    created_at REAL,
                    updated_at REAL
                )
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_skills_name ON skills(name)
            """)
            self._conn.commit()

    def save(self, skill: SkillDefinition) -> None:
        skill.updated_at = time.time()
        with self._lock:
            self._conn.execute("""
                INSERT OR REPLACE INTO skills
                (skill_id, name, description, trigger_patterns, tool_sequence,
                 confidence, usage_count, success_count, failure_count, version,
                 created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                skill.skill_id, skill.name, skill.description,
                json.dumps(skill.trigger_patterns, ensure_ascii=False),
                json.dumps(skill.tool_sequence, ensure_ascii=False),
                skill.confidence, skill.usage_count,
                skill.success_count, skill.failure_count,
                skill.version, skill.created_at, skill.updated_at,
            ))
            self._conn.commit()
        self._cache[skill.skill_id] = skill

    def get(self, skill_id: str) -> SkillDefinition | None:
        if skill_id in self._cache:
            return self._cache[skill_id]
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM skills WHERE skill_id = ?", (skill_id,)
            ).fetchone()
        if row is None:
            return None
        skill = self._row_to_skill(row)
        self._cache[skill_id] = skill
        return skill

    def get_by_name(self, name: str) -> SkillDefinition | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM skills WHERE name = ?", (name,)
            ).fetchone()
        if row is None:
            return None
        skill = self._row_to_skill(row)
        self._cache[skill.skill_id] = skill
        return skill

    def list_all(self) -> list[SkillDefinition]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM skills ORDER BY confidence DESC, usage_count DESC"
            ).fetchall()
        return [self._row_to_skill(r) for r in rows]

    def find_matching(self, query: str) -> list[SkillDefinition]:
        """Find skills matching a query via trigger patterns or name."""
        query_lower = query.lower()
        results: list[SkillDefinition] = []
        with self._lock:
            rows = self._conn.execute("SELECT * FROM skills").fetchall()
        for row in rows:
            skill = self._row_to_skill(row)
            if not skill.is_reliable:
                continue
            if query_lower in skill.name.lower():
                results.append(skill)
                continue
            for pattern in skill.trigger_patterns:
                if pattern.lower() in query_lower:
                    results.append(skill)
                    break
        results.sort(key=lambda s: s.confidence, reverse=True)
        return results

    def update_stats(
        self, skill_id: str, success: bool = True
    ) -> SkillDefinition | None:
        skill = self.get(skill_id)
        if skill is None:
            return None
        skill.usage_count += 1
        if success:
            skill.success_count += 1
        else:
            skill.failure_count += 1
        total = skill.success_count + skill.failure_count
        skill.confidence = skill.success_count / max(total, 1)
        skill.updated_at = time.time()
        self.save(skill)
        return skill

    def delete(self, skill_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM skills WHERE skill_id = ?", (skill_id,)
            )
            self._conn.commit()
            deleted = cursor.rowcount > 0
        self._cache.pop(skill_id, None)
        return deleted

    def cleanup_expired(self, min_usage: int = 1, min_confidence: float = 0.3) -> int:
        """Remove skills below quality thresholds."""
        removed = 0
        for skill in self.list_all():
            if skill.usage_count < min_usage and skill.confidence < min_confidence:
                self.delete(skill.skill_id)
                removed += 1
            elif skill.success_rate < 0.3 and skill.usage_count >= 3:
                self.delete(skill.skill_id)
                removed += 1
        return removed

    def _row_to_skill(self, row: sqlite3.Row) -> SkillDefinition:
        return SkillDefinition(
            skill_id=row["skill_id"],
            name=row["name"],
            description=row["description"] or "",
            trigger_patterns=json.loads(row["trigger_patterns"] or "[]"),
            tool_sequence=json.loads(row["tool_sequence"] or "[]"),
            confidence=row["confidence"],
            usage_count=row["usage_count"],
            success_count=row["success_count"],
            failure_count=row["failure_count"],
            version=row["version"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()
        self._cache.clear()

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            count = self._conn.execute("SELECT COUNT(*) FROM skills").fetchone()[0]
            reliable = self._conn.execute(
                "SELECT COUNT(*) FROM skills WHERE confidence >= 0.6 AND usage_count >= 2"
            ).fetchone()[0]
        return {"total": count, "reliable": reliable}