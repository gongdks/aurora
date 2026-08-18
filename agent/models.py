"""Agent data models — result types and status enums."""

from __future__ import annotations

from enum import Enum
from typing import Any


class AgentStatus(str, Enum):
    """Final status of an agent invocation."""

    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"
    RUNNING = "running"


class AgentResult:
    """Result of a single agent invocation."""

    def __init__(
        self,
        content: str = "",
        status: AgentStatus = AgentStatus.COMPLETED,
        elapsed: float = 0.0,
        tool_calls: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.content = content
        self.status = status
        self.elapsed = elapsed
        self.tool_calls = tool_calls or []
        self.metadata = metadata or {}

    @property
    def is_ok(self) -> bool:
        return self.status in (AgentStatus.COMPLETED, AgentStatus.PARTIAL)

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "status": self.status.value,
            "elapsed": self.elapsed,
            "tool_calls": self.tool_calls,
            "metadata": self.metadata,
        }