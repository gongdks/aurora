"""Agent event types — typed events flowing from agent to UI.

All events use Pydantic BaseModel for validation and serialization.
Every event has a `type` field that the UI dispatches on.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class BaseEvent(BaseModel):
    """Base class for all agent events."""
    model_config = ConfigDict(extra="allow")

    ts: float = Field(default_factory=time.time)


class LogEvent(BaseEvent):
    """Log / status message (thinking, progress, warnings)."""
    type: Literal["log"] = "log"
    message: str


class ToolEvent(BaseEvent):
    """Tool call event — fired when a tool starts or completes."""
    type: Literal["tool"] = "tool"
    name: str
    input: str = ""
    output: str | None = None
    call_id: int = 0


class StreamingTokenEvent(BaseEvent):
    """Single LLM token during streaming output."""
    type: Literal["streaming_token"] = "streaming_token"
    token: str


class DoneEvent(BaseEvent):
    """Agent completed successfully."""
    type: Literal["done"] = "done"
    answer: str


class ErrorEvent(BaseEvent):
    """Agent encountered an error."""
    type: Literal["error"] = "error"
    message: str


class PlanEvent(BaseEvent):
    """Plan step event — planning / re-planning progress."""
    type: Literal["plan"] = "plan"
    goal: str = ""
    steps: list[str] = []


class StatusEvent(BaseEvent):
    """Status change event — running / cancelling / etc."""
    type: Literal["status"] = "status"
    message: str


AgentEvent = LogEvent | ToolEvent | StreamingTokenEvent | DoneEvent | ErrorEvent | PlanEvent | StatusEvent


def event_from_dict(data: dict[str, Any]) -> AgentEvent:
    """Reconstruct a typed event from a raw dict (for deserialization)."""
    t = data.get("type", "")
    if t == "log":
        return LogEvent(**data)
    if t == "tool":
        return ToolEvent(**data)
    if t == "streaming_token":
        return StreamingTokenEvent(**data)
    if t == "done":
        return DoneEvent(**data)
    if t == "error":
        return ErrorEvent(**data)
    if t == "plan":
        return PlanEvent(**data)
    if t == "status":
        return StatusEvent(**data)
    return BaseEvent(**data)