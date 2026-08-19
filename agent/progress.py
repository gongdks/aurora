"""Progress event factory — creates typed events for the UI.

Uses Pydantic models from agent.events for type safety.
All functions return dicts for backward compatibility with signal/slot.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from agent.events import (
    AgentEvent,
    DoneEvent,
    ErrorEvent,
    LogEvent,
    PlanEvent,
    StatusEvent,
    StreamingTokenEvent,
    ToolEvent,
)


def make_log(message: str) -> dict[str, Any]:
    """Create a log message event."""
    return LogEvent(message=message).model_dump()


def make_tool(
    name: str,
    input_str: str,
    output: str | None = None,
    *,
    call_id: int | None = None,
) -> dict[str, Any]:
    """Create a tool call event."""
    data: dict[str, Any] = {
        "type": "tool",
        "name": name,
        "input": input_str[:500],
        "ts": time.time(),
    }
    if output is not None:
        data["output"] = output[:1000]
    if call_id is not None:
        data["call_id"] = call_id
    return ToolEvent(**data).model_dump()


def make_streaming_token(token: str) -> dict[str, Any]:
    """Create a streaming token event."""
    return StreamingTokenEvent(token=token).model_dump()


def make_done(answer: str) -> dict[str, Any]:
    """Create a completion event."""
    return DoneEvent(answer=answer).model_dump()


def make_error(message: str) -> dict[str, Any]:
    """Create an error event."""
    return ErrorEvent(message=message).model_dump()


def make_plan(goal: str = "", steps: list[str] | None = None) -> dict[str, Any]:
    """Create a plan progress event."""
    return PlanEvent(goal=goal, steps=steps or []).model_dump()


def make_status(message: str) -> dict[str, Any]:
    """Create a status change event."""
    return StatusEvent(message=message).model_dump()


def safe_done(
    callback: Callable[[dict[str, Any]], None] | None,
    answer: str,
) -> None:
    """Safely emit a 'done' event via the callback, if provided."""
    if callback is not None:
        try:
            callback(make_done(answer))
        except Exception:
            pass


def safe_emit(
    callback: Callable[[dict[str, Any]], None] | None,
    event: AgentEvent,
) -> None:
    """Safely emit any typed event via the callback."""
    if callback is not None:
        try:
            callback(event.model_dump())
        except Exception:
            pass