"""Agent event types — typed events flowing from agent to UI."""

from __future__ import annotations

from typing import Any, TypedDict


class AgentEvent(TypedDict, total=False):
    """Generic agent event passed through the event bus."""

    type: str
    message: str
    name: str
    input: str
    output: str | None
    token: str
    answer: str
    _ts: float
    _call_id: int


class ToolEvent(TypedDict, total=False):
    """Tool call event — fired when a tool starts or completes."""

    type: str
    name: str
    input: str
    output: str | None
    _ts: float
    _call_id: int