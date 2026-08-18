"""Progress event factory — creates typed progress events for the UI."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


def make_log(message: str) -> dict[str, Any]:
    """Create a log message event."""
    return {
        "type": "log",
        "message": message,
        "_ts": time.time(),
    }


def make_tool(name: str, input_str: str, output: str | None = None, *, call_id: int | None = None) -> dict[str, Any]:
    """Create a tool call event."""
    event: dict[str, Any] = {
        "type": "tool",
        "name": name,
        "input": input_str[:500],
        "_ts": time.time(),
    }
    if output is not None:
        event["output"] = output[:1000]
    if call_id is not None:
        event["_call_id"] = call_id
    return event


def make_streaming_token(token: str) -> dict[str, Any]:
    """Create a streaming token event."""
    return {
        "type": "streaming_token",
        "token": token,
        "_ts": time.time(),
    }


def make_done(answer: str) -> dict[str, Any]:
    """Create a completion event."""
    return {
        "type": "done",
        "answer": answer,
        "_ts": time.time(),
    }


def make_error(message: str) -> dict[str, Any]:
    """Create an error event."""
    return {
        "type": "error",
        "message": message,
        "_ts": time.time(),
    }


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