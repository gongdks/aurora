"""Progress event protocol — typed event passing from agent to UI.

Events flow from AgentSession → UI via progress_callback.
Each event is a dict with a 'type' field.

TypedDict types are provided for IDE autocompletion and mypy checking.
"""

from __future__ import annotations

import time as _time
from typing import Any, TypedDict


# ---- Event type constants ----
TOOL = "tool"
LOG = "log"
DONE = "done"
ERROR = "error"
STREAMING_TOKEN = "streaming_token"


# ---- Typed event dictionaries ----
class ToolEvent(TypedDict, total=False):
    type: str
    name: str
    input: str
    output: str | None
    _ts: float
    _call_id: int


class LogEvent(TypedDict, total=False):
    type: str
    message: str


class DoneEvent(TypedDict, total=False):
    type: str
    answer: str


class ErrorEvent(TypedDict, total=False):
    type: str
    message: str


class StreamingTokenEvent(TypedDict, total=False):
    type: str
    token: str


ProgressEvent = ToolEvent | LogEvent | DoneEvent | ErrorEvent | StreamingTokenEvent


# ---- Event factories ----
def make_tool(
    name: str, input_str: str, output: str | None = None, call_id: int | None = None,
) -> dict[str, Any]:
    return {
        "type": TOOL,
        "name": name,
        "input": input_str,
        "output": output,
        "_ts": _time.time(),
        "_call_id": call_id or 0,
    }


def make_log(message: str) -> dict[str, Any]:
    return {"type": LOG, "message": message}


def make_done(answer: str) -> dict[str, Any]:
    return {"type": DONE, "answer": answer}


def make_error(message: str) -> dict[str, Any]:
    return {"type": ERROR, "message": message}


def make_streaming_token(token: str) -> dict[str, Any]:
    return {"type": STREAMING_TOKEN, "token": token}


# ---- Convenience helpers ----
def safe_log(cb, message: str) -> None:
    if cb:
        cb(make_log(message))


def safe_done(cb, answer: str) -> None:
    if cb:
        cb(make_done(answer))


# ---- Progress tracker ----
_MAX_EVENTS = 500
_DEFAULT_EXPECTED = 20


class ProgressTracker:
    """Accumulates progress events for UI rendering.

    Automatically caps stored events at _MAX_EVENTS to prevent
    unbounded memory growth during long sessions.
    """

    def __init__(self, expected_event_count: int = _DEFAULT_EXPECTED) -> None:
        self._events: list[dict[str, Any]] = []
        self._tool_events: list[dict[str, Any]] = []
        self._done_answer: str = ""
        self._has_error: bool = False
        self._error_message: str = ""
        self._start_time: float | None = None
        self._expected = expected_event_count

    def feed(self, event: dict[str, Any]) -> None:
        self._events.append(event)
        etype = event.get("type")

        if etype == TOOL:
            if self._start_time is None:
                self._start_time = _time.time()
            self._tool_events.append(event)
        elif etype == DONE:
            self._done_answer = event["answer"]
        elif etype == ERROR:
            self._has_error = True
            self._error_message = event["message"]

        # Trim to prevent unbounded growth
        self._trim_if_needed()

    def _trim_if_needed(self) -> None:
        """Keep only the most recent events when over capacity."""
        limit = max(_MAX_EVENTS, self._expected * 4)
        if len(self._events) > limit:
            self._events = self._events[-limit:]
        if len(self._tool_events) > limit:
            self._tool_events = self._tool_events[-limit:]

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        return list(self._tool_events)

    @property
    def done_answer(self) -> str:
        return self._done_answer

    @property
    def has_error(self) -> bool:
        return self._has_error

    @property
    def error_message(self) -> str:
        return self._error_message

    @property
    def elapsed(self) -> float:
        if self._start_time is None:
            return 0.0
        return _time.time() - self._start_time

    @property
    def events(self) -> list[dict[str, Any]]:
        return list(self._events)

    def reset(self) -> None:
        """Reset tracker for a new agent invocation."""
        self._events.clear()
        self._tool_events.clear()
        self._done_answer = ""
        self._has_error = False
        self._error_message = ""
        self._start_time = None