"""Event handler — processes agent events and updates display state."""

from __future__ import annotations

import time
from typing import Any


class DisplayState:
    """Tracks display-level statistics (elapsed time, tool count, step count, tokens)."""

    def __init__(self) -> None:
        self._start_time: float | None = None
        self._tool_count: int = 0
        self._step_count: int = 0
        self._token_count: int = 0

    @property
    def elapsed(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    @property
    def tool_count(self) -> int:
        return self._tool_count

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def token_count(self) -> int:
        return self._token_count

    def add_tokens(self, count: int) -> None:
        self._token_count += count

    def reset(self) -> None:
        self._start_time = None
        self._tool_count = 0
        self._step_count = 0
        self._token_count = 0

    def _ensure_started(self) -> None:
        if self._start_time is None:
            self._start_time = time.time()


def handle_event(display: DisplayState, event: dict[str, Any]) -> None:
    """Feed an agent event into the display state tracker.

    Args:
        display: DisplayState instance to update.
        event: Agent event dict with 'type' field.
    """
    event_type = event.get("type", "")

    if event_type == "tool":
        display._ensure_started()
        display._tool_count += 1

    elif event_type == "log":
        display._ensure_started()
        display._step_count += 1

    elif event_type == "plan":
        display._ensure_started()

    elif event_type == "status":
        display._ensure_started()

    elif event_type == "done":
        display._ensure_started()

    elif event_type == "error":
        display._ensure_started()