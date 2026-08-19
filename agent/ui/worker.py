"""Agent worker — QThread that runs the agent session and emits signals."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from PyQt6.QtCore import QThread, pyqtSignal

from agent.agent import AgentSession
from agent.models import AgentResult, AgentStatus

logger = logging.getLogger(__name__)


class EventBus:
    """Simple event bus for subscribing to agent progress events."""

    def __init__(self) -> None:
        self._subscribers: list[Callable[[dict[str, Any]], None]] = []
        self._lock = threading.Lock()

    def subscribe(self, callback: Callable[[dict[str, Any]], None]) -> None:
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[dict[str, Any]], None]) -> None:
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def emit(self, event: dict[str, Any]) -> None:
        with self._lock:
            callbacks = list(self._subscribers)
        for cb in callbacks:
            try:
                cb(event)
            except Exception as exc:
                logger.error("Event subscriber error: %s", exc)


class AgentWorker(QThread):
    """QThread that runs an AgentSession invocation.

    Signals:
        result_signal: Emitted when a result is available (AgentResult).
        finished_signal: Emitted when the worker completes (answer string).
        error_signal: Emitted on error (error message string).
    """

    result_signal = pyqtSignal(object)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(dict)

    def __init__(
        self,
        session: AgentSession,
        message: str,
        history: list[dict[str, Any]],
    ) -> None:
        super().__init__()
        self._session = session
        self._message = message
        self._history = list(history)
        self._cancel_event = threading.Event()
        self.event_bus = EventBus()

    def run(self) -> None:
        try:
            start_time = __import__("time").time()

            def _progress_cb(event: dict[str, Any]) -> None:
                self.progress_signal.emit(event)

            answer = self._session.invoke(
                self._message,
                self._history,
                progress_callback=_progress_cb,
                cancel_event=self._cancel_event,
            )

            elapsed = __import__("time").time() - start_time

            token_usage = self._session.token_usage

            if self._cancel_event.is_set():
                result = AgentResult(
                    content="⏹ 已停止",
                    status=AgentStatus.CANCELLED,
                    elapsed=elapsed,
                    tool_calls=[],
                    metadata={"token_usage": token_usage},
                )
                self.result_signal.emit(result)
                self.finished_signal.emit("⏹ 已停止")
                return

            status = AgentStatus.COMPLETED
            if answer.startswith("Error:") or answer.startswith("[Error]"):
                status = AgentStatus.FAILED

            result = AgentResult(
                content=answer,
                status=status,
                elapsed=elapsed,
                tool_calls=[],
                metadata={"token_usage": token_usage},
            )
            self.result_signal.emit(result)
            self.finished_signal.emit(answer)

        except Exception as exc:
            logger.error("AgentWorker error: %s", exc, exc_info=True)
            self.error_signal.emit(str(exc))

    def cancel(self) -> None:
        """Request cancellation of the running agent."""
        self._cancel_event.set()
        self._session.stop()