"""Executor — runs ReAct tool-calling loop for a single user request.

Uses the shared runner module for ReAct agent creation and execution.
"""

import logging
import threading
from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel

from agent.config import settings
from agent.runner import (
    StreamingCallbackHandler,
    _BaseToolEventTracker,
    build_react_executor,
    run_react_step,
)

logger = logging.getLogger(__name__)


class Executor:
    """ReAct executor — runs the tool-calling loop for one request."""

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm
        self._cancel_flag = threading.Event()

    def execute(
        self,
        user_input: str,
        chat_history_messages: list,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> str:
        """Execute a tool-calling loop for the given user input.

        Args:
            user_input: The user's request
            chat_history_messages: LangChain message list (HumanMessage/AIMessage)
            progress_callback: Optional progress event callback
            cancel_event: Optional cancel event — set to stop execution

        Returns:
            Final answer string
        """
        self._cancel_flag.clear()
        logger.info("[Exec] Starting tool-calling loop: %s", user_input[:60])

        # Merge external cancel_event with local flag
        effective_cancel = cancel_event or self._cancel_flag

        tracker = _BaseToolEventTracker(progress_callback)
        executor = build_react_executor(
            self._llm,
            max_iterations=settings.MAX_ITERATIONS,
            max_execution_time=settings.MAX_EXECUTION_TIME_SEC,
            verbose=True,
        )

        streaming_handler = StreamingCallbackHandler(progress_callback)
        result = run_react_step(
            executor,
            user_input,
            chat_history=chat_history_messages,
            progress_callback=progress_callback,
            tracker=tracker,
            cancel_event=effective_cancel,
            extra_callbacks=[streaming_handler],
        )

        if effective_cancel.is_set():
            return "⏹ 已停止"
        if result["status"] == "completed":
            return result["result"]
        if result["status"] == "cancelled":
            return "⏹ 已停止"
        return f"[Error] {result.get('error', 'Execution failed')}"

    def stop(self) -> None:
        self._cancel_flag.set()