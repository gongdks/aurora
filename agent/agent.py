"""AgentSession — AI agent with pure LangGraph orchestration.

Architecture:
  1. LangGraph StateGraph handles routing (classify → simple/complex),
     planning, execution loop, verification, and re-planning.
  2. Each plan step executes via LangChain ReAct tool-calling executor.
  3. Simple queries are auto-detected and routed to the fast ReAct path.

Thread safety:
  GraphOrchestrator uses threading.local() for per-run context,
  so multiple AgentSession instances (or concurrent invoke() calls)
  can run safely without blocking each other.

Orchestration flow:
  classify → simple → react_fast → END
  classify → complex → plan → execute_loop → verify → END/re-plan
"""

import logging
import threading
from collections.abc import Callable
from typing import Any

from agent.graph_orchestrator import GraphOrchestrator
from agent.llm.factory import create_llm
from agent.memory.memory_manager import MemoryManager
from agent.progress import make_error, safe_done

logger = logging.getLogger(__name__)


class AgentSession:
    """Plan-and-Execute agent session powered by pure LangGraph.

    One instance = one agent runtime with:
        - LLM (remote API or local Ollama)
        - Short-term conversation memory
        - Long-term knowledge persistence
        - LangGraph orchestrator with planning, execution, verification
        - Auto-routing: simple queries → fast ReAct, complex → Plan-and-Execute

    Thread-safe: no global lock. Each invoke() call runs independently.
    """

    def __init__(self) -> None:
        self._cancel_flag = threading.Event()
        self._is_running = False
        self._token_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
        }
        self._current_user_input: str = ""
        self._current_progress_cb: Callable | None = None

        self._llm_provider = create_llm()
        self.memory = MemoryManager()
        self._graph_orchestrator = GraphOrchestrator()
        logger.info("AgentSession created | mode=graph | LLM: %s", self._llm_provider.model_name)

    @property
    def token_usage(self) -> dict[str, int]:
        return dict(self._token_usage)

    def _track_usage(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
    ) -> None:
        self._token_usage["prompt_tokens"] += prompt_tokens
        self._token_usage["completion_tokens"] += completion_tokens
        self._token_usage["total_tokens"] += total_tokens
        self._token_usage["calls"] += 1

    def reset_token_usage(self) -> None:
        self._token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
        }

    def _execute_with_cancel(
        self,
        worker: Callable[[str, list], str],
        user_input: str,
        chat_history: list,
        progress_callback: Callable[[dict[str, Any]], None] | None,
        extra_context: str = "",
        cancel_event: threading.Event | None = None,
    ) -> str:
        self._cancel_flag.clear()
        if cancel_event:
            cancel_event.clear()
        self._is_running = True
        short_term_text = self.memory.format_short_term(chat_history)
        messages_list = self.memory.format_short_term_messages(chat_history)

        context = short_term_text
        if extra_context:
            context = f"{short_term_text}\n\n{extra_context}"

        try:
            answer = worker(context, messages_list)
        except Exception as exc:
            logger.error("Worker error: %s", exc, exc_info=True)
            answer = f"错误: {exc}"
        finally:
            self._is_running = False

        if self._cancel_flag.is_set():
            answer = "⏹ 已停止"

        self.memory.add_interaction(user_input, answer)
        safe_done(progress_callback, answer)
        return answer

    def _graph_worker(self, context: str, messages: list) -> str:
        def _track_tokens(count: int) -> None:
            self._track_usage(total_tokens=count)

        return self._graph_orchestrator.run(
            self._current_user_input,
            context,
            messages,
            progress_callback=self._current_progress_cb,
            cancel_event=self._cancel_flag,
            token_tracker=_track_tokens,
        )

    def invoke(
        self,
        user_input: str,
        chat_history: list,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> str:
        if not user_input.strip():
            return ""

        self._current_user_input = user_input
        self._current_progress_cb = progress_callback

        long_term_context = self.memory.format_long_term_context(user_input)
        extra_context = f"## 长期记忆\n{long_term_context}" if long_term_context else ""

        return self._execute_with_cancel(
            self._graph_worker,
            user_input,
            chat_history,
            progress_callback,
            extra_context=extra_context,
            cancel_event=cancel_event,
        )

    def stop(self) -> None:
        self._cancel_flag.set()
        self._graph_orchestrator.request_stop()
        logger.info("Cancel requested")

    @property
    def is_running(self) -> bool:
        return self._is_running

    def clear_long_term_memory(self) -> str:
        return self.memory.clear_long_term()

    @property
    def model_info(self) -> str:
        return (
            f"Model: {self._llm_provider.model_name}  |  "
            f"Mode: LangGraph Plan-and-Execute"
        )

    def __enter__(self) -> "AgentSession":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()