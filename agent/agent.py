"""AgentSession — AI agent with hybrid LangGraph + AutoGen orchestration.

Architecture:
  1. LangGraph StateGraph handles outer orchestration (routing, state,
     verification loop), delegating complex tasks to AutoGen GroupChat
     and simple tasks to the ReAct fast path.
  2. Executor (LangChain ReAct) handles simple single-step queries.
  3. AutoGen GroupChat with 4 agents (Planner, Executor, Verifier,
     UserProxy) handles complex multi-step tasks inside a LangGraph node.

Two orchestrator modes supported:
  - "graph" (default): LangGraph + AutoGen hybrid — better routing,
        state management, checkpointing, and debug visualization.
  - "autogen": Original AutoGen-only mode — simpler, battle-tested.

Simple queries are auto-detected via LLM classification and routed to
the fast ReAct path, skipping the multi-agent overhead.
"""

import logging
import threading
from collections.abc import Callable
from typing import Any


from agent.executor import Executor
from agent.llm.factory import create_llm
from agent.memory.memory_manager import MemoryManager
from agent.progress import safe_done
from agent.utils.classifier import classify_query as _classify_query
from agent.utils.retry import llm_invoke_with_guard

# AutoGen is optional — only imported when using "autogen" mode
try:
    from agent.orchestrator import AutoGenOrchestrator
    _HAS_AUTOGEN = True
except ImportError:
    _HAS_AUTOGEN = False

# LangGraph is optional — only imported when using "graph" mode
try:
    from agent.graph_orchestrator import GraphOrchestrator
    _HAS_GRAPH = True
except ImportError:
    _HAS_GRAPH = False

logger = logging.getLogger(__name__)


class AgentSession:
    """Plan-and-Execute agent session.

    One instance = one agent runtime with:
        - LLM (remote API or local Ollama)
        - Short-term conversation memory
        - Orchestrator with planning, execution, reflection, verification
        - Auto-routing: simple queries → fast ReAct, complex queries → Plan-and-Execute

    Usage:
        session = AgentSession()                              # default: graph mode
        session = AgentSession(orchestrator_mode="autogen")  # original AutoGen mode
        answer = session.invoke("help me analyze this log", chat_history=[])
    """

    def __init__(self, orchestrator_mode: str = "graph") -> None:
        self._lock = threading.Lock()
        self._cancel_flag = threading.Event()
        self._is_running = False
        self._orchestrator_mode = orchestrator_mode
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

        llm = self._llm_provider.get_model()
        self._legacy_executor = Executor(llm)

        self._graph_orchestrator: Any = GraphOrchestrator() if (orchestrator_mode == "graph" and _HAS_GRAPH) or (_HAS_GRAPH and not _HAS_AUTOGEN) else None
        self._autogen_orchestrator: Any = AutoGenOrchestrator() if _HAS_AUTOGEN else None

        if self._graph_orchestrator is None and self._autogen_orchestrator is None:
            logger.warning("No orchestrator available, only basic ReAct mode")
        elif self._graph_orchestrator is not None:
            logger.info("AgentSession created | mode=graph | LLM: %s", self._llm_provider.model_name)
        else:
            logger.info("AgentSession created | mode=autogen | LLM: %s", self._llm_provider.model_name)

    @property
    def orchestrator_mode(self) -> str:
        """Return the current effective orchestrator mode."""
        if self._orchestrator_mode == "graph" and self._graph_orchestrator is not None:
            return "graph"
        if self._orchestrator_mode == "autogen" and self._autogen_orchestrator is not None:
            return "autogen"
        if self._graph_orchestrator is not None:
            return "graph"
        if self._autogen_orchestrator is not None:
            return "autogen"
        return "fallback"

    @property
    def token_usage(self) -> dict[str, int]:
        """Return accumulated token usage statistics."""
        return dict(self._token_usage)

    def _track_tokens(self, response: Any) -> None:
        """Extract and accumulate token usage from an LLM response."""
        try:
            usage = getattr(response, "usage_metadata", None)
            if usage:
                self._token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                self._token_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                self._token_usage["total_tokens"] += usage.get("total_tokens", 0)
                self._token_usage["calls"] += 1
        except Exception:
            pass

    def _track_usage_dict(self, usage: dict[str, int]) -> None:
        """Accumulate token usage from a dict (e.g. from executor stats)."""
        self._token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        self._token_usage["completion_tokens"] += usage.get("completion_tokens", 0)
        self._token_usage["total_tokens"] += usage.get("total_tokens", 0)
        self._token_usage["calls"] += 1

    def reset_token_usage(self) -> None:
        """Reset token usage counters."""
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
        """Execute a worker function with cancel-event support and memory tracking.

        Runs synchronously in the calling thread. If cancel is requested
        during execution, returns the cancellation message. The final
        answer is stored into short-term memory regardless of success.

        Args:
            worker: Callable(short_term_text, messages) -> answer string
            user_input: Original user message (for memory)
            chat_history: Original chat history (for memory formatting)
            progress_callback: Progress callback
            extra_context: Additional context (e.g. long-term memory) to append
            cancel_event: Optional external cancel event

        Returns:
            Answer string (or cancel/error message)
        """
        self._cancel_flag.clear()
        if cancel_event:
            cancel_event.clear()
        self._is_running = True
        short_term_text = self.memory.format_short_term(chat_history)
        messages_list = self.memory.format_short_term_messages(chat_history)

        context = short_term_text
        if extra_context:
            context = f"{short_term_text}\n\n{extra_context}"

        with self._lock:
            try:
                answer = worker(context, messages_list)
            except Exception as exc:
                logger.error("Worker error: %s", exc, exc_info=True)
                answer = f"Error: {exc}"
            finally:
                self._is_running = False

        if self._cancel_flag.is_set():
            answer = "⏹ 已停止"

        self.memory.add_interaction(user_input, answer)
        safe_done(progress_callback, answer)
        return answer

    def _graph_worker(self, context: str, messages: list) -> str:
        return self._graph_orchestrator.run(
            self._current_user_input, context, messages,
            progress_callback=self._current_progress_cb,
            cancel_event=self._cancel_flag,
        )

    def _legacy_worker(self, _: str, messages: list) -> str:
        return self._legacy_executor.execute(
            self._current_user_input, messages,
            progress_callback=self._current_progress_cb,
            cancel_event=self._cancel_flag,
        )

    def _autogen_worker(self, context: str, messages: list) -> str:
        return self._autogen_orchestrator.run(
            self._current_user_input, context, messages,
            progress_callback=self._current_progress_cb,
            cancel_event=self._cancel_flag,
        )

    def invoke(
        self, user_input: str, chat_history: list,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> str:
        """Run Plan-and-Execute loop for user input.

        In "graph" mode: LangGraph handles routing (classify → simple/complex)
            internally, plus verification and re-planning loops.
        In "autogen" mode: LLM classification selects between fast ReAct and
            AutoGen GroupChat (original behavior).

        Args:
            user_input: User message text
            chat_history: Chat history list
            progress_callback: Optional progress event callback
            cancel_event: Optional external cancel event

        Returns:
            Agent's final text response
        """
        if not user_input.strip():
            return ""

        self._current_user_input = user_input
        self._current_progress_cb = progress_callback

        long_term_context = self.memory.format_long_term_context(user_input)
        extra_context = f"## Long-term Memory\n{long_term_context}" if long_term_context else ""

        if self._orchestrator_mode == "graph" and self._graph_orchestrator:
            return self._execute_with_cancel(
                self._graph_worker, user_input, chat_history, progress_callback,
                extra_context=extra_context,
                cancel_event=cancel_event,
            )

        if self._autogen_orchestrator is None:
            logger.warning("[Agent] No orchestrator available, using legacy ReAct")
            return self._execute_with_cancel(
                self._legacy_worker, user_input, chat_history, progress_callback,
                cancel_event=cancel_event,
            )

        classification = _classify_query(self._llm_provider.get_model(), user_input)
        logger.info("[Agent] Query classified as '%s': %s", classification, user_input[:60])

        if classification == "simple":
            return self._execute_with_cancel(
                self._legacy_worker, user_input, chat_history, progress_callback,
                cancel_event=cancel_event,
            )

        return self._execute_with_cancel(
            self._autogen_worker, user_input, chat_history, progress_callback,
            extra_context=extra_context,
            cancel_event=cancel_event,
        )

    def invoke_legacy(
        self, user_input: str, chat_history: list,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> str:
        """Run legacy ReAct loop (single-step, no planning).

        Useful for simple queries that don't need multi-step planning.
        """
        if not user_input.strip():
            return ""

        self._current_user_input = user_input
        self._current_progress_cb = progress_callback

        def _worker(short_term_text: str, messages: list) -> str:
            return self._legacy_executor.execute(
                user_input, messages,
                progress_callback=progress_callback,
                cancel_event=self._cancel_flag,
            )

        return self._execute_with_cancel(
            _worker, user_input, chat_history, progress_callback,
            cancel_event=cancel_event,
        )

    def switch_mode(self, mode: str) -> None:
        """Switch orchestrator mode at runtime.

        Args:
            mode: "graph" or "autogen"
        """
        if mode not in ("graph", "autogen"):
            raise ValueError(f"Unknown mode: {mode}. Choose 'graph' or 'autogen'.")

        if mode == "graph" and _HAS_GRAPH:
            self._graph_orchestrator = GraphOrchestrator()
            self._orchestrator_mode = mode
            logger.info("Switched to graph mode")
        elif mode == "autogen" and _HAS_AUTOGEN:
            self._autogen_orchestrator = AutoGenOrchestrator()
            self._orchestrator_mode = mode
            logger.info("Switched to autogen mode")
        else:
            logger.warning("Switch failed: required dependency not available for '%s'", mode)

    def stop(self) -> None:
        """Request cancellation of the current invocation."""
        self._cancel_flag.set()
        logger.info("Cancel requested")

    @property
    def is_running(self) -> bool:
        """Check if an invocation is currently in progress."""
        return self._is_running

    def clear_long_term_memory(self) -> str:
        with self._lock:
            return self.memory.clear_long_term()

    @property
    def model_info(self) -> str:
        mode_label = (
            "LangGraph + AutoGen hybrid"
            if self._orchestrator_mode == "graph"
            else "AutoGen GroupChat (Planner + Executor + Verifier)"
        )
        return (
            f"Model: {self._llm_provider.model_name}  |  "
            f"Mode: {mode_label}"
        )

    def __enter__(self) -> "AgentSession":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()