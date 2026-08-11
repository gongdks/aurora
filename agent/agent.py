"""AgentSession — AI agent with AutoGen-based multi-agent orchestration.

Architecture:
  1. AutoGen GroupChat with 4 specialized agents (Planner, Executor,
     Verifier, UserProxy) handles complex multi-step tasks via a
     deterministic state machine.
  2. Executor (Legacy ReAct) handles simple single-step queries via fast path.
  3. LLM-based query classification auto-routes between the two paths.

Supports both Plan-and-Execute mode (default, via AutoGen) and legacy ReAct mode.
Simple queries are auto-detected via LLM classification and routed to
the fast ReAct path, skipping the GroupChat overhead.
"""

import logging
import threading
from collections.abc import Callable
from typing import Any


from agent.executor import Executor
from agent.llm.factory import create_llm
from agent.memory.memory_manager import MemoryManager
from agent.orchestrator import AutoGenOrchestrator
from agent.progress import safe_done
from agent.utils.retry import llm_invoke_with_guard

logger = logging.getLogger(__name__)

# Minimal prompt for LLM-based query complexity classification.
# The model must reply with exactly "simple" or "complex" — first token wins.
_QUERY_CLASSIFIER_PROMPT = """Classify this user request. Reply with exactly one word: "simple" or "complex".

Rules:
- "simple": a single-step task — math, factual lookup, translation, brief Q&A, read a file, simple web search.
- "complex": anything requiring multiple steps, planning, chaining tools (e.g. "search X then summarize Y"), analyzing and then acting, or involving conditional logic.

Request: {user_input}

Classification (simple/complex):"""

# Time budget for the classifier call — keep it tight
_CLASSIFIER_TIMEOUT = 5.0
_CLASSIFIER_MAX_RETRIES = 1


def _classify_query(llm: Any, user_input: str) -> str:
    """Use the LLM to classify a query as 'simple' or 'complex'.

    Args:
        llm: LangChain BaseChatModel instance.
        user_input: The user's raw message.

    Returns:
        "simple" or "complex". Falls back to "complex" on any error
        (over-planning is safer than silently skipping steps).
    """
    prompt = _QUERY_CLASSIFIER_PROMPT.format(user_input=user_input)
    try:
        response = llm_invoke_with_guard(
            llm, [{"role": "user", "content": prompt}],
            timeout=_CLASSIFIER_TIMEOUT,
            max_retries=_CLASSIFIER_MAX_RETRIES,
        )
        result = response.content.strip().lower()
        # First non-empty line, take only the first word
        first_word = result.split("\n")[0].strip().split()[0] if result else ""
        if first_word in ("simple", "complex"):
            return first_word
        # Fuzzy: if "simple" appears before "complex", treat as simple
        if "simple" in result and "complex" not in result:
            return "simple"
        return "complex"
    except Exception:
        # Safe default: treat as complex so user sees steps/reflection
        logger.warning("Query classifier failed, defaulting to complex path")
        return "complex"


class AgentSession:
    """Plan-and-Execute agent session.

    One instance = one agent runtime with:
        - LLM (remote API or local Ollama)
        - Short-term conversation memory
        - Orchestrator with planning, execution, reflection, verification
        - Auto-routing: simple queries → fast ReAct, complex queries → Plan-and-Execute

    Usage:
        session = AgentSession()
        answer = session.invoke("help me analyze this log", chat_history=[])
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancel_flag = threading.Event()

        self._llm_provider = create_llm()
        self.memory = MemoryManager()

        llm = self._llm_provider.get_model()
        self._orchestrator = AutoGenOrchestrator()
        self._legacy_executor = Executor(llm)

        logger.info(
            "AgentSession(AutoGen) created | LLM: %s",
            self._llm_provider.model_name,
        )

    def _run_in_thread(
        self,
        worker: Callable[[str, list], str],
        user_input: str,
        chat_history: list,
        progress_callback: Callable[[dict[str, Any]], None] | None,
    ) -> str:
        """Run a worker function in the calling thread with cancel support.

        Args:
            worker: Callable(short_term_text, messages) -> answer string
            user_input: Original user message (for memory)
            chat_history: Original chat history (for memory formatting)
            progress_callback: Progress callback

        Returns:
            Answer string (or cancel/error message)
        """
        self._cancel_flag.clear()
        short_term_text = self.memory.format_short_term(chat_history)
        messages_list = self.memory.format_short_term_messages(chat_history)

        with self._lock:
            try:
                answer = worker(short_term_text, messages_list)
            except Exception as exc:
                logger.error("Worker error: %s", exc, exc_info=True)
                answer = f"Error: {exc}"

        if self._cancel_flag.is_set():
            answer = "⏹ 已停止"

        self.memory.add_interaction(user_input, answer)
        safe_done(progress_callback, answer)
        return answer

    def invoke(
        self, user_input: str, chat_history: list,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> str:
        """Run Plan-and-Execute loop for user input.

        Simple queries are auto-routed to the fast ReAct path via LLM classification,
        skipping the full planning overhead.

        Args:
            user_input: User message text
            chat_history: Chat history list
            progress_callback: Optional progress event callback

        Returns:
            Agent's final text response
        """
        if not user_input.strip():
            return ""

        # LLM-based classification: simple → fast ReAct, complex → Plan-and-Execute
        classification = _classify_query(self._llm_provider.get_model(), user_input)
        logger.info("[Agent] Query classified as '%s': %s", classification, user_input[:60])

        if classification == "simple":
            def _fast_worker(_: str, messages: list) -> str:
                return self._legacy_executor.execute(
                    user_input, messages,
                    progress_callback=progress_callback,
                    cancel_event=self._cancel_flag,
                )
            return self._run_in_thread(_fast_worker, user_input, chat_history, progress_callback)

        # Full Plan-and-Execute path for complex/multi-step queries
        def _worker(short_term_text: str, messages: list) -> str:
            return self._orchestrator.run(
                user_input, short_term_text, messages,
                progress_callback=progress_callback,
                cancel_event=self._cancel_flag,
            )

        return self._run_in_thread(_worker, user_input, chat_history, progress_callback)

    def invoke_legacy(
        self, user_input: str, chat_history: list,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> str:
        """Run legacy ReAct loop (single-step, no planning).

        Useful for simple queries that don't need multi-step planning.
        """
        if not user_input.strip():
            return ""

        def _worker(short_term_text: str, messages: list) -> str:
            return self._legacy_executor.execute(
                user_input, messages,
                progress_callback=progress_callback,
            )

        return self._run_in_thread(_worker, user_input, chat_history, progress_callback)

    def stop(self) -> None:
        """Request cancellation of the current invocation."""
        self._cancel_flag.set()
        logger.info("Cancel requested")

    @property
    def is_running(self) -> bool:
        """Check if an invocation is currently in progress."""
        return self._cancel_flag.is_set() or (
            self._lock.locked()
        )

    def clear_long_term_memory(self) -> str:
        with self._lock:
            return self.memory.clear_long_term()

    @property
    def model_info(self) -> str:
        return (
            f"Model: {self._llm_provider.model_name}  |  "
            f"Mode: AutoGen GroupChat (Planner + Executor + Verifier)"
        )

    def __enter__(self) -> "AgentSession":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()