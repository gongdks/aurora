"""GraphOrchestrator — LangGraph + AutoGen hybrid orchestration.

Architecture:

  ┌──────────┐    ┌──────────────┐    ┌──────────┐
  │ classify  │───→│ react_fast   │───→│   END    │
  └────┬─────┘    └──────────────┘    └──────────┘
       │ complex
       ▼
  ┌──────────────┐    ┌──────────────┐    ┌──────────┐
  │ autogen_complex│───→│check_verification│───→│   END    │
  └──────────────┘    └──────┬───────┘    └──────────┘
                              │ not_complete
                              ▼
                        re-plan via autogen_complex

LangGraph provides: declarative routing, checkpointing, human-in-the-loop,
debug visualization, and typed state management.

AutoGen GroupChat runs as a single "complex task" node inside the graph,
reusing all existing agent definitions and tools unchanged.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agent.llm.factory import create_llm
from agent.progress import make_log, make_streaming_token

try:
    from agent.orchestrator import AutoGenOrchestrator
    _HAS_AUTOGEN = True
except ImportError:
    _HAS_AUTOGEN = False

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    """Shared state across all LangGraph nodes.

    This is the single source of truth flowing through the graph.
    """

    user_input: str
    short_term_text: str
    chat_history_messages: list
    classification: str
    plan: str
    plan_rounds: int
    result: str
    is_done: bool


class GraphOrchestrator:
    """LangGraph + AutoGen hybrid Plan-and-Execute orchestrator.

    Combines LangGraph's strengths (deterministic routing, state management,
    checkpointing, debug visualization) with AutoGen's multi-agent conversation
    abilities for complex tasks.

    The graph:
      1. classify → simple → react_fast → END
      2. classify → complex → autogen_complex → check_verification → END/re-plan

    Usage:
        orch = GraphOrchestrator()
        answer = orch.run(user_input, short_term_text, messages, progress_cb=cb)
    """

    def __init__(self) -> None:
        self._cancel_event: threading.Event | None = None
        self._progress_cb: Callable[[dict[str, Any]], None] | None = None
        self._llm_provider = create_llm()
        self._llm = self._llm_provider.get_model()
        self._autogen = AutoGenOrchestrator() if _HAS_AUTOGEN else None
        self._executor = None
        self._last_token_count = 0
        try:
            self._graph = self._build_graph()
        except Exception as exc:
            logger.error("[GraphOrch] Graph build failed: %s", exc, exc_info=True)
            self._graph = None
        logger.info("[GraphOrch] LangGraph ready | LLM: %s | AutoGen: %s",
                    self._llm_provider.model_name,
                    "available" if self._autogen else "not installed")

    @property
    def _cached_executor(self) -> Any:
        """Lazy-init and cache the Executor instance."""
        if self._executor is None:
            from agent.executor import Executor
            self._executor = Executor(self._llm)
        return self._executor

    def _build_graph(self) -> Any:
        graph = StateGraph(AgentState)

        graph.add_node("classify", self._node_classify)
        graph.add_node("react_fast", self._node_react_fast)
        graph.add_node("autogen_complex", self._node_autogen_complex)
        graph.add_node("check_verification", self._node_check_verification)

        graph.add_edge(START, "classify")

        graph.add_conditional_edges(
            "classify",
            self._route_after_classify,
            {
                "simple": "react_fast",
                "complex": "autogen_complex",
            },
        )

        graph.add_edge("react_fast", END)

        graph.add_edge("autogen_complex", "check_verification")

        graph.add_conditional_edges(
            "check_verification",
            self._route_after_verify,
            {
                "complete": END,
                "incomplete": "autogen_complex",
            },
        )

        return graph.compile()

    def run(
        self,
        user_input: str,
        short_term_text: str,
        chat_history_messages: list,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> str:
        self._cancel_event = cancel_event
        self._progress_cb = progress_callback

        if self._graph is None:
            return "[Error] LangGraph not available. Please check the logs for build errors."

        initial_state: AgentState = {
            "user_input": user_input,
            "short_term_text": short_term_text,
            "chat_history_messages": chat_history_messages,
            "classification": "",
            "plan": "",
            "plan_rounds": 0,
            "result": "",
            "is_done": False,
        }

        try:
            final_state = self._graph.invoke(initial_state)
        except Exception as exc:
            logger.error("[GraphOrch] Graph error: %s, falling back to AutoGen", exc, exc_info=True)
            if progress_callback:
                progress_callback(make_log(f"⚠️ Graph error, falling back to direct AutoGen: {exc}"))
            if self._autogen is not None:
                try:
                    return self._autogen.run(
                        user_input, short_term_text, chat_history_messages,
                        progress_callback=progress_callback,
                        cancel_event=cancel_event,
                    )
                except Exception as fallback_exc:
                    logger.error("[GraphOrch] Fallback also failed: %s", fallback_exc)
                    if progress_callback:
                        progress_callback(make_log(f"❌ Fallback failed: {fallback_exc}"))
                    return f"Error: {fallback_exc}"
            return f"Error: {exc} (AutoGen not installed, cannot fall back)"

        return final_state.get("result", "已完成，但未产生文本输出。")

    # ------------------------------------------------------------------
    # Node implementations
    # ------------------------------------------------------------------

    def _node_classify(self, state: AgentState) -> dict[str, Any]:
        """Classify query complexity — simple (fast path) or complex (plan+execute)."""
        from agent.utils.classifier import classify_query

        if self._cancel_event and self._cancel_event.is_set():
            return {"result": "⏹ 已停止", "is_done": True}

        if self._progress_cb:
            self._progress_cb(make_log("🤔 **Analyzing query complexity...**"))

        classification = classify_query(self._llm, state["user_input"])

        if self._progress_cb:
            icon = "⚡" if classification == "simple" else "🧩"
            self._progress_cb(
                make_log(f"{icon} **Route: {classification}** — {'fast path' if classification == 'simple' else 'multi-step Plan-and-Execute'}")
            )
            self._progress_cb(make_streaming_token(f" → {classification}"))

        return {"classification": classification}

    def _stream_llm_response(self, prompt: str) -> str:
        """Stream LLM response, emitting tokens via progress_callback."""
        collected = []
        token_count = 0
        try:
            for chunk in self._llm.stream(prompt):
                if self._cancel_event and self._cancel_event.is_set():
                    break
                token = chunk.content
                if token:
                    collected.append(token)
                    token_count += 1
                    if self._progress_cb:
                        self._progress_cb(make_streaming_token(token))
        except Exception:
            pass
        self._last_token_count = token_count
        return "".join(collected)

    def _node_react_fast(self, state: AgentState) -> dict[str, Any]:
        """Execute a simple query via the LangChain ReAct fast path."""
        if self._cancel_event and self._cancel_event.is_set():
            return {"result": "⏹ 已停止", "is_done": True}

        if self._progress_cb:
            self._progress_cb(make_log("⚡ **Running fast ReAct path...**"))

        result = self._cached_executor.execute(
            state["user_input"],
            state.get("chat_history_messages", []),
            progress_callback=self._progress_cb,
            cancel_event=self._cancel_event,
        )

        return {"result": result, "is_done": True}

    def _node_autogen_complex(self, state: AgentState) -> dict[str, Any]:
        """Execute complex multi-step plan via AutoGen GroupChat.

        Delegates to AutoGenOrchestrator.run_complex() which handles
        all GroupChat creation, speaker selection, and message extraction.
        """
        if self._cancel_event and self._cancel_event.is_set():
            return {"result": "⏹ 已停止", "is_done": True}

        if self._autogen is None:
            return {
                "result": "⚠️ AutoGen 未安装，无法执行复杂多步任务。请安装: pip install pyautogen",
                "is_done": True,
            }

        plan_rounds = state.get("plan_rounds", 0)

        try:
            result = self._autogen.run_complex(
                user_input=state["user_input"],
                chat_history_text=state.get("short_term_text", ""),
                progress_callback=self._progress_cb,
                cancel_event=self._cancel_event,
                plan_rounds=plan_rounds,
                previous_plan=state.get("plan", ""),
                previous_result=state.get("result", ""),
            )
        except Exception as exc:
            logger.error("[GraphOrch] AutoGen GroupChat error: %s", exc, exc_info=True)
            if self._progress_cb:
                self._progress_cb(make_log(f"❌ AutoGen error: {exc}"))
            return {"result": f"Error: {exc}", "is_done": True}

        if self._cancel_event and self._cancel_event.is_set():
            return {"result": "⏹ 已停止", "is_done": True}

        return {
            "result": result["answer"],
            "plan": result["plan"],
            "plan_rounds": result["plan_rounds"],
        }

    def _node_check_verification(self, state: AgentState) -> dict[str, Any]:
        """Verify whether the goal is achieved.

        Uses a lightweight LLM call to check if the result is sufficient.
        Falls back to "complete" if the result looks good or if max re-plans exceeded.
        """
        plan_rounds = state.get("plan_rounds", 0)
        result = state.get("result", "")
        user_input = state.get("user_input", "")

        if plan_rounds >= 3:
            logger.info("[GraphOrch] Max re-plan rounds (%d) reached, ending", plan_rounds)
            return {"is_done": True}

        if not result or result in ("⏹ 已停止", ""):
            return {"is_done": True}

        if self._progress_cb:
            self._progress_cb(make_log("🔍 **Verifying goal achievement...**"))

        prompt = (
            f"Original goal: {user_input}\n\n"
            f"Current result: {result}\n\n"
            "Is the goal fully achieved? Answer exactly 'yes' or 'no'."
        )

        try:
            answer = self._stream_llm_response(prompt).strip().lower()

            if self._progress_cb:
                self._progress_cb(make_log(f"🔍 Verification: {answer}"))

            if answer.startswith("yes"):
                return {"is_done": True}

            if plan_rounds >= 2:
                if self._progress_cb:
                    self._progress_cb(make_log("⚠️ Max re-plans reached, accepting current result"))
                return {"is_done": True}

            return {"is_done": False}

        except Exception:
            logger.warning("[GraphOrch] Verification LLM call failed, defaulting to complete")
            if self._progress_cb:
                self._progress_cb(make_log("⚠️ Verification failed, proceeding with result"))
            return {"is_done": True}

    # ------------------------------------------------------------------
    # Routing logic
    # ------------------------------------------------------------------

    @staticmethod
    def _route_after_classify(state: AgentState) -> str:
        return state.get("classification", "complex")

    @staticmethod
    def _route_after_verify(state: AgentState) -> str:
        return "complete" if state.get("is_done") else "incomplete"