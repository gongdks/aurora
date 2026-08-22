"""ToolExecutorGraph — pure LangGraph tool execution engine.

Replaces the procedural runner.py (LangChain AgentExecutor) with a
LangGraph StateGraph + ToolNode approach for native tool execution.

Architecture (LangGraph StateGraph + ToolNode):

  ┌──────────┐    ┌──────────┐    ┌──────────┐
  │  reason  │───→│ tool_node│───→│  observe │───→ loop
  └────┬─────┘    └──────────┘    └──────────┘
       │ no tools
       ▼
  ┌──────────┐
  │   END    │
  └──────────┘

LangGraph design principles followed:
  1. Entry point (execute) prepares initial state — nodes trust state
  2. ToolNode manages `messages` (adds AIMessage/ToolMessage natively)
  3. reason node is pure: read state → LLM call → return update
  4. Config carries runtime context (llm, callbacks, cancel_event)
  5. No side effects in nodes — LLM configured once at entry

Key features:
  • ToolNode for native tool execution (no AgentExecutor)
  • StateGraph with TypedDict state for clean data flow
  • Command-based routing for reason/act/observe loop
  • Shared checkpointer for unified state persistence
  • Subgraph composition support for embedding in parent graphs

Note: _observe only emits progress events and clears state fields —
it MUST NOT modify `messages` to avoid corrupting the conversation.
"""

from __future__ import annotations

import logging
import operator
import time
import threading
from collections.abc import Callable
from typing import Annotated, Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from agent.config import settings
from agent.tools.registry import list_tools, list_scene_tools, ToolRouter

logger = logging.getLogger(__name__)

_MAX_TOOL_ITERATIONS = 10


class ToolExecutorState(TypedDict, total=False):
    input_text: str
    chat_history: list
    messages: Annotated[list, operator.add]
    tool_calls: list
    tool_outputs: list
    iteration: int
    max_iterations: int
    status: str
    result: str
    total_time: float
    cancelled: bool


class ToolExecutorGraph:
    """LangGraph-native tool execution engine.

    Supports both standalone invocation and subgraph composition.
    When used as a subgraph, the parent graph's checkpointer is shared.

    Usage (standalone):
        executor = ToolExecutorGraph(llm=my_llm, tools=my_tools)
        result = executor.execute(
            input_text="分析这份数据",
            cancel_event=my_event,
        )

    Usage (as subgraph):
        main_graph.add_node("tool_exec", executor.compiled_graph)
    """

    def __init__(
        self,
        llm: Any = None,
        tools: list | None = None,
        scene: str | None = None,
        checkpointer: Any | None = None,
        max_iterations: int | None = None,
    ) -> None:
        self._llm = llm
        self._max_iterations = max_iterations or settings.MAX_ITERATIONS

        if tools is not None:
            self._tools = tools
        elif scene:
            self._tools = list_scene_tools(scene)
        else:
            self._tools = list_tools()

        self._tool_names = ", ".join(
            getattr(t, "name", str(t)) for t in self._tools
        )

        self._checkpointer = checkpointer
        self._graph = self._build_graph()

    @property
    def compiled_graph(self) -> Any:
        """Expose compiled graph for subgraph composition."""
        return self._graph

    def _build_graph(self) -> Any:
        graph = StateGraph(ToolExecutorState)

        graph.add_node("reason", self._node_reason)
        graph.add_node("execute_tools", ToolNode(self._tools))
        graph.add_node("observe", self._node_observe)

        graph.add_edge(START, "reason")
        graph.add_edge("execute_tools", "observe")

        return graph.compile(checkpointer=self._checkpointer)

    def execute(
        self,
        input_text: str,
        chat_history: list | None = None,
        tools: list | None = None,
        extra_tools: list | None = None,
        cancel_event: threading.Event | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        token_tracker: Callable[[int], None] | None = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if cancel_event and cancel_event.is_set():
            return {
                "status": "cancelled",
                "result": "⏹ 已停止",
                "iterations": 0,
                "time": 0.0,
                "hit_limit": False,
                "tool_calls": 0,
                "llm_calls": 0,
            }

        messages = self._prepare_initial_messages(input_text, chat_history)

        logger.info("[ToolExecGraph] execute: input_text=%s, chat_history_len=%d, prepared_msgs=%d",
                     str(input_text)[:80], len(chat_history or []), len(messages))

        llm_for_graph = self._prepare_llm_for_graph()

        initial_state: ToolExecutorState = {
            "input_text": input_text,
            "chat_history": chat_history or [],
            "messages": messages,
            "tool_calls": [],
            "tool_outputs": [],
            "iteration": 0,
            "max_iterations": self._max_iterations,
            "status": "running",
            "result": "",
            "total_time": 0.0,
            "cancelled": False,
        }

        configurable = {}
        if config and "configurable" in config:
            configurable = dict(config["configurable"])
        configurable["thread_id"] = f"tool_exec_{id(self)}_{id(input_text)}"
        configurable.setdefault("cancel_event", cancel_event)
        configurable.setdefault("progress_callback", progress_callback)
        configurable.setdefault("token_tracker", token_tracker)
        configurable.setdefault("llm", llm_for_graph)

        run_config = {"configurable": configurable}

        start_time = time.time()

        try:
            final = self._graph.invoke(initial_state, config=run_config)
            elapsed = time.time() - start_time

            tool_call_count = final.get("iteration", 0)
            result = final.get("result", "")
            status = final.get("status", "completed")

            if final.get("cancelled"):
                status = "cancelled"
                result = "⏹ 已停止"

            return {
                "status": status,
                "result": result,
                "iterations": final.get("iteration", 0),
                "time": elapsed,
                "hit_limit": status == "max_iterations",
                "tool_calls": tool_call_count,
                "llm_calls": final.get("iteration", 0),
            }

        except Exception as exc:
            elapsed = time.time() - start_time
            logger.error("[ToolExecGraph] Execution failed: %s", exc, exc_info=True)
            return {
                "status": "failed",
                "result": "",
                "error": str(exc),
                "iterations": 0,
                "time": elapsed,
                "hit_limit": False,
                "tool_calls": 0,
                "llm_calls": 0,
            }

    def _prepare_initial_messages(
        self, input_text: str, chat_history: list | None
    ) -> list:
        """Entry-point message preparation (LangGraph best practice).

        Ensures the initial state always contains a HumanMessage with
        the current user input, appended AFTER any chat history so the
        LLM sees the full conversation in chronological order.
        Done ONCE at entry — nodes trust state.

        LangGraph principle: state is the single source of truth.
        The entry point is responsible for assembling the complete
        message history; nodes never patch or reconstruct messages.
        """
        from langchain_core.messages import HumanMessage

        raw = list(chat_history or [])
        messages = ToolExecutorGraph._ensure_langchain_messages(raw)

        if input_text:
            messages.append(HumanMessage(content=input_text))

        return messages

    def _prepare_llm_for_graph(self) -> Any:
        """Entry-point LLM configuration (LangGraph best practice).

        Disables streaming for graph-internal LLM calls since the
        graph itself orchestrates multi-turn flow. Done ONCE at entry
        so nodes receive a clean, pre-configured LLM via config.
        """
        llm = self._llm
        if llm is None:
            return None
        if not getattr(llm, "streaming", False):
            return llm
        try:
            return llm.model_copy(update={"streaming": False})
        except Exception:
            return llm

    def _node_reason(self, state: ToolExecutorState, config: RunnableConfig) -> Command:
        cfg = config.get("configurable", {}) if config else {}
        cancel_event = cfg.get("cancel_event")
        token_tracker = cfg.get("token_tracker")

        if cancel_event and cancel_event.is_set():
            return Command(goto=END, update={
                "result": "⏹ 已停止",
                "status": "cancelled",
                "cancelled": True,
            })

        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", 10)

        if iteration >= max_iterations:
            messages = state.get("messages", [])
            if messages:
                last_msg = messages[-1]
                last_content = (
                    last_msg.get("content", "")
                    if isinstance(last_msg, dict)
                    else str(last_msg.content) if hasattr(last_msg, "content") else ""
                )
                return Command(goto=END, update={
                    "result": last_content or "(达到最大迭代次数)",
                    "status": "max_iterations",
                })
            return Command(goto=END, update={"result": "", "status": "max_iterations"})

        messages = state.get("messages", [])
        input_text = state.get("input_text", "")

        if not messages and input_text:
            from langchain_core.messages import HumanMessage
            messages = [HumanMessage(content=input_text)]

        if not messages:
            return Command(goto=END, update={
                "result": "无消息可处理",
                "status": "failed",
            })

        llm = cfg.get("llm")
        if llm is None:
            return Command(goto=END, update={
                "result": "LLM not configured",
                "status": "failed",
            })

        msg_debug = []
        for m in messages:
            cls_name = type(m).__name__
            mtype = getattr(m, "type", "?")
            content = getattr(m, "content", "")
            if isinstance(content, list):
                content = str(content)[:100]
            elif isinstance(content, str):
                content = content[:100]
            else:
                content = str(content)[:100]
            msg_debug.append(f"{cls_name}(type={mtype}, len={len(str(content))})")
        logger.info("[ToolExecGraph] Reason node: iter=%d, msgs=[%s]",
                     iteration, ", ".join(msg_debug))

        bound_llm = llm.bind_tools(self._tools)

        try:
            response = bound_llm.invoke(messages)

            if token_tracker:
                token_count = 0
                if hasattr(response, "response_metadata"):
                    token_count = response.response_metadata.get("token_usage", {}).get("total_tokens", 0)
                if token_count > 0:
                    token_tracker(token_count)

            progress_callback = cfg.get("progress_callback")
            ToolExecutorGraph._emit_thought(progress_callback, iteration, response)

            if not hasattr(response, "tool_calls") or not response.tool_calls:
                content = response.content if hasattr(response, "content") else str(response)
                if isinstance(content, list):
                    content = "".join(str(c) for c in content)
                return Command(goto=END, update={
                    "messages": [response],
                    "result": str(content),
                    "status": "completed",
                    "iteration": iteration + 1,
                })

            tool_calls = response.tool_calls
            if isinstance(tool_calls, list):
                tool_calls = [
                    {
                        "name": tc.get("name", ""),
                        "args": tc.get("args", tc.get("arguments", {})),
                        "id": tc.get("id", f"call_{i}"),
                    }
                    for i, tc in enumerate(tool_calls)
                ]

            return Command(goto="execute_tools", update={
                "messages": [response],
                "tool_calls": tool_calls,
                "iteration": iteration + 1,
            })

        except Exception as exc:
            logger.error("[ToolExecGraph] Reason failed: %s", exc)
            return Command(goto=END, update={
                "result": f"执行出错: {exc}",
                "status": "failed",
                "iteration": iteration + 1,
            })

    @staticmethod
    def _node_observe(state: ToolExecutorState, config: RunnableConfig) -> Command:
        cfg = config.get("configurable", {}) if config else {}
        progress_callback = cfg.get("progress_callback")

        tool_calls = state.get("tool_calls", [])

        if progress_callback and tool_calls:
            messages = state.get("messages", [])
            tool_results: dict[str, str] = {}
            for msg in messages:
                if hasattr(msg, "tool_call_id") and hasattr(msg, "content"):
                    tool_results[msg.tool_call_id] = str(msg.content)
                elif isinstance(msg, dict) and msg.get("role") == "tool":
                    tool_results[msg.get("tool_call_id", "")] = str(msg.get("content", ""))

            for tc in tool_calls:
                call_id = tc.get("id", "")
                output = tool_results.get(call_id, "")
                progress_callback({
                    "type": "tool",
                    "name": tc.get("name", ""),
                    "args": tc.get("args", {}),
                    "output": output,
                })

        return Command(goto="reason", update={
            "tool_calls": [],
            "tool_outputs": [],
            "status": "running",
        })

    @staticmethod
    def _ensure_langchain_messages(messages: list) -> list:
        from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

        result = []
        for msg in messages:
            if isinstance(msg, (HumanMessage, AIMessage, ToolMessage, SystemMessage)):
                result.append(msg)
            elif isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    result.append(HumanMessage(content=content))
                elif role == "assistant":
                    result.append(AIMessage(content=content))
                elif role == "tool":
                    result.append(ToolMessage(
                        content=content,
                        tool_call_id=msg.get("tool_call_id", ""),
                        name=msg.get("name", ""),
                    ))
                elif role == "system":
                    result.append(SystemMessage(content=content))
                else:
                    result.append(HumanMessage(content=str(content)))
            else:
                result.append(HumanMessage(content=str(msg)))
        return result

    @staticmethod
    def _emit_thought(
        progress_callback: Callable[[dict[str, Any]], None] | None,
        iteration: int,
        response: Any,
    ) -> None:
        if progress_callback:
            prefix = f"💭 [迭代 {iteration + 1}]: "
            content = ""
            if hasattr(response, "tool_calls") and response.tool_calls:
                tools_str = ", ".join(
                    tc.get("name", "") for tc in response.tool_calls
                )
                content = f"调用工具: {tools_str}"
            elif hasattr(response, "content"):
                content = str(response.content)[:200]
            progress_callback({
                "type": "log",
                "message": f"{prefix}{content}",
            })


def create_tool_executor(
    llm: Any = None,
    scene: str | None = None,
    checkpointer: Any | None = None,
) -> ToolExecutorGraph:
    """Factory function to create a stateless ToolExecutorGraph."""
    if scene:
        tools = list_scene_tools(scene)
    else:
        tools = list_tools()

    return ToolExecutorGraph(
        llm=llm,
        tools=tools,
        scene=scene,
        checkpointer=checkpointer,
    )