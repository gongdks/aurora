"""ToolExecutorGraph — pure LangGraph tool execution engine.

Replaces the procedural runner.py (LangChain AgentExecutor) with a
LangGraph StateGraph + ToolNode approach for native tool execution.

Architecture (LangGraph StateGraph + ToolNode):

  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │  reason  │───→│ tool_node│───→│  observe │───→│   END    │
  └────┬─────┘    └──────────┘    └────┬─────┘    └──────────┘
       │ no tools                       │ more tools needed
       ▼                                ▼
  ┌──────────┐                   ┌──────────┐
  │   END    │                   │  reason  │
  └──────────┘                   └──────────┘

Key LangGraph features leveraged:
  1. ToolNode for native tool execution (no AgentExecutor)
  2. StateGraph with TypedDict state for clean data flow
  3. Conditional edges for reason/act/observe loop
  4. SqliteSaver for execution state persistence
  5. Send API ready for future parallel tool execution
  6. Proper cancellation via configurable interrupts
"""

from __future__ import annotations

import logging
import time
import threading
from collections.abc import Callable
from typing import Any, TypedDict

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
    messages: list
    tools: list
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

    Uses ToolNode for native tool execution with full checkpoint
    and streaming support. Runtime params (cancel_event,
    progress_callback, token_tracker) are passed via config
    rather than constructor — enabling stateless, thread-safe
    executor reuse across runs.

    Usage:
        executor = ToolExecutorGraph(llm=my_llm, tools=my_tools)
        result = executor.execute(
            input_text="分析这份数据",
            chat_history=[],
            cancel_event=my_event,
            progress_callback=my_cb,
        )
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

        self._graph = self._build_graph(checkpointer)

    def _build_graph(self, checkpointer: Any | None = None) -> Any:
        graph = StateGraph(ToolExecutorState)

        graph.add_node("reason", self._node_reason)
        graph.add_node("execute_tools", ToolNode(self._tools))
        graph.add_node("observe", self._node_observe)

        graph.add_edge(START, "reason")
        graph.add_edge("execute_tools", "observe")

        if checkpointer:
            return graph.compile(checkpointer=checkpointer)
        return graph.compile()

    def execute(
        self,
        input_text: str,
        chat_history: list | None = None,
        tools: list | None = None,
        extra_tools: list | None = None,
        cancel_event: threading.Event | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        token_tracker: Callable[[int], None] | None = None,
    ) -> dict[str, Any]:
        """Execute the tool-calling loop via LangGraph.

        Runtime params are passed via config["configurable"] so the
        executor instance remains stateless and thread-safe.

        Args:
            input_text: The user query or task description.
            chat_history: LangChain message list.
            tools: Override default tools for this execution.
            extra_tools: Additional tools to append.
            cancel_event: Optional cancellation event.
            progress_callback: Optional progress event callback.
            token_tracker: Optional token usage tracker.

        Returns:
            dict with result, iterations, time, status, tool_calls.
        """
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

        active_tools = tools or self._tools
        if extra_tools:
            active_tools = list(active_tools) + list(extra_tools)

        messages = list(chat_history or [])

        initial_state: ToolExecutorState = {
            "input_text": input_text,
            "chat_history": chat_history or [],
            "messages": messages,
            "tools": active_tools,
            "tool_calls": [],
            "tool_outputs": [],
            "iteration": 0,
            "max_iterations": self._max_iterations,
            "status": "running",
            "result": "",
            "total_time": 0.0,
            "cancelled": False,
        }

        config = {
            "configurable": {
                "thread_id": f"tool_exec_{id(self)}",
                "cancel_event": cancel_event,
                "progress_callback": progress_callback,
                "token_tracker": token_tracker,
                "llm": self._llm,
            }
        }

        start_time = time.time()

        try:
            final = self._graph.invoke(initial_state, config=config)
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

    # ------------------------------------------------------------------
    # Graph nodes (read runtime params from config, not self)
    # ------------------------------------------------------------------

    @staticmethod
    def _node_reason(state: ToolExecutorState, config: Any) -> Command:
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

        if not messages:
            messages = [{"role": "user", "content": input_text}]

        messages = ToolExecutorGraph._ensure_langchain_messages(messages)

        tools = state.get("tools", [])
        llm = cfg.get("llm")
        if llm is None:
            return Command(goto=END, update={
                "result": "LLM not configured",
                "status": "failed",
            })

        bound_llm = llm.bind_tools(tools)

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
                    "messages": messages + [response],
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
                "messages": messages + [response],
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
    def _node_observe(state: ToolExecutorState, config: Any) -> Command:
        cfg = config.get("configurable", {}) if config else {}
        progress_callback = cfg.get("progress_callback")

        messages = state.get("messages", [])
        tool_calls = state.get("tool_calls", [])
        tool_outputs = state.get("tool_outputs", [])

        if not tool_calls:
            return Command(goto=END, update={"status": "completed"})

        new_messages = list(messages)
        for tc in tool_calls:
            call_id = tc.get("id", "")
            output = ""
            for out in tool_outputs:
                if isinstance(out, dict) and out.get("tool_call_id") == call_id:
                    output = out.get("content", "")
                    break
            if not output:
                output = "(no output)"

            new_messages.append({
                "tool_call_id": call_id,
                "role": "tool",
                "name": tc.get("name", ""),
                "content": str(output),
            })

        if progress_callback:
            for tc in tool_calls:
                progress_callback({
                    "type": "tool",
                    "tool": tc.get("name", ""),
                    "args": tc.get("args", {}),
                    "output": next(
                        (out.get("content", "") for out in tool_outputs
                         if isinstance(out, dict) and out.get("tool_call_id") == tc.get("id", "")),
                        "",
                    ),
                })

        return Command(goto="reason", update={
            "messages": new_messages,
            "tool_calls": [],
            "tool_outputs": [],
            "status": "running",
        })

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
    """Factory function to create a stateless ToolExecutorGraph.

    Runtime params (cancel_event, progress_callback, token_tracker)
    are passed to execute() at call time, not to the factory.
    """
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