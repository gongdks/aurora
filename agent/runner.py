"""共享工具调用执行器 —— executor 和 orchestrator 共用。

封装 LangChain tool-calling agent 的创建、调用、错误处理逻辑。
使用原生函数调用（native tool calling）而非文本 ReAct 格式。
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

from agent.config import settings
from agent.progress import make_log, make_streaming_token, make_tool
from agent.prompts import AGENT_PROMPT
from agent.tools.registry import list_tools
from agent.utils.retry import CancelledError

logger = logging.getLogger(__name__)


class _BaseToolEventTracker(BaseCallbackHandler):
    """基础工具调用追踪器 —— 供 executor/orchestrator 复用。

    在原生 tool-calling agent 模式下，on_agent_action 不会被触发。
    改用 on_llm_start / on_tool_start / on_tool_end 追踪执行过程。
    """

    MAX_LOG_ENTRIES = 200

    def __init__(self, progress_callback: Callable[[dict[str, Any]], None] | None) -> None:
        super().__init__()
        self._cb = progress_callback
        self._current_tool: str | None = None
        self._call_counter: int = 0
        self.thoughts: list[str] = []
        self.tool_calls: list[dict[str, Any]] = []
        self._step_label: str = ""
        self._llm_call_count: int = 0

    def _add_thought(self, thought: str) -> None:
        self.thoughts.append(thought)
        if len(self.thoughts) > self.MAX_LOG_ENTRIES:
            self.thoughts = self.thoughts[-self.MAX_LOG_ENTRIES:]

    def _add_tool_call(self, entry: dict[str, Any]) -> None:
        self.tool_calls.append(entry)
        if len(self.tool_calls) > self.MAX_LOG_ENTRIES:
            self.tool_calls = self.tool_calls[-self.MAX_LOG_ENTRIES:]

    def set_step_label(self, label: str) -> None:
        self._step_label = label

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        """LLM 开始调用时发送思考状态。"""
        self._llm_call_count += 1
        if self._cb:
            prefix = f"💭 {self._step_label}: " if self._step_label else "💭 "
            self._cb(make_log(f"{prefix}Thinking..."))

    def on_agent_action(self, action: Any, **kwargs: Any) -> None:
        """Legacy ReAct agent action callback — kept for backward compatibility."""
        if not self._cb:
            return
        log_text: str = getattr(action, "log", "")
        if not log_text:
            return
        thought: str = log_text
        if "Action:" in log_text:
            thought = log_text.split("Action:")[0].strip()
        if "Thought:" in thought:
            thought = thought.split("Thought:", 1)[1].strip()
        thought = thought.rstrip()
        if thought:
            self._add_thought(thought)
            prefix = f"💭 {self._step_label}: " if self._step_label else "💭 "
            self._cb(make_log(f"{prefix}{thought}"))

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> None:
        self._call_counter += 1
        tool_name = serialized.get("name", "unknown")
        self._current_tool = tool_name
        if self._cb:
            self._cb(make_tool(tool_name, input_str, call_id=self._call_counter))

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        if self._current_tool and self._cb:
            self._cb(make_tool(
                self._current_tool, "",
                output=output, call_id=self._call_counter,
            ))
            self._add_tool_call({
                "tool": self._current_tool,
                "output": output[:500],
            })
        self._current_tool = None

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        if self._current_tool and self._cb:
            self._cb(make_tool(
                self._current_tool, "",
                output=f"[ERR] {error}", call_id=self._call_counter,
            ))
        self._current_tool = None


class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler that forwards LLM streaming tokens as progress events.

    When streaming is enabled on the LLM, LangChain calls on_llm_new_token
    for each token as it arrives. This handler converts those tokens into
    STREAMING_TOKEN progress events so the UI can render them in near-real-time.
    """

    def __init__(self, progress_callback: Callable[[dict[str, Any]], None] | None) -> None:
        super().__init__()
        self._cb = progress_callback
        self._buffer: list[str] = []

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if self._cb:
            self._cb(make_streaming_token(token))


def build_react_executor(
    llm: BaseChatModel,
    max_iterations: int | None = None,
    max_execution_time: int | None = None,
    return_intermediate_steps: bool = False,
    verbose: bool = False,
) -> AgentExecutor:
    """构建一个 tool-calling AgentExecutor 实例（原生函数调用）。

    Args:
        llm: 语言模型实例
        max_iterations: 最大迭代次数，None 使用全局默认
        max_execution_time: 最大执行时间，None 使用全局默认
        return_intermediate_steps: 是否返回中间步骤
        verbose: 是否启用详细日志

    Returns:
        配置好的 AgentExecutor
    """
    tools = list_tools()
    agent = create_tool_calling_agent(llm, tools, AGENT_PROMPT)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        max_iterations=max_iterations or settings.MAX_ITERATIONS,
        max_execution_time=max_execution_time or settings.MAX_EXECUTION_TIME_SEC,
        handle_parsing_errors=True,
        return_intermediate_steps=return_intermediate_steps,
    )


def run_react_step(
    executor: AgentExecutor,
    input_text: str,
    chat_history: list | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    tracker: _BaseToolEventTracker | None = None,
    cancel_event: threading.Event | None = None,
    extra_callbacks: list | None = None,
) -> dict[str, Any]:
    """Execute a single tool-calling step with full error handling.

    Args:
        executor: Configured AgentExecutor
        input_text: Input text (step description or user request)
        chat_history: LangChain message list (HumanMessage/AIMessage)
        progress_callback: Progress callback function
        tracker: Tool call tracker
        cancel_event: Cancel event, set to abort execution
        extra_callbacks: Additional LangChain callbacks (e.g. streaming handler)

    Returns:
        Dict with status, result, iterations, time, hit_limit
    """
    invoke_kwargs: dict[str, Any] = {
        "input": input_text,
        "chat_history": chat_history or [],
    }
    callbacks_list: list = []
    if tracker is not None:
        callbacks_list.append(tracker)
    if extra_callbacks:
        callbacks_list.extend(extra_callbacks)
    if callbacks_list:
        invoke_kwargs["config"] = {"callbacks": callbacks_list}

    start_time = time.time()

    try:
        resp = executor.invoke(invoke_kwargs)
        elapsed = time.time() - start_time
        result = resp.get("output", "(no output)")
        iterations = len(resp.get("intermediate_steps", []))

        logger.info("[Run] Completed in %.1fs, %d iterations", elapsed, iterations)
        return {
            "status": "completed",
            "result": result,
            "iterations": iterations,
            "time": elapsed,
            "hit_limit": False,
        }

    except ValueError as exc:
        err_msg = str(exc)
        elapsed = time.time() - start_time
        if "iteration limit" in err_msg.lower() or "time limit" in err_msg.lower():
            logger.warning("[Run] Hit limit: %s", err_msg)
            return {
                "status": "completed",
                "result": f"(partial - hit limit) {err_msg}",
                "iterations": 0,
                "time": elapsed,
                "hit_limit": True,
            }
        logger.error("[Run] ValueError: %s", exc)
        return {
            "status": "failed",
            "result": "",
            "error": str(exc),
            "hit_limit": False,
        }

    except CancelledError:
        elapsed = time.time() - start_time
        logger.info("[Run] Cancelled by user after %.1fs", elapsed)
        return {
            "status": "cancelled",
            "result": "⏹ 已停止",
            "iterations": 0,
            "time": elapsed,
            "hit_limit": False,
        }

    except (TimeoutError, ConnectionError) as exc:
        elapsed = time.time() - start_time
        logger.error("[Run] %s: %s", type(exc).__name__, exc)
        return {
            "status": "failed",
            "result": "",
            "error": f"{type(exc).__name__}: {exc}",
            "hit_limit": False,
        }

    except Exception as exc:
        elapsed = time.time() - start_time
        logger.error("[Run] Failed: %s", exc, exc_info=True)
        return {
            "status": "failed",
            "result": "",
            "error": str(exc),
            "hit_limit": False,
        }