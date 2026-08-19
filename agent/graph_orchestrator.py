"""GraphOrchestrator — pure LangGraph Plan-and-Execute orchestration.

Architecture:

  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
  │ classify  │───→│ react_fast   │───→│ adaptive_chk │───→│   END    │
  └────┬─────┘    └──────────────┘    └──────┬───────┘    └──────────┘
       │ complex                              │ budget exceeded
       ▼                                      ▼
  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
  │   plan    │───→│ execute_step │───→│  verify      │───→│   END    │
  └──────────┘    └──────┬───────┘    └──────┬───────┘    └──────────┘
                          │ more steps        │ not complete
                          ▼                 ▼
                     check_steps         re-plan → plan

Thread safety:
  Uses threading.local() for per-run context. The compiled LangGraph
  is read-only after __init__, so it is safe to share across threads.
  Each run() call creates an independent _RunContext.

Cancellation:
  cancel_event is checked at every node entry and between every LLM token.
  On cancellation, nodes return immediately with a "cancelled" result.
  HTTP-level timeouts are configured via LLM provider settings.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agent.config import settings
from agent.llm.factory import create_llm
from agent.progress import make_log, make_plan, make_status, make_streaming_token
from agent.utils.retry import CancelledError
from agent.runner import (
    StreamingCallbackHandler,
    _BaseToolEventTracker,
    build_react_executor,
    run_react_step,
)
from agent.tools.registry import list_tools, ToolRouter

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    """Shared state across all LangGraph nodes."""

    user_input: str
    short_term_text: str
    chat_history_messages: list
    classification: str
    plan: list[str]
    current_step: int
    results: list[str]
    result: str
    is_done: bool
    plan_rounds: int
    graph_start_time: float
    fast_iterations: int
    fast_tool_calls: int
    fast_elapsed: float
    fast_hit_limit: bool
    complex_steps_executed: int
    complex_elapsed: float


@dataclass
class _RunContext:
    """Per-run context — stored in threading.local for thread safety."""

    cancel_event: threading.Event = field(default_factory=threading.Event)
    progress_cb: Callable[[dict[str, Any]], None] | None = None
    token_tracker: Callable[[int], None] | None = None
    llm: Any = None
    executor: Any = None
    scene_executors: dict[str, Any] = field(default_factory=dict)
    last_token_count: int = 0


class GraphOrchestrator:
    """Pure LangGraph Plan-and-Execute orchestrator.

    Thread-safe: the compiled graph is read-only after __init__.
    Per-run state (cancel event, progress callback, LLM) is stored
    in threading.local() and accessed via the _ctx property.
    """

    _tls = threading.local()

    def __init__(self) -> None:
        self._llm_provider = create_llm()
        self._llm = self._llm_provider.get_model()
        self._executor = build_react_executor(
            self._llm,
            max_iterations=settings.MAX_ITERATIONS,
            max_execution_time=settings.MAX_EXECUTION_TIME_SEC,
            verbose=settings.VERBOSE,
        )
        try:
            self._graph = self._build_graph()
        except Exception as exc:
            logger.error("[GraphOrch] Graph build failed: %s", exc, exc_info=True)
            self._graph = None
        logger.info(
            "[GraphOrch] LangGraph ready | LLM: %s",
            self._llm_provider.model_name,
        )

    @property
    def _ctx(self) -> _RunContext:
        """Get current thread's run context."""
        ctx = getattr(self._tls, "ctx", None)
        if ctx is None:
            raise RuntimeError("No active run context. run() must be called first.")
        return ctx

    @staticmethod
    def _is_cancelled() -> bool:
        """Check if the current run has been cancelled."""
        ctx = getattr(GraphOrchestrator._tls, "ctx", None)
        if ctx is None:
            return False
        return ctx.cancel_event.is_set()

    @staticmethod
    def _emit_log(message: str) -> None:
        """Emit a log event via the current run's progress callback."""
        ctx = getattr(GraphOrchestrator._tls, "ctx", None)
        if ctx and ctx.progress_cb:
            ctx.progress_cb(make_log(message))

    @staticmethod
    def _emit_plan(goal: str = "", steps: list[str] | None = None) -> None:
        """Emit a plan event via the current run's progress callback."""
        ctx = getattr(GraphOrchestrator._tls, "ctx", None)
        if ctx and ctx.progress_cb:
            ctx.progress_cb(make_plan(goal=goal, steps=steps or []))

    @staticmethod
    def _emit_status(message: str) -> None:
        """Emit a status event via the current run's progress callback."""
        ctx = getattr(GraphOrchestrator._tls, "ctx", None)
        if ctx and ctx.progress_cb:
            ctx.progress_cb(make_status(message))

    def request_stop(self) -> None:
        """Signal cancellation to the current run (called from any thread)."""
        ctx = getattr(self._tls, "ctx", None)
        if ctx:
            ctx.cancel_event.set()

    def _get_or_build_executor(self, scene: str | None = None) -> Any:
        """Get or create a scene-specific tool executor (per-run cache)."""
        ctx = self._ctx
        if not scene:
            return ctx.executor
        if scene not in ctx.scene_executors:
            ctx.scene_executors[scene] = build_react_executor(
                ctx.llm,
                max_iterations=settings.MAX_ITERATIONS,
                max_execution_time=settings.MAX_EXECUTION_TIME_SEC,
                verbose=settings.VERBOSE,
                scene=scene,
            )
        return ctx.scene_executors[scene]

    def _build_graph(self) -> Any:
        graph = StateGraph(AgentState)

        graph.add_node("classify", self._node_classify)
        graph.add_node("react_fast", self._node_react_fast)
        graph.add_node("adaptive_check", self._node_adaptive_check)
        graph.add_node("plan", self._node_plan)
        graph.add_node("execute_step", self._node_execute_step)
        graph.add_node("check_steps", self._node_check_steps)
        graph.add_node("verify", self._node_verify)
        graph.add_node("re_plan", self._node_re_plan)

        graph.add_edge(START, "classify")

        graph.add_conditional_edges(
            "classify",
            self._route_after_classify,
            {
                "simple": "react_fast",
                "complex": "plan",
            },
        )

        graph.add_edge("react_fast", "adaptive_check")

        graph.add_conditional_edges(
            "adaptive_check",
            self._route_after_adaptive,
            {
                "end": END,
                "upgrade": "plan",
            },
        )

        graph.add_edge("plan", "execute_step")
        graph.add_edge("execute_step", "check_steps")

        graph.add_conditional_edges(
            "check_steps",
            self._route_after_check,
            {
                "continue": "execute_step",
                "verify": "verify",
            },
        )

        graph.add_conditional_edges(
            "verify",
            self._route_after_verify,
            {
                "complete": END,
                "incomplete": "re_plan",
            },
        )

        graph.add_edge("re_plan", "execute_step")

        return graph.compile()

    def run(
        self,
        user_input: str,
        short_term_text: str,
        chat_history_messages: list,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_event: threading.Event | None = None,
        token_tracker: Callable[[int], None] | None = None,
    ) -> str:
        ctx = _RunContext(
            cancel_event=cancel_event or threading.Event(),
            progress_cb=progress_callback,
            token_tracker=token_tracker,
            llm=self._llm,
            executor=self._executor,
        )
        self._tls.ctx = ctx

        try:
            if self._graph is None:
                return "[错误] LangGraph 不可用，请检查日志。"

            self._emit_status("开始分析...")

            initial_state: AgentState = {
                "user_input": user_input,
                "short_term_text": short_term_text,
                "chat_history_messages": chat_history_messages,
                "classification": "",
                "plan": [],
                "current_step": 0,
                "results": [],
                "result": "",
                "is_done": False,
                "plan_rounds": 0,
                "graph_start_time": time.time(),
                "fast_iterations": 0,
                "fast_tool_calls": 0,
                "fast_elapsed": 0.0,
                "fast_hit_limit": False,
                "complex_steps_executed": 0,
                "complex_elapsed": 0.0,
            }

            try:
                final_state = self._graph.invoke(initial_state)
            except Exception as exc:
                logger.error("[GraphOrch] Graph error: %s", exc, exc_info=True)
                self._emit_log(f"❌ 执行出错: {exc}")
                return f"错误: {exc}"

            self._record_complex_downgrade_feedback(final_state)

            result = final_state.get("result", "")
            if not result:
                result = "已完成，但未产生文本输出。"

            return result

        finally:
            self._tls.ctx = None

    @staticmethod
    def _format_chat_history(messages: list) -> str:
        """Convert LangChain message list to readable transcript."""
        if not messages:
            return ""
        parts: list[str] = []
        for msg in messages[-10:]:
            role = getattr(msg, "type", "") or ""
            content = getattr(msg, "content", "") or ""
            if isinstance(content, list):
                content = " ".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                )
            if role == "human":
                parts.append(f"用户: {content}")
            elif role == "ai":
                parts.append(f"助手: {content}")
            elif role == "system":
                parts.append(f"系统: {content[:200]}")
        return "\n".join(parts)

    @staticmethod
    def _build_tools_description() -> str:
        """Dynamically build tool description from the registry."""
        tools = list_tools()
        if not tools:
            return "(无可用工具)"
        lines = ["可用工具:"]
        for t in tools:
            name = getattr(t, "name", "未知")
            desc = (getattr(t, "description", "") or "")[:120]
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)

    def _execute_step_with_tools(
        self,
        input_text: str,
        chat_history_messages: list,
        scene: str | None = None,
    ) -> dict[str, Any]:
        """Run a single ReAct tool-calling step with cancellation support."""
        ctx = self._ctx

        if not scene and input_text:
            scene, _ = ToolRouter().smart_route(input_text)

        executor = self._get_or_build_executor(scene)

        tracker = _BaseToolEventTracker(ctx.progress_cb)
        streaming_handler = StreamingCallbackHandler(
            ctx.progress_cb,
            cancel_event=ctx.cancel_event,
            token_tracker=ctx.token_tracker,
        )

        try:
            result = run_react_step(
                executor,
                input_text,
                chat_history=chat_history_messages,
                progress_callback=ctx.progress_cb,
                tracker=tracker,
                cancel_event=ctx.cancel_event,
                extra_callbacks=[streaming_handler],
            )
        except Exception as exc:
            logger.error("[GraphOrch] Tool execution failed: %s", exc)
            self._emit_log(f"⚠️ 工具执行失败: {exc}")
            return {
                "result": f"工具执行出错: {exc}",
                "status": "error",
                "iterations": 0,
                "time": 0.0,
                "hit_limit": False,
                "tool_calls": 0,
                "llm_calls": 0,
            }

        if ctx.cancel_event.is_set():
            return {
                "result": "⏹ 已停止",
                "status": "cancelled",
                "iterations": 0,
                "time": 0.0,
                "hit_limit": False,
                "tool_calls": 0,
                "llm_calls": 0,
            }

        enriched = dict(result)
        enriched["tool_calls"] = tracker._call_counter
        enriched["llm_calls"] = tracker._llm_call_count
        return enriched

    def _record_complex_downgrade_feedback(self, final_state: AgentState) -> None:
        """After graph completion: learn if complex queries could have been simple."""
        from agent.utils.classifier import record_feedback

        classification = final_state.get("classification", "")
        if classification != "complex":
            return

        graph_start = final_state.get("graph_start_time", time.time())
        total_elapsed = time.time() - graph_start
        steps_executed = final_state.get("complex_steps_executed", 0)

        user_input = final_state.get("user_input", "")
        record_feedback(
            user_input=user_input,
            original_label="complex",
            steps_executed=steps_executed,
            elapsed=total_elapsed,
        )

    # ------------------------------------------------------------------
    # Node implementations
    # ------------------------------------------------------------------

    def _node_classify(self, state: AgentState) -> dict[str, Any]:
        """Classify query complexity — simple (fast path) or complex (plan+execute)."""
        from agent.utils.classifier import classify_query

        if self._is_cancelled():
            return {"result": "⏹ 已停止", "is_done": True}

        classification = classify_query(self._ctx.llm, state["user_input"])

        self._emit_log(f"🔍 查询分类: {classification}")

        return {"classification": classification}

    def _node_react_fast(self, state: AgentState) -> dict[str, Any]:
        """Execute a simple query via the LangChain ReAct fast path with budget tracking."""
        if self._is_cancelled():
            return {"result": "⏹ 已停止", "is_done": True}

        self._emit_log("⚡ 执行快速路径...")

        exec_result = self._execute_step_with_tools(
            state["user_input"],
            state.get("chat_history_messages", []),
        )

        result_text = exec_result.get("result", "")
        if exec_result.get("status") == "cancelled":
            result_text = "⏹ 已停止"

        return {
            "result": result_text,
            "is_done": True,
            "fast_iterations": exec_result.get("iterations", 0),
            "fast_tool_calls": exec_result.get("tool_calls", 0),
            "fast_elapsed": exec_result.get("time", 0.0),
            "fast_hit_limit": exec_result.get("hit_limit", False),
        }

    def _node_adaptive_check(self, state: AgentState) -> dict[str, Any]:
        """After fast path: check if budget was exceeded and upgrade to complex if needed."""
        from agent.utils.classifier import record_feedback, get_budget

        if self._is_cancelled():
            return {"result": "⏹ 已停止", "is_done": True}

        classification = state.get("classification", "simple")
        iterations = state.get("fast_iterations", 0)
        tool_calls = state.get("fast_tool_calls", 0)
        elapsed = state.get("fast_elapsed", 0.0)
        hit_limit = state.get("fast_hit_limit", False)

        if classification != "simple":
            return {}

        budget = get_budget()
        exceeded = budget.exceeded_simple_budget(iterations, tool_calls, elapsed)

        if hit_limit or exceeded:
            reason_parts = []
            if hit_limit:
                reason_parts.append("达到迭代/时间上限")
            if iterations >= budget.MAX_SIMPLE_ITERATIONS:
                reason_parts.append(f"迭代次数={iterations}>={budget.MAX_SIMPLE_ITERATIONS}")
            if tool_calls >= budget.MAX_SIMPLE_TOOL_CALLS:
                reason_parts.append(f"工具调用={tool_calls}>={budget.MAX_SIMPLE_TOOL_CALLS}")
            if elapsed >= budget.MAX_SIMPLE_TIME_SEC:
                reason_parts.append(f"耗时={elapsed:.1f}s>={budget.MAX_SIMPLE_TIME_SEC}s")
            reason = ", ".join(reason_parts) if reason_parts else "超出预算"
            self._emit_log(
                f"⚠️ 快速路径预算已超 ({reason}) → 升级到复杂路径"
            )

            user_input = state.get("user_input", "")
            record_feedback(
                user_input=user_input,
                original_label="simple",
                iterations=iterations,
                tool_calls=tool_calls,
                elapsed=elapsed,
            )

            return {
                "classification": "complex",
                "result": state.get("result", ""),
                "is_done": False,
            }

        logger.info(
            "[Adaptive] Fast path OK: iter=%d tools=%d time=%.1fs",
            iterations, tool_calls, elapsed,
        )
        return {}

    def _node_plan(self, state: AgentState) -> dict[str, Any]:
        """Create a step-by-step execution plan via LLM."""
        if self._is_cancelled():
            return {"result": "⏹ 已停止", "is_done": True}

        plan_rounds = state.get("plan_rounds", 0)

        if plan_rounds == 0:
            self._emit_status("📋 正在创建执行计划...")
        else:
            self._emit_status(f"🔄 重新规划 (第 {plan_rounds + 1} 轮)...")

        user_input = state.get("user_input", "")
        previous_plan = state.get("plan", [])
        previous_results = state.get("results", [])
        chat_history_text = self._format_chat_history(state.get("chat_history_messages", []))
        tools_desc = self._build_tools_description()

        context_parts = [f"用户目标: {user_input}"]

        if chat_history_text:
            context_parts.append(f"对话历史:\n{chat_history_text}")

        if previous_plan:
            context_parts.append("之前的计划:")
            for i, step in enumerate(previous_plan):
                status = previous_results[i] if i < len(previous_results) else "(未执行)"
                context_parts.append(f"  步骤 {i + 1}: {step} → {status[:200]}")

        context = "\n".join(context_parts)

        prompt = f"""{context}

{tools_desc}

请创建一个清晰的、分步的执行计划来实现用户目标。
将其分解为 2-5 个具体的、可执行的步骤。每一步应该是一个单独的工具调用或简单操作。

重要：您必须使用中文回复。所有步骤描述必须使用中文。

请按以下格式输出，每行一个步骤，带编号:
1. [步骤描述 — 具体说明要做什么]
2. [步骤描述]
...

只输出编号步骤，不要输出其他内容。"""

        try:
            plan_text = self._stream_llm_response(prompt)
        except Exception as exc:
            logger.error("[GraphOrch] Plan generation failed: %s", exc)
            self._emit_log("⚠️ 计划生成失败，使用备用计划")
            plan_text = f"1. 分析请求: {user_input}\n2. 执行所需操作\n3. 总结结果"

        plan = self._parse_plan(plan_text)

        if not plan:
            plan = [
                f"分析请求: {user_input}",
                "执行所需操作",
                "总结结果",
            ]

        self._emit_log(f"📋 计划已创建: {len(plan)} 个步骤")
        self._emit_plan(goal=user_input, steps=plan)
        for i, step in enumerate(plan):
            self._emit_log(f"  {i + 1}. {step[:120]}")

        return {
            "plan": plan,
            "current_step": 0,
            "results": [],
        }

    def _node_execute_step(self, state: AgentState) -> dict[str, Any]:
        """Execute the current plan step using the ReAct executor."""
        if self._is_cancelled():
            return {"result": "⏹ 已停止", "is_done": True}

        current_step = state.get("current_step", 0)
        plan = state.get("plan", [])

        if current_step >= len(plan):
            return {}

        step_description = plan[current_step]

        self._emit_log(
            f"▶️ 步骤 {current_step + 1}/{len(plan)}: {step_description[:120]}"
        )

        user_input = state.get("user_input", "")
        short_term_text = state.get("short_term_text", "")
        chat_history = state.get("chat_history_messages", [])

        context_sections = []
        if short_term_text:
            context_sections.append(f"之前的上下文:\n{short_term_text[:800]}")

        step_prompt = f"""执行此步骤以实现用户目标。

用户原始目标: {user_input}
当前步骤: {step_description}
{chr(10).join(context_sections)}

使用可用工具执行所需操作。如果需要读取文件、搜索网页或运行代码，请立即使用相应工具。

重要：您必须使用中文回复。用中文思考和回复。

完成此步骤后，简要总结您做了什么以及发现了什么。"""

        result = self._execute_step_with_tools(step_prompt, chat_history)

        result_text = result.get("result", "") if isinstance(result, dict) else str(result)
        result_text = self._truncate_output(result_text)

        results = list(state.get("results", []))
        while len(results) <= current_step:
            results.append("")
        results[current_step] = result_text

        complex_steps = state.get("complex_steps_executed", 0) + 1

        self._emit_log(f"✅ 步骤 {current_step + 1} 完成: {result_text[:200]}")

        return {
            "results": results,
            "current_step": current_step + 1,
            "complex_steps_executed": complex_steps,
        }

    def _node_check_steps(self, state: AgentState) -> dict[str, Any]:
        """Check if all plan steps are completed."""
        plan = state.get("plan", [])
        current_step = state.get("current_step", 0)

        if current_step >= len(plan):
            self._emit_log("📋 所有计划步骤已完成")

        return {}

    def _node_verify(self, state: AgentState) -> dict[str, Any]:
        """Verify whether the goal is achieved and produce final result."""
        user_input = state.get("user_input", "")
        results = state.get("results", [])
        plan = state.get("plan", [])
        plan_rounds = state.get("plan_rounds", 0)
        chat_history_text = self._format_chat_history(state.get("chat_history_messages", []))

        if self._is_cancelled():
            return {"result": "⏹ 已停止", "is_done": True}

        if not results:
            return {"is_done": True}

        if plan_rounds >= settings.MAX_PLAN_ROUNDS:
            logger.info("[GraphOrch] Max re-plan rounds (%d) reached, ending", plan_rounds)
            self._emit_log("⚠️ 达到最大重新规划次数，结束执行")
            return {"is_done": True}

        self._emit_log("🔍 正在验证目标达成情况...")

        summary_parts = ["## 执行结果\n"]
        for i, (step, result) in enumerate(zip(plan, results)):
            summary_parts.append(f"**步骤 {i + 1}**: {step}")
            summary_parts.append(f"结果: {result[:300]}")
            summary_parts.append("")

        full_summary = "\n".join(summary_parts)

        context_block = ""
        if chat_history_text:
            context_block = f"\n\n对话上下文:\n{chat_history_text[:600]}"

        prompt = f"""{full_summary}{context_block}

用户原始目标: {user_input}

基于以上执行结果，判断用户目标是否已完全实现？

重要：您必须使用中文回复。

请明确回答"是"或"否"，然后用中文提供简要解释。
如果回答"是"，请同时用中文为用户提供简洁的最终答案。
如果回答"否"，请用中文解释还缺少什么。"""

        try:
            verification = self._stream_llm_response(prompt)

            self._emit_log(f"🔍 验证结果: {verification[:200]}")

            is_complete = verification.strip().lower().startswith(("yes", "是"))

            if is_complete:
                final_answer = self._extract_final_answer(verification, user_input, results)
                self._emit_log("🎯 目标已达成!")
                return {
                    "result": final_answer,
                    "is_done": True,
                }

            if plan_rounds >= 2:
                self._emit_log("⚠️ 达到最大重新规划次数，接受当前结果")
                final_answer = self._extract_final_answer(verification, user_input, results)
                return {
                    "result": final_answer,
                    "is_done": True,
                }

            self._emit_log("🔍 目标未完全达成，需要重新规划...")

            return {
                "result": verification,
                "is_done": False,
            }

        except Exception as exc:
            logger.warning("[GraphOrch] Verification failed: %s", exc)
            self._emit_log("⚠️ 验证失败，基于已有结果继续")
            final_answer = self._extract_final_answer(
                f"验证失败: {exc}", user_input, results
            )
            return {
                "result": final_answer,
                "is_done": True,
            }

    def _node_re_plan(self, state: AgentState) -> dict[str, Any]:
        """Re-plan remaining steps after verification found gaps."""
        if self._is_cancelled():
            return {"result": "⏹ 已停止", "is_done": True}

        plan_rounds = state.get("plan_rounds", 0)
        new_plan_rounds = plan_rounds + 1

        self._emit_status(f"📋 重新规划 (第 {new_plan_rounds} 轮)...")

        user_input = state.get("user_input", "")
        results = state.get("results", [])
        plan = state.get("plan", [])
        verification_feedback = state.get("result", "")
        chat_history_text = self._format_chat_history(state.get("chat_history_messages", []))

        plan_lines = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(plan))
        result_lines = "\n".join(f"步骤 {i + 1}: {r[:200]}" for i, r in enumerate(results))

        prompt = f"""原始目标: {user_input}

之前的计划:
{plan_lines}

之前的结果:
{result_lines}

验证反馈: {verification_feedback[:500]}
{chr(10).join('对话上下文:' if chat_history_text else '')}
{chat_history_text[:400] if chat_history_text else ''}

请创建一个新的执行计划来完成剩余工作。
只列出仍需完成的步骤。

重要：您必须使用中文回复。所有步骤描述必须使用中文。

请只输出编号步骤。"""

        try:
            plan_text = self._stream_llm_response(prompt)
            new_plan = self._parse_plan(plan_text)
        except Exception as exc:
            logger.error("[GraphOrch] Re-plan failed: %s", exc)
            self._emit_log("⚠️ 重新规划失败，使用简单替代方案")
            new_plan = [f"完成剩余工作: {user_input}"]

        if not new_plan:
            new_plan = [f"完成剩余工作: {user_input}"]

        self._emit_log(f"📋 新计划: {len(new_plan)} 个步骤")
        for i, step in enumerate(new_plan):
            self._emit_log(f"  {i + 1}. {step[:120]}")

        return {
            "plan": new_plan,
            "current_step": 0,
            "results": [],
            "plan_rounds": new_plan_rounds,
        }

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _stream_llm_response(self, prompt: str, emit_tokens: bool = False) -> str:
        """Stream LLM response with cancellation support.

        Timeout is handled by the LLM provider's HTTP client (timeout config).
        Cancellation is checked between every token for responsive stopping.
        """
        ctx = self._ctx
        collected: list[str] = []
        token_count = 0
        try:
            for chunk in ctx.llm.stream(prompt):
                if ctx.cancel_event.is_set():
                    logger.info("[GraphOrch] Stream cancelled by user")
                    break
                token = chunk.content
                if token:
                    collected.append(token)
                    token_count += 1
                    if emit_tokens and ctx.progress_cb:
                        ctx.progress_cb(make_streaming_token(token))
        except Exception as exc:
            logger.warning("[GraphOrch] Stream error: %s", exc)
            self._emit_log(f"⚠️ 流式输出中断: {exc}")
        ctx.last_token_count = token_count
        if ctx.token_tracker and token_count > 0:
            ctx.token_tracker(token_count)
        return "".join(collected)

    def _truncate_output(self, text: str) -> str:
        """Truncate long tool outputs to avoid context bloat."""
        limit = settings.TOOL_OUTPUT_TRUNCATE
        if len(text) <= limit:
            return text
        return text[:limit] + f"\n... [输出已截断，共 {len(text)} 字符]"

    @staticmethod
    def _parse_plan(plan_text: str) -> list[str]:
        """Parse a numbered plan from LLM output."""
        steps: list[str] = []
        for line in plan_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line[0].isdigit():
                for sep in (".", "、", ")", ":", "："):
                    if sep in line:
                        idx = line.index(sep)
                        step_text = line[idx + 1:].strip()
                        if step_text:
                            steps.append(step_text)
                        break
                else:
                    steps.append(line)
        return steps

    @staticmethod
    def _extract_final_answer(
        verification: str, user_input: str, results: list[str]
    ) -> str:
        """Extract a clean final answer from verification output."""
        lines = verification.strip().split("\n")

        answer_lines: list[str] = []
        found_yes = False

        for line in lines:
            stripped = line.strip()
            if not found_yes and (
                stripped.lower().startswith("yes")
                or stripped.startswith("是")
            ):
                found_yes = True
                remainder = stripped[3:].strip(".,;:：,。； ") if stripped.lower().startswith("yes") else stripped[1:].strip("，。； ")
                if remainder and len(remainder) > 5:
                    answer_lines.append(remainder)
                continue
            if found_yes and stripped:
                answer_lines.append(stripped)

        if answer_lines:
            return "\n".join(answer_lines)

        if results:
            return "\n\n".join(
                f"**结果 {i + 1}**: {r[:500]}" for i, r in enumerate(results)
            )

        return f"任务已完成: {user_input}"

    # ------------------------------------------------------------------
    # Routing logic
    # ------------------------------------------------------------------

    @staticmethod
    def _route_after_classify(state: AgentState) -> str:
        return state.get("classification", "complex")

    @staticmethod
    def _route_after_adaptive(state: AgentState) -> str:
        if state.get("classification") == "complex":
            return "upgrade"
        return "end"

    @staticmethod
    def _route_after_check(state: AgentState) -> str:
        plan = state.get("plan", [])
        current_step = state.get("current_step", 0)
        if current_step < len(plan):
            return "continue"
        return "verify"

    @staticmethod
    def _route_after_verify(state: AgentState) -> str:
        return "complete" if state.get("is_done") else "incomplete"