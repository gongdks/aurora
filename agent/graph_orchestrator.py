"""GraphOrchestrator — pure LangGraph Plan-and-Execute orchestration.

Architecture (LangGraph StateGraph + Checkpointer + Command + Send + Subgraphs):

  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
  │ classify  │───→│ react_fast   │───→│ adaptive_chk │───→│   END    │
  └────┬─────┘    └──────┬───────┘    └──────┬───────┘    └──────────┘
       │ complex         │                    │ upgrade
       ▼                 ▼                    ▼
  ┌──────────┐    ┌──────────────┐    ┌──────────────┐
  │   plan    │───→│  dispatch    │───→│ execute_step │ ──┐
  └──────────┘    └──────┬───────┘    └──────┬───────┘   │
                   Send(fan-out)             │            │
                        │                    │            │
                        ▼                    ▼            ▼
                   ┌──────────────┐    ┌──────────────────────┐
                   │  verify       │←──│  check_steps (fan-in) │
                   └──────┬───────┘    └──────────────────────┘
                          │                    │
                          │         pending>0 → END (wait)
                          │         pending=0 → dispatch / verify
                          ▼
                   ┌──────────────┐    complete ┌──────┐
                   │  reflect     │────────────→│ END  │
                   └──────┬───────┘             └──────┘
                          │ replan
                          ▼
                     ┌──────────┐
                     │ re_plan  │──→ dispatch
                     └──────────┘

Subgraphs (shared checkpointer):
  • ToolExecutorGraph  — Tool execution with reason/act/observe loop
  • ReflectionEngine   — Multi-layered self-evaluation + strategy adjustment

Key LangGraph features leveraged:
  1. StateGraph with proper TypedDict state + Annotated reducers
  2. SQLite shared checkpointer for state persistence across subgraphs
  3. Command(goto=...) for clean loop control
  4. Send API for true parallel fan-out/fan-in of plan steps
  5. Subgraph composition with shared checkpoints (ToolExecutorGraph, ReflectionEngine)
  6. Config-based runtime context (no threading.local())
  7. Streaming via stream() API with native stream modes
  8. Checkpoint resume capability
  9. ToolNode-based tool execution engine

Thread safety:
  Runtime context is passed via LangGraph's config mechanism,
  not threading.local(). Each invoke() call gets an independent
  config dict, so concurrent runs are safe without TLS.

Cancellation:
  cancel_event is stored in config["configurable"] and checked
  at every node entry. On cancellation, nodes return immediately
  with a "cancelled" result. request_stop() sets the instance-level
  cancel event reference for external callers.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable
from typing import Annotated, Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, Send

from agent.config import settings
from agent.llm.factory import create_llm
from agent.progress import make_log, make_plan, make_status
from agent.utils.cache import get_cache
from agent.utils.classifier import classify_query, record_feedback, get_budget
from agent.utils.reflection import ReflectionEngine, ReflectionResult, ReflectionScore
from agent.utils.skill_learner_graph import SkillLearnerGraph
from agent.utils.skill_store import SkillStore
from agent.utils.autonomous_graph import AutonomousGraph, Goal
from agent.utils.multi_agent_graph import MultiAgentGraph, create_default_roles
from agent.utils.retry import CancelledError
from agent.utils.tool_executor_graph import ToolExecutorGraph
from agent.tools.registry import list_tools

logger = logging.getLogger(__name__)

_CHECKPOINT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".langgraph_checkpoints",
)


def _merge_results(existing: list, updates: list) -> list:
    if not updates:
        return []
    merged = list(existing)
    for item in updates:
        if isinstance(item, tuple) and len(item) == 2:
            idx, val = item
            while len(merged) <= idx:
                merged.append("")
            merged[idx] = val
        elif isinstance(item, dict) and "step_index" in item:
            idx = item["step_index"]
            val = item.get("result", "")
            while len(merged) <= idx:
                merged.append("")
            merged[idx] = val
    return merged


def _sum_complex_steps(existing: int, updates: int) -> int:
    return existing + updates


def _pending_parallel_reduce(existing: int, updates: int) -> int:
    return max(0, existing + updates)


class AgentState(TypedDict, total=False):
    user_input: str
    graph_context: str
    chat_history_messages: list
    classification: str
    plan: list[str]
    parallel_groups: list[list[int]]
    current_step: int
    results: Annotated[list, _merge_results]
    parallel_results: dict[str, list[str]]
    result: str
    is_done: bool
    plan_rounds: int
    graph_start_time: float
    fast_iterations: int
    fast_tool_calls: int
    fast_elapsed: float
    fast_hit_limit: bool
    complex_steps_executed: Annotated[int, _sum_complex_steps]
    complex_elapsed: float
    reflection_scores: dict[str, float]
    reflection_summary: str
    reflection_adjustments: list[str]
    reflection_rounds: int
    reflection_should_replan: bool
    reflection_confidence: float
    max_plan_rounds: int
    parallel_execution: bool
    pending_parallel: Annotated[int, _pending_parallel_reduce]


class _NodeContext:
    """Runtime context passed through LangGraph config.

    Wraps config["configurable"] to provide clean, type-hinted access
    to per-run values (cancel_event, progress_cb, llm, etc.) without
    relying on threading.local().
    """

    def __init__(self, config: Any) -> None:
        self._cfg = config.get("configurable", {}) if config else {}

    @property
    def cancel_event(self) -> threading.Event | None:
        return self._cfg.get("cancel_event")

    @property
    def progress_cb(self) -> Callable[[dict[str, Any]], None] | None:
        return self._cfg.get("progress_cb")

    @property
    def token_tracker(self) -> Callable[[int], None] | None:
        return self._cfg.get("token_tracker")

    @property
    def llm(self) -> Any:
        return self._cfg.get("llm")

    def is_cancelled(self) -> bool:
        ce = self.cancel_event
        return ce is not None and ce.is_set()

    def emit_log(self, message: str) -> None:
        cb = self.progress_cb
        if cb:
            cb(make_log(message))

    def emit_status(self, message: str) -> None:
        cb = self.progress_cb
        if cb:
            cb(make_status(message))

    def emit_plan(self, goal: str = "", steps: list[str] | None = None) -> None:
        cb = self.progress_cb
        if cb:
            cb(make_plan(goal=goal, steps=steps or []))


class GraphOrchestrator:
    """Pure LangGraph Plan-and-Execute orchestrator.

    Thread-safe: runtime context flows through LangGraph's config dict
    rather than threading.local(). The compiled graph is read-only after
    __init__, so it is safe to share across threads.

    Enhanced with LangGraph-native features:
      - SQLite/Memory checkpointer for state persistence
      - Command(goto=...) for re-planning loops
      - Send API for parallel step execution
      - Config-based runtime context (no TLS)
      - Seamless integration with AutonomousGraph and MultiAgentGraph
    """

    def __init__(self, llm_provider: Any = None) -> None:
        if llm_provider is not None:
            self._llm_provider = llm_provider
            self._llm = self._llm_provider.get_model()
        else:
            self._llm_provider = create_llm()
            self._llm = self._llm_provider.get_model()

        self._skill_store = SkillStore()
        self._skill_learner_graph = SkillLearnerGraph(store=self._skill_store)

        self._checkpointer = self._init_checkpointer()

        self._reflection_engine = ReflectionEngine(
            llm=self._llm,
            checkpointer=self._checkpointer,
        )

        self._tool_executor = ToolExecutorGraph(
            llm=self._llm,
            tools=list_tools(),
            scene="plan_execute",
            checkpointer=self._checkpointer,
            max_iterations=settings.MAX_ITERATIONS,
        )

        self._autonomous_graph = AutonomousGraph(
            llm=self._llm,
            action_callback=self._autonomous_action,
            reflection_engine=self._reflection_engine,
            cycle_interval=5.0,
        )

        self._multi_agent_graph = MultiAgentGraph(
            roles=create_default_roles(),
            llm=self._llm,
            coordinator_llm=self._llm,
        )

        self._use_multi_agent = False
        self._current_cancel_event: threading.Event | None = None

        try:
            self._graph = self._build_graph()
        except Exception as exc:
            logger.error("[GraphOrch] Graph build failed: %s", exc, exc_info=True)
            self._graph = None

        logger.info(
            "[GraphOrch] LangGraph ready | LLM: %s | features: [checkpointer, command, send, subgraph, config]",
            self._llm_provider.model_name,
        )

    def _init_checkpointer(self) -> Any:
        try:
            import sqlite3
            from langgraph.checkpoint.sqlite import SqliteSaver

            os.makedirs(_CHECKPOINT_DIR, exist_ok=True)
            db_path = os.path.join(_CHECKPOINT_DIR, "graph_checkpoints.db")
            logger.info("[GraphOrch] Using SQLite checkpointer: %s", db_path)
            conn = sqlite3.connect(db_path, check_same_thread=False)
            return SqliteSaver(conn)
        except BaseException as exc:
            logger.warning("[GraphOrch] SQLite checkpointer failed (%s), falling back to MemorySaver", exc)
            return MemorySaver()

    def request_stop(self) -> None:
        if self._current_cancel_event is not None:
            self._current_cancel_event.set()

    def _build_graph(self) -> Any:
        graph = StateGraph(AgentState)

        graph.add_node("classify", self._node_classify)
        graph.add_node("react_fast", self._node_react_fast)
        graph.add_node("adaptive_check", self._node_adaptive_check)
        graph.add_node("plan", self._node_plan)
        graph.add_node("dispatch", self._node_dispatch)
        graph.add_node("execute_step", self._node_execute_step)
        graph.add_node("check_steps", self._node_check_steps)
        graph.add_node("verify", self._node_verify)
        graph.add_node("reflect", self._node_reflect)
        graph.add_node("re_plan", self._node_re_plan)
        graph.add_node("multi_agent_execute", self._node_multi_agent_execute)

        graph.add_edge(START, "classify")

        graph.add_conditional_edges(
            "classify",
            self._route_after_classify,
            {
                "simple": "react_fast",
                "complex": "plan",
                "multi_agent": "multi_agent_execute",
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

        graph.add_edge("plan", "dispatch")

        graph.add_edge("execute_step", "check_steps")

        graph.add_edge("verify", "reflect")

        graph.add_conditional_edges(
            "reflect",
            self._route_after_reflect,
            {
                "complete": END,
                "replan": "re_plan",
            },
        )

        graph.add_edge("re_plan", "dispatch")

        graph.add_edge("multi_agent_execute", END)

        return graph.compile(checkpointer=self._checkpointer)

    def run(
        self,
        user_input: str,
        graph_context: str,
        chat_history_messages: list,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_event: threading.Event | None = None,
        token_tracker: Callable[[int], None] | None = None,
    ) -> str:
        cancel_evt = cancel_event or threading.Event()
        self._current_cancel_event = cancel_evt

        try:
            if self._graph is None:
                return "[错误] LangGraph 不可用，请检查日志。"

            initial_state: AgentState = {
                "user_input": user_input,
                "graph_context": graph_context,
                "chat_history_messages": chat_history_messages,
                "classification": "",
                "plan": [],
                "parallel_groups": [],
                "current_step": 0,
                "results": [],
                "parallel_results": {},
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
                "max_plan_rounds": settings.MAX_PLAN_ROUNDS,
                "parallel_execution": True,
                "pending_parallel": 0,
            }

            config = {
                "configurable": {
                    "thread_id": f"run_{hash(user_input) & 0xFFFFFFFF}",
                    "cancel_event": cancel_evt,
                    "progress_cb": progress_callback,
                    "token_tracker": token_tracker,
                    "llm": self._llm,
                }
            }

            ctx = _NodeContext(config)
            ctx.emit_status("开始分析...")

            try:
                final_state = self._graph.invoke(initial_state, config=config)
            except Exception as exc:
                logger.error("[GraphOrch] Graph error: %s", exc, exc_info=True)
                ctx.emit_log(f"❌ 执行出错: {exc}")
                return f"错误: {exc}"

            self._record_complex_downgrade_feedback(final_state)

            logger.info(
                "[GraphOrch] Final state: result_len=%d, is_done=%s, results_count=%d, classification=%s",
                len(final_state.get("result", "") or ""),
                final_state.get("is_done"),
                len(final_state.get("results", []) or []),
                final_state.get("classification"),
            )

            result = final_state.get("result", "")
            if not result:
                result = "已完成，但未产生文本输出。"

            return result

        finally:
            self._current_cancel_event = None

    def stream_run(
        self,
        user_input: str,
        graph_context: str,
        chat_history_messages: list,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_event: threading.Event | None = None,
        token_tracker: Callable[[int], None] | None = None,
        stream_mode: str = "updates",
    ) -> tuple[str, list[dict[str, Any]]]:
        cancel_evt = cancel_event or threading.Event()
        self._current_cancel_event = cancel_evt

        state_updates: list[dict[str, Any]] = []

        try:
            if self._graph is None:
                return "[错误] LangGraph 不可用，请检查日志。", state_updates

            initial_state: AgentState = {
                "user_input": user_input,
                "graph_context": graph_context,
                "chat_history_messages": chat_history_messages,
                "classification": "",
                "plan": [],
                "parallel_groups": [],
                "current_step": 0,
                "results": [],
                "parallel_results": {},
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
                "max_plan_rounds": settings.MAX_PLAN_ROUNDS,
                "parallel_execution": True,
                "pending_parallel": 0,
            }

            config = {
                "configurable": {
                    "thread_id": f"stream_{hash(user_input) & 0xFFFFFFFF}",
                    "cancel_event": cancel_evt,
                    "progress_cb": progress_callback,
                    "token_tracker": token_tracker,
                    "llm": self._llm,
                }
            }

            ctx = _NodeContext(config)
            ctx.emit_status("开始分析...")

            try:
                final_state = {}
                for chunk in self._graph.stream(
                    initial_state,
                    config=config,
                    stream_mode=stream_mode,
                ):
                    if cancel_evt.is_set():
                        ctx.emit_log("⏹ 流式执行已取消")
                        break

                    if stream_mode == "updates":
                        for node_name, node_state in chunk.items():
                            ctx.emit_log(f"📦 节点 [{node_name}] 完成")
                            state_updates.append({
                                "node": node_name,
                                "state": dict(node_state) if isinstance(node_state, dict) else {},
                            })
                            if progress_callback:
                                progress_callback({
                                    "type": "node_complete",
                                    "node": node_name,
                                    "state": dict(node_state) if isinstance(node_state, dict) else {},
                                })
                    elif stream_mode == "values":
                        state_updates.append({
                            "node": "graph_state",
                            "state": dict(chunk) if isinstance(chunk, dict) else {},
                        })

                final_state = state_updates[-1].get("state", {}) if state_updates else {}

            except Exception as exc:
                logger.error("[GraphOrch] Stream error: %s", exc, exc_info=True)
                ctx.emit_log(f"❌ 流式执行出错: {exc}")
                return f"错误: {exc}", state_updates

            self._record_complex_downgrade_feedback(final_state)

            result = final_state.get("result", "")
            if not result:
                result = "已完成，但未产生文本输出。"

            return result, state_updates

        finally:
            self._current_cancel_event = None

    @staticmethod
    def _build_tools_description() -> str:
        tools = list_tools()
        if not tools:
            return "(无可用工具)"
        lines = ["可用工具:"]
        for t in tools:
            name = getattr(t, "name", "未知")
            desc = (getattr(t, "description", "") or "")[:120]
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)

    def _record_complex_downgrade_feedback(self, final_state: AgentState) -> None:
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

    def _llm_invoke(self, prompt: str, ctx: _NodeContext) -> str:
        cache = get_cache()
        cache_key = cache.make_key(self._llm_provider.model_name, prompt)

        cached = cache.get(cache_key)
        if cached is not None:
            logger.debug("[GraphOrch] LLM cache hit for prompt hash")
            if ctx.token_tracker:
                ctx.token_tracker(50)
            return cached

        try:
            response = ctx.llm.invoke(prompt)
            result = str(response.content) if hasattr(response, "content") else str(response)
            if isinstance(result, list):
                result = "".join(str(c) for c in result)

            token_count = 0
            if hasattr(response, "response_metadata"):
                token_count = response.response_metadata.get("token_usage", {}).get("total_tokens", 0)
            if ctx.token_tracker and token_count > 0:
                ctx.token_tracker(token_count)

            if result and token_count > 0:
                cache.set(cache_key, result)
            return result
        except Exception as exc:
            logger.warning("[GraphOrch] LLM invoke failed: %s", exc)
            ctx.emit_log(f"⚠️ LLM 调用失败: {exc}")
            return ""

    def _truncate_output(self, text: str) -> str:
        limit = settings.TOOL_OUTPUT_TRUNCATE
        if len(text) <= limit:
            return text
        return text[:limit] + f"\n... [输出已截断，共 {len(text)} 字符]"

    @staticmethod
    def _parse_plan(plan_text: str) -> list[str]:
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
    def _parse_plan_with_parallel(plan_text: str) -> tuple[list[str], list[list[int]]]:
        steps: list[str] = []
        parallel_groups: list[list[int]] = []

        for line in plan_text.split("\n"):
            line = line.strip()
            if not line:
                continue

            is_parallel_group = "[" in line and "]" in line and "|" in line

            if is_parallel_group:
                idx = line.index("]")
                content = line[idx + 1:].strip()
                if content.startswith(":"):
                    content = content[1:].strip()

                sub_steps = [s.strip() for s in content.split("|") if s.strip()]
                group_indices = []
                for sub in sub_steps:
                    clean = sub.lstrip("•-· \t").strip()
                    if clean:
                        steps.append(clean)
                        group_indices.append(len(steps) - 1)

                if group_indices:
                    parallel_groups.append(group_indices)
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

        return steps, parallel_groups

    def _run_planning(
        self,
        ctx: _NodeContext,
        state: AgentState,
        *,
        prompt: str,
        fallback_plan: list[str],
        log_prefix: str,
        error_msg: str,
        emit_plan_event: bool = True,
    ) -> dict[str, Any]:
        plan_rounds = state.get("plan_rounds", 0)
        new_plan_rounds = plan_rounds + 1
        user_input = state.get("user_input", "")

        try:
            plan_text = self._llm_invoke(prompt, ctx)
            plan, parallel_groups = self._parse_plan_with_parallel(plan_text)
        except Exception as exc:
            logger.error("[GraphOrch] %s failed: %s", log_prefix, exc)
            ctx.emit_log(f"⚠️ {error_msg}，使用备用计划")
            plan = list(fallback_plan)
            parallel_groups = []

        if not plan:
            plan = list(fallback_plan)
            parallel_groups = []

        ctx.emit_log(f"📋 {log_prefix}: {len(plan)} 个步骤")
        if emit_plan_event:
            if parallel_groups:
                ctx.emit_log(f"⚡ 检测到 {len(parallel_groups)} 组可并行步骤")
            ctx.emit_plan(goal=user_input, steps=plan)
        for i, step in enumerate(plan):
            ctx.emit_log(f"  {i + 1}. {step[:120]}")

        return {
            "plan": plan,
            "parallel_groups": parallel_groups,
            "current_step": 0,
            "results": [],
            "parallel_results": {},
            "plan_rounds": new_plan_rounds,
            "pending_parallel": 0,
        }

    @staticmethod
    def _extract_final_answer(
        verification: str, user_input: str, results: list[str]
    ) -> str:
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
                if stripped.lower().startswith("yes"):
                    remainder = stripped[3:].strip(".,;:：,。； ")
                else:
                    remainder = stripped[1:].strip("，。； ")
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

    def _node_classify(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        ctx = _NodeContext(config)
        if ctx.is_cancelled():
            return {"result": "⏹ 已停止", "is_done": True}

        classification = classify_query(state["user_input"])

        if self._use_multi_agent and classification == "complex":
            user_input = state["user_input"]
            if any(kw in user_input.lower() for kw in (
                "团队", "协作", "分工", "角色", "多方面", "综合",
                "分析", "设计", "开发", "研究",
            )):
                classification = "multi_agent"
                ctx.emit_log(f"🔍 查询分类: {classification} (多 Agent 协作模式)")
                return {"classification": classification}

        ctx.emit_log(f"🔍 查询分类: {classification}")
        return {"classification": classification}

    def _node_react_fast(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        ctx = _NodeContext(config)
        if ctx.is_cancelled():
            return {"result": "⏹ 已停止", "is_done": True}

        ctx.emit_log("⚡ 执行快速路径...")

        exec_result = self._tool_executor.execute(
            input_text=state["user_input"],
            chat_history=state.get("chat_history_messages", []),
            cancel_event=ctx.cancel_event,
            progress_callback=ctx.progress_cb,
            token_tracker=ctx.token_tracker,
            config=config,
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

    def _node_adaptive_check(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        ctx = _NodeContext(config)
        if ctx.is_cancelled():
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
            ctx.emit_log(
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

    def _node_plan(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        ctx = _NodeContext(config)
        if ctx.is_cancelled():
            return {"result": "⏹ 已停止", "is_done": True}

        plan_rounds = state.get("plan_rounds", 0)

        if plan_rounds == 0:
            ctx.emit_status("📋 正在创建执行计划...")
        else:
            ctx.emit_status(f"🔄 重新规划 (第 {plan_rounds + 1} 轮)...")

        user_input = state.get("user_input", "")
        previous_plan = state.get("plan", [])
        previous_results = state.get("results", [])
        graph_context = state.get("graph_context", "")
        tools_desc = self._build_tools_description()

        context_parts = [f"用户目标: {user_input}"]

        if graph_context:
            context_parts.append(f"上下文:\n{graph_context}")

        if previous_plan:
            context_parts.append("之前的计划:")
            for i, step in enumerate(previous_plan):
                status = previous_results[i] if i < len(previous_results) else "(未执行)"
                context_parts.append(f"  步骤 {i + 1}: {step} → {status[:200]}")

        context = "\n".join(context_parts)

        parallel_hint = ""
        if state.get("parallel_execution", True):
            parallel_hint = (
                "\n\n注意：如果某些步骤可以并行执行（互不依赖），请用 [并行组] 标记。"
                "格式：将可并行的步骤放在同一行，用 | 分隔。例如：\n"
                "1. [并行组] 搜索文档 | 分析代码结构\n"
                "2. 整合信息\n"
                "如果不确定，就按顺序列出所有步骤。"
            )

        prompt = f"""{context}

{tools_desc}{parallel_hint}

请创建一个清晰的、分步的执行计划来实现用户目标。
将其分解为 2-5 个具体的、可执行的步骤。每一步应该是一个单独的工具调用或简单操作。

重要：您必须使用中文回复。所有步骤描述必须使用中文。

请按以下格式输出，每行一个步骤，带编号:
1. [步骤描述 — 具体说明要做什么]
2. [步骤描述]
...

只输出编号步骤，不要输出其他内容。"""

        fallback_plan = [
            f"分析请求: {user_input}",
            "执行所需操作",
            "总结结果",
        ]

        return self._run_planning(
            ctx, state,
            prompt=prompt,
            fallback_plan=fallback_plan,
            log_prefix="计划已创建",
            error_msg="计划生成失败",
            emit_plan_event=True,
        )

    def _node_execute_step(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        ctx = _NodeContext(config)
        if ctx.is_cancelled():
            return {"result": "⏹ 已停止", "is_done": True}

        current_step = state.get("current_step", 0)
        plan = state.get("plan", [])
        parallel_groups = state.get("parallel_groups", [])

        logger.info(
            "[GraphOrch] execute_step called: current_step=%d, plan_len=%d, results_in_state=%d",
            current_step, len(plan), len(state.get("results", []) or []),
        )

        if current_step >= len(plan):
            logger.warning("[GraphOrch] execute_step: current_step >= plan length, returning empty")
            return {}

        step_description = plan[current_step]

        is_parallel = any(
            current_step in group for group in parallel_groups
        )

        ctx.emit_log(
            f"▶️ 步骤 {current_step + 1}/{len(plan)}: {step_description[:120]}"
            + (" [并行]" if is_parallel else "")
        )

        user_input = state.get("user_input", "")
        graph_context = state.get("graph_context", "")
        chat_history = state.get("chat_history_messages", [])

        chat_debug = []
        for m in chat_history:
            cls_name = type(m).__name__
            mtype = getattr(m, "type", "?")
            content = getattr(m, "content", "")
            if isinstance(content, list):
                content = str(content)[:80]
            elif isinstance(content, str):
                content = content[:80]
            else:
                content = str(content)[:80]
            chat_debug.append(f"{cls_name}({mtype}, len={len(str(content))})")
        logger.info("[GraphOrch] execute_step: chat_history_msgs=[%s]", ", ".join(chat_debug))

        context_sections = []
        if graph_context:
            context_sections.append(f"之前的上下文:\n{graph_context[:1200]}")

        step_prompt = f"""执行此步骤以实现用户目标。

用户原始目标: {user_input}
当前步骤: {step_description}
{chr(10).join(context_sections)}

使用可用工具执行所需操作。如果需要读取文件、搜索网页或运行代码，请立即使用相应工具。

重要：您必须使用中文回复。用中文思考和回复。

完成此步骤后，简要总结您做了什么以及发现了什么。"""

        result = self._tool_executor.execute(
            input_text=step_prompt,
            chat_history=chat_history,
            cancel_event=ctx.cancel_event,
            progress_callback=ctx.progress_cb,
            token_tracker=ctx.token_tracker,
            config=config,
        )

        result_text = result.get("result", "") if isinstance(result, dict) else str(result)
        result_text = self._truncate_output(result_text)

        ctx.emit_log(f"✅ 步骤 {current_step + 1} 完成: {result_text[:200]}")
        logger.info(
            "[GraphOrch] execute_step: current_step=%d, result_len=%d, plan_len=%d",
            current_step, len(result_text), len(plan),
        )

        return {
            "results": [(current_step, result_text)],
            "complex_steps_executed": 1,
            "pending_parallel": -1,
        }

    def _node_check_steps(self, state: AgentState, config: RunnableConfig) -> Command:
        ctx = _NodeContext(config)
        plan = state.get("plan", [])
        current_step = state.get("current_step", 0)
        pending_parallel = state.get("pending_parallel", 0)

        logger.info(
            "[GraphOrch] check_steps: current_step=%d, pending_parallel=%d, plan_len=%d",
            current_step, pending_parallel, len(plan),
        )

        if pending_parallel > 0:
            ctx.emit_log(f"⏳ 等待 {pending_parallel} 个并行步骤完成...")
            return Command(goto=END)

        if current_step >= len(plan):
            ctx.emit_log("📋 所有计划步骤已完成")
            logger.info("[GraphOrch] check_steps: all steps done, going to verify")
            return Command(goto="verify")

        next_step = current_step + 1
        parallel_execution = state.get("parallel_execution", True)
        if parallel_execution:
            for group in state.get("parallel_groups", []):
                if current_step in group:
                    next_step = max(group) + 1
                    ctx.emit_log(f"📋 并行组完成，推进到步骤 {next_step}")
                    break

        logger.info("[GraphOrch] check_steps: advancing to step %d", next_step)
        return Command(goto="dispatch", update={"current_step": next_step})

    def _node_dispatch(self, state: AgentState, config: RunnableConfig) -> Command:
        ctx = _NodeContext(config)
        if ctx.is_cancelled():
            return Command(goto="verify", update={"result": "⏹ 已停止", "is_done": True})

        plan = state.get("plan", [])
        current_step = state.get("current_step", 0)
        parallel_groups = state.get("parallel_groups", [])
        parallel_execution = state.get("parallel_execution", True)

        logger.info(
            "[GraphOrch] dispatch: current_step=%d, plan_len=%d, parallel_groups=%s",
            current_step, len(plan), parallel_groups,
        )

        if current_step >= len(plan):
            ctx.emit_log("📋 所有计划步骤已完成")
            logger.info("[GraphOrch] dispatch: all steps done, going to verify")
            return Command(goto="verify")

        if not parallel_execution or not parallel_groups:
            logger.info("[GraphOrch] dispatch: serial step %d", current_step)
            return Command(
                goto="execute_step",
                update={"pending_parallel": 1},
            )

        parallel_indices = set()
        for group in parallel_groups:
            if current_step in group:
                parallel_indices = set(group)
                break

        if not parallel_indices or len(parallel_indices) <= 1:
            logger.info("[GraphOrch] dispatch: single step %d (no parallel group)", current_step)
            return Command(
                goto="execute_step",
                update={"pending_parallel": 1},
            )

        sorted_indices = sorted(parallel_indices)
        ctx.emit_log(f"⚡ 并行调度 {len(sorted_indices)} 个步骤: {sorted_indices}")
        logger.info("[GraphOrch] dispatch: true parallel steps %s", sorted_indices)

        sends = [
            Send("execute_step", self._build_exec_step_state(state, idx))
            for idx in sorted_indices
        ]
        return Command(
            goto=sends,
            update={"pending_parallel": len(sorted_indices)},
        )

    @staticmethod
    def _build_exec_step_state(state: AgentState, step_index: int) -> dict[str, Any]:
        return {
            "current_step": step_index,
            "plan": state.get("plan", []),
            "parallel_groups": state.get("parallel_groups", []),
            "chat_history_messages": state.get("chat_history_messages", []),
            "user_input": state.get("user_input", ""),
            "graph_context": state.get("graph_context", ""),
            "complex_steps_executed": state.get("complex_steps_executed", 0),
            "results": state.get("results", []),
        }

    def _node_verify(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        ctx = _NodeContext(config)
        user_input = state.get("user_input", "")
        results = state.get("results", [])
        plan = state.get("plan", [])
        plan_rounds = state.get("plan_rounds", 0)
        graph_context = state.get("graph_context", "")

        logger.info(
            "[GraphOrch] verify: results_count=%d, plan_count=%d, is_done=%s",
            len(results), len(plan), state.get("is_done"),
        )

        if ctx.is_cancelled():
            return {"result": "⏹ 已停止", "is_done": True}

        if not results:
            logger.warning("[GraphOrch] verify: results is EMPTY, skipping verification")
            ctx.emit_log("⚠️ 没有执行结果，结束执行")
            return {"is_done": True}

        max_plan_rounds = state.get("max_plan_rounds", settings.MAX_PLAN_ROUNDS)
        if plan_rounds >= max_plan_rounds:
            logger.info("[GraphOrch] Max re-plan rounds (%d) reached, ending", plan_rounds)
            ctx.emit_log("⚠️ 达到最大重新规划次数，接受当前结果")
            if results:
                final_answer = "\n\n".join(
                    f"**结果 {i + 1}**: {r[:500]}" for i, r in enumerate(results)
                )
            else:
                final_answer = f"任务已完成: {user_input}"
            return {"result": final_answer, "is_done": True}

        ctx.emit_log("🔍 正在验证目标达成情况...")

        summary_parts = ["## 执行结果\n"]
        for i, (step, result) in enumerate(zip(plan, results)):
            summary_parts.append(f"**步骤 {i + 1}**: {step}")
            summary_parts.append(f"结果: {result[:300]}")
            summary_parts.append("")

        full_summary = "\n".join(summary_parts)

        context_block = ""
        if graph_context:
            context_block = f"\n\n上下文:\n{graph_context[:800]}"

        prompt = f"""{full_summary}{context_block}

用户原始目标: {user_input}

基于以上执行结果，判断用户目标是否已完全实现？

重要：您必须使用中文回复。

请明确回答"是"或"否"，然后用中文提供简要解释。
如果回答"是"，请同时用中文为用户提供简洁的最终答案。
如果回答"否"，请用中文解释还缺少什么。"""

        try:
            verification = self._llm_invoke(prompt, ctx)

            ctx.emit_log(f"🔍 验证结果: {verification[:200]}")

            is_complete = verification.strip().lower().startswith(("yes", "是"))

            if is_complete:
                final_answer = self._extract_final_answer(verification, user_input, results)
                ctx.emit_log("🎯 目标已达成!")
                return {
                    "result": final_answer,
                    "is_done": True,
                }

            if plan_rounds >= max_plan_rounds:
                ctx.emit_log("⚠️ 达到最大重新规划次数，接受当前结果")
                final_answer = self._extract_final_answer(verification, user_input, results)
                return {
                    "result": final_answer,
                    "is_done": True,
                }

            ctx.emit_log("🔍 目标未完全达成，需要重新规划...")

            return {
                "result": verification,
                "is_done": False,
            }

        except Exception as exc:
            logger.warning("[GraphOrch] Verification failed: %s", exc)
            ctx.emit_log("⚠️ 验证失败，基于已有结果继续")
            final_answer = self._extract_final_answer(
                f"验证失败: {exc}", user_input, results
            )
            return {
                "result": final_answer,
                "is_done": True,
            }

    def _node_re_plan(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        ctx = _NodeContext(config)
        if ctx.is_cancelled():
            return {"result": "⏹ 已停止", "is_done": True}

        plan_rounds = state.get("plan_rounds", 0)
        new_plan_rounds = plan_rounds + 1

        ctx.emit_status(f"📋 重新规划 (第 {new_plan_rounds} 轮)...")

        user_input = state.get("user_input", "")
        results = state.get("results", [])
        plan = state.get("plan", [])
        verification_feedback = state.get("result", "")
        graph_context = state.get("graph_context", "")
        reflection_adjustments = state.get("reflection_adjustments", [])
        reflection_summary = state.get("reflection_summary", "")

        plan_lines = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(plan))
        result_lines = "\n".join(f"步骤 {i + 1}: {r[:200]}" for i, r in enumerate(results))

        context_block = f"\n对话上下文:\n{graph_context[:500]}" if graph_context else ""

        reflection_block = ""
        if reflection_summary:
            reflection_block = f"\n\n## 深度反思结果\n{reflection_summary[:500]}"
        if reflection_adjustments:
            reflection_block += "\n\n## 策略调整建议:\n" + "\n".join(
                f"- {a}" for a in reflection_adjustments[:5]
            )

        prompt = f"""原始目标: {user_input}

之前的计划:
{plan_lines}

之前的结果:
{result_lines}

验证反馈: {verification_feedback[:500]}
{reflection_block}
{context_block}

请根据反思结果和策略调整建议，创建一个改进后的执行计划。
只列出仍需完成的步骤，优先解决反思中发现的问题。

重要：您必须使用中文回复。所有步骤描述必须使用中文。

请只输出编号步骤。"""

        fallback_plan = [f"完成剩余工作: {user_input}"]

        return self._run_planning(
            ctx, state,
            prompt=prompt,
            fallback_plan=fallback_plan,
            log_prefix="新计划",
            error_msg="重新规划失败",
            emit_plan_event=False,
        )

    def _node_multi_agent_execute(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        ctx = _NodeContext(config)
        if ctx.is_cancelled():
            return {"result": "⏹ 已停止", "is_done": True}

        user_input = state.get("user_input", "")
        ctx.emit_status("👥 启动多 Agent 协作...")
        ctx.emit_log(f"📋 分解任务: {user_input[:80]}")

        try:
            result = self._multi_agent_graph.invoke(user_input)

            synthesis = result.get("synthesis", "")
            results_list = result.get("role_results", [])
            success_count = sum(1 for r in results_list if r.get("status") == "success")
            total_count = len(results_list)

            for r in results_list:
                role = r.get("role", "unknown")
                status = r.get("status", "unknown")
                ctx.emit_log(f"  → [{role}]: {status}")

            ctx.emit_log(
                f"✅ 协作完成: {success_count}/{total_count} 角色成功"
            )

            return {
                "result": synthesis or "(多 Agent 协作未产生结果)",
                "is_done": True,
                "complex_steps_executed": total_count,
                "complex_elapsed": result.get("duration", 0.0),
            }

        except Exception as exc:
            logger.error("[GraphOrch] Multi-agent execution failed: %s", exc, exc_info=True)
            ctx.emit_log(f"❌ 多 Agent 执行失败: {exc}")
            return {
                "result": f"多 Agent 执行失败: {exc}",
                "is_done": True,
            }

    def _node_reflect(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        ctx = _NodeContext(config)
        if ctx.is_cancelled():
            return {"result": "⏹ 已停止", "is_done": True}

        if state.get("is_done"):
            return {}

        user_input = state.get("user_input", "")
        plan = state.get("plan", [])
        results = state.get("results", [])
        graph_context = state.get("graph_context", "")
        verification_feedback = state.get("result", "")
        reflection_rounds = state.get("reflection_rounds", 0)

        ctx.emit_log("🤔 正在进行深度反思...")

        ctx.emit_log(
            f"📊 执行统计: {len(results)} 步骤, "
            f"用时 {state.get('complex_elapsed', 0):.1f}s"
        )

        relevant_strategy = self._reflection_engine.get_relevant_strategy(user_input)
        if relevant_strategy:
            ctx.emit_log("💡 发现相关历史策略，参考之前的成功经验")

        execution_metrics = {
            "steps": state.get("complex_steps_executed", len(plan)),
            "tool_calls": state.get("fast_tool_calls", 0),
            "time": state.get("complex_elapsed", 0),
        }

        try:
            reflection_result = self._reflection_engine.invoke(
                user_input=user_input,
                plan=plan,
                results=results,
                execution_metrics=execution_metrics,
                previous_feedback=verification_feedback,
                config=config,
            )
        except Exception as exc:
            logger.warning("[GraphOrch] Reflection failed: %s", exc)
            reflection_result = ReflectionResult(
                scores=ReflectionScore(overall=0.5),
                should_replan=True,
                summary=f"反思失败: {exc}",
            )

        scores_dict = {
            "goal_completeness": reflection_result.scores.goal_completeness,
            "output_quality": reflection_result.scores.output_quality,
            "process_efficiency": reflection_result.scores.process_efficiency,
            "tool_selection": reflection_result.scores.tool_selection,
            "overall": reflection_result.scores.overall,
        }

        ctx.emit_log(
            f"📈 反思评分: 综合={reflection_result.scores.overall:.2f} | "
            f"完整性={reflection_result.scores.goal_completeness:.2f} | "
            f"质量={reflection_result.scores.output_quality:.2f}"
        )

        if reflection_result.went_well:
            for item in reflection_result.went_well[:3]:
                ctx.emit_log(f"  ✅ {item}")

        if reflection_result.went_wrong:
            for item in reflection_result.went_wrong[:3]:
                ctx.emit_log(f"  ❌ {item}")

        if reflection_result.strategy_adjustments:
            for item in reflection_result.strategy_adjustments[:3]:
                ctx.emit_log(f"  🔧 {item}")

        ctx.emit_log(
            f"{'🎯 目标达成，结束执行' if not reflection_result.should_replan else '🔄 需要调整策略，重新规划'}"
        )

        if not reflection_result.should_replan and reflection_result.scores.overall >= 0.5:
            try:
                learned = self._skill_learner_graph.learn_from_execution(
                    user_input=user_input,
                    plan=plan,
                    results=results,
                    success=True,
                    reflection_score=reflection_result.scores.overall,
                )
                if learned:
                    ctx.emit_log(f"📚 新技能已学习: {learned.name}")
            except Exception as exc:
                logger.debug("[GraphOrch] Skill learning failed: %s", exc)

        new_reflection_rounds = reflection_rounds + 1

        return {
            "reflection_scores": scores_dict,
            "reflection_summary": reflection_result.summary,
            "reflection_adjustments": reflection_result.strategy_adjustments,
            "reflection_rounds": new_reflection_rounds,
            "reflection_should_replan": reflection_result.should_replan,
            "reflection_confidence": reflection_result.confidence,
        }

    @staticmethod
    def _route_after_classify(state: AgentState) -> str:
        return state.get("classification", "complex")

    @staticmethod
    def _route_after_adaptive(state: AgentState) -> str:
        if state.get("classification") == "complex":
            return "upgrade"
        return "end"

    @staticmethod
    def _route_after_reflect(state: AgentState) -> str:
        if state.get("is_done"):
            return "complete"

        should_replan = state.get("reflection_should_replan", False)
        plan_rounds = state.get("plan_rounds", 0)
        max_plan_rounds = state.get("max_plan_rounds", settings.MAX_PLAN_ROUNDS)

        if not should_replan:
            return "complete"

        if plan_rounds >= max_plan_rounds:
            return "complete"

        return "replan"

    def _autonomous_action(self, goal: Goal) -> str:
        try:
            result = self.run(
                user_input=goal.description,
                graph_context=str(goal.context),
                chat_history_messages=[],
            )
            if result and "错误" not in result:
                return result
        except Exception as exc:
            logger.warning("[GraphOrch] Autonomous action failed: %s", exc)
        return ""

    def start_autonomous_mode(self) -> None:
        self._autonomous_graph.start()
        self._emit_status_safe("🌀 自主运行模式已启动 (LangGraph 驱动)")

    def stop_autonomous_mode(self) -> None:
        self._autonomous_graph.stop()
        self._emit_status_safe("⏹ 自主运行模式已停止")

    def add_autonomous_goal(
        self,
        description: str,
        priority: int = 0,
        deadline: float | None = None,
        context: dict[str, Any] | None = None,
    ) -> Goal:
        goal = self._autonomous_graph.add_goal(
            description=description,
            priority=priority,
            deadline=deadline,
            context=context,
        )
        self._emit_log_safe(f"🎯 目标已添加: {description}")
        return goal

    @property
    def autonomous_stats(self) -> dict[str, Any]:
        return self._autonomous_graph.stats

    def enable_multi_agent_mode(self, enabled: bool = True) -> None:
        self._use_multi_agent = enabled
        self._emit_log_safe(
            f"{'👥 多 Agent 协作模式已启用 (LangGraph Send 并行)' if enabled else '👤 单 Agent 模式'}"
        )

    def run_multi_agent(
        self,
        user_input: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._emit_log_safe("👥 启动多 Agent 协作 (LangGraph 并行执行)...")
        self._emit_log_safe(f"📋 分解任务: {user_input[:80]}")

        result = self._multi_agent_graph.invoke(user_input, context=context)

        synthesis = result.get("synthesis", "")
        results_list = result.get("role_results", [])
        success_count = sum(1 for r in results_list if r.get("status") == "success")
        total_count = len(results_list)

        for r in results_list:
            role = r.get("role", "unknown")
            status = r.get("status", "unknown")
            self._emit_log_safe(f"  → [{role}]: {status}")

        self._emit_log_safe(
            f"✅ 协作完成: {success_count}/{total_count} 角色成功 | 耗时 {result.get('duration', 0):.1f}s"
        )

        return result

    @property
    def team_stats(self) -> dict[str, Any]:
        return {
            "roles": self._multi_agent_graph.available_roles,
            "engine_type": "LangGraph MultiAgentGraph (Send API)",
        }

    @property
    def checkpointer_info(self) -> dict[str, Any]:
        return {
            "type": type(self._checkpointer).__name__,
            "configurable_keys": ["thread_id"],
            "db_path": os.path.join(_CHECKPOINT_DIR, "graph_checkpoints.db"),
        }

    @property
    def skill_learner_info(self) -> dict[str, Any]:
        return {
            "type": "LangGraph SkillLearnerGraph",
            "nodes": ["extract_sequence", "derive_patterns", "validate", "store", "skip", "finalize"],
        }

    def resume_from_checkpoint(
        self,
        thread_id: str,
        user_input: str,
        graph_context: str = "",
        chat_history_messages: list | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> str:
        cancel_evt = cancel_event or threading.Event()
        self._current_cancel_event = cancel_evt

        try:
            if self._graph is None:
                return "[错误] LangGraph 不可用，请检查日志。"

            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "cancel_event": cancel_evt,
                    "progress_cb": progress_callback,
                    "token_tracker": None,
                    "llm": self._llm,
                }
            }

            ctx = _NodeContext(config)
            ctx.emit_status("从检查点恢复执行...")

            try:
                checkpoint = self._checkpointer.get_tuple(config)
                if checkpoint is None:
                    ctx.emit_log(f"⚠️ 未找到检查点: {thread_id}")
                    return f"未找到检查点: {thread_id}"

                ctx.emit_log(
                    f"📂 恢复检查点: {thread_id} | 步骤: {checkpoint.checkpoint_id[:8]}..."
                )

                state_update: AgentState = {
                    "user_input": user_input,
                    "graph_context": graph_context,
                    "chat_history_messages": chat_history_messages or [],
                    "results": [],
                    "parallel_results": {},
                    "is_done": False,
                }

                final_state = self._graph.invoke(state_update, config=config)

                result = final_state.get("result", "")
                if not result:
                    result = "已完成，但未产生文本输出。"

                return result

            except Exception as exc:
                logger.error("[GraphOrch] Resume error: %s", exc, exc_info=True)
                ctx.emit_log(f"❌ 恢复执行出错: {exc}")
                return f"恢复错误: {exc}"

        finally:
            self._current_cancel_event = None

    def list_checkpoints(self, limit: int = 20) -> list[dict[str, Any]]:
        checkpoints = []
        try:
            if hasattr(self._checkpointer, 'list'):
                for checkpoint in self._checkpointer.list({}):
                    if len(checkpoints) >= limit:
                        break
                    checkpoints.append({
                        "thread_id": checkpoint.get("thread_id", ""),
                        "checkpoint_id": checkpoint.get("checkpoint_id", ""),
                        "created_at": checkpoint.get("created_at", ""),
                    })
        except Exception as exc:
            logger.debug("[GraphOrch] List checkpoints failed: %s", exc)
        return checkpoints

    def graph_structure_info(self) -> dict[str, Any]:
        return {
            "type": "StateGraph (Plan-and-Execute)",
            "nodes": [
                "classify", "react_fast", "adaptive_check",
                "plan", "dispatch", "execute_step",
                "check_steps", "verify", "reflect", "re_plan",
                "multi_agent_execute",
            ],
            "subgraphs": [
                {
                    "name": "ToolExecutorGraph",
                    "purpose": "Tool execution with reason/act/observe loop",
                    "checkpointer": "shared (inherited from parent)",
                    "integration": "called as subgraph via config",
                },
                {
                    "name": "ReflectionEngine",
                    "purpose": "Multi-layered self-evaluation",
                    "checkpointer": "shared (inherited from parent)",
                    "integration": "called as subgraph via config",
                },
                {
                    "name": "MultiAgentGraph",
                    "purpose": "Role-based parallel agent collaboration (researcher, analyst, executor)",
                    "checkpointer": "shared (inherited from parent)",
                    "integration": "routed from classify when multi-agent mode enabled",
                },
            ],
            "edges": [
                "classify → react_fast|plan|multi_agent",
                "react_fast → adaptive_check",
                "adaptive_check → END|plan",
                "plan → dispatch",
                "dispatch → execute_step (serial) | Send(execute_step) (parallel) | verify (all done)",
                "execute_step → check_steps",
                "check_steps → END (wait for parallel) | dispatch (next step) | verify (all done)",
                "verify → reflect",
                "reflect → END|re_plan",
                "re_plan → dispatch",
                "multi_agent_execute → END",
            ],
            "features": [
                "SqliteSaver shared checkpointer (persistent across subgraphs)",
                "Send API for true parallel fan-out/fan-in execution",
                "Subgraph composition (ToolExecutorGraph, ReflectionEngine, MultiAgentGraph)",
                "Shared checkpoints across graph hierarchy",
                "Config-based runtime context (no threading.local)",
                "Streaming via stream() API",
                "Checkpoint resume capability",
                "ToolNode-based tool execution engine",
                "Multi-agent role-based collaboration (researcher/analyst/executor)",
                "Autonomous goal-driven execution with skill learning",
            ],
        }

    def _emit_log_safe(self, message: str) -> None:
        """Emit log without requiring an active run context.

        Used by autonomous/multi-agent modes which call run() internally
        but need progress emitted at the orchestrator level.
        """
        cb = getattr(self, "_external_progress_cb", None)
        if cb:
            try:
                cb(make_log(message))
            except Exception:
                pass

    def _emit_status_safe(self, message: str) -> None:
        cb = getattr(self, "_external_progress_cb", None)
        if cb:
            try:
                cb(make_status(message))
            except Exception:
                pass

    def _store_external_progress_cb(self, callback: Callable[[dict[str, Any]], None] | None) -> None:
        self._external_progress_cb = callback