"""Multi-agent collaboration graph — LangGraph-native parallel execution.

Replaces the old serial multi_agent.py with LangGraph's Send API
for true parallel fan-out execution of role-based agents.

Architecture (LangGraph StateGraph + Send API):

                    ┌──────────────┐
                    │ coordinator  │
                    └──────┬───────┘
                           │ decompose
                    ┌──────▼───────┐
                    │  fan-out      │
                    │  (Send API)   │
                    └──┬───┬───┬───┘
                       │   │   │
          ┌────────────┘   │   └────────────┐
          ▼                ▼                 ▼
    ┌──────────┐    ┌──────────┐      ┌──────────┐
    │researcher │    │ analyst  │      │ executor  │
    │ (parallel)│    │(parallel)│      │(parallel) │
    └────┬─────┘    └────┬─────┘      └────┬─────┘
         │               │                 │
         └───────────────┼─────────────────┘
                         │ fan-in
                   ┌─────▼──────┐
                   │ synthesizer │
                   └─────┬──────┘
                         ▼
                      ┌──────┐
                      │ END  │
                      └──────┘

Key improvements over the old implementation:
  1. Send API enables TRUE parallel execution of role agents
  2. Annotated reducers manage shared state (results, context)
  3. Dynamic role dispatch — new roles can be added without graph changes
  4. StateGraph provides built-in checkpointing and streaming
  5. Proper error propagation and aggregation
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langgraph.graph import StateGraph, END, START
from langgraph.types import Command, Send

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Role definitions
# ---------------------------------------------------------------------------


@dataclass
class Role:
    """A specialized agent role in the multi-agent system."""

    name: str
    description: str
    system_prompt: str = ""
    tool_names: list[str] = field(default_factory=list)
    llm: Any = None
    action_fn: Callable[[str, dict[str, Any] | None], dict[str, Any]] | None = None

    def execute(self, task: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute a task within this role's specialization."""
        if self.action_fn is not None:
            try:
                return self.action_fn(task, context) or {
                    "role": self.name,
                    "task": task,
                    "result": "completed",
                    "status": "success",
                }
            except Exception as exc:
                return {
                    "role": self.name,
                    "task": task,
                    "result": f"Error: {exc}",
                    "status": "failed",
                }

        return {
            "role": self.name,
            "task": task,
            "result": f"[{self.name}] {task}",
            "status": "success",
        }


# ---------------------------------------------------------------------------
# Default roles
# ---------------------------------------------------------------------------


_RESEARCHER_PROMPT = (
    "你是一个研究专家，负责查找和收集信息。"
    "请使用搜索和文件工具收集相关数据，汇总成清晰的摘要。"
)

_ANALYST_PROMPT = (
    "你是一个分析专家，负责深入分析数据。"
    "请基于收集到的信息进行趋势分析、对比和预测。"
)

_EXECUTOR_PROMPT = (
    "你是一个执行专家，负责执行具体操作。"
    "请编写代码、生成报告或完成用户要求的具体任务。"
)


def create_default_roles() -> dict[str, Role]:
    """Create the set of default specialized roles."""
    return {
        "researcher": Role(
            name="researcher",
            description="研究专家：查找信息、收集数据",
            system_prompt=_RESEARCHER_PROMPT,
            tool_names=["web_search", "web_fetch", "read_file", "list_files"],
        ),
        "analyst": Role(
            name="analyst",
            description="分析专家：深度分析、趋势预测",
            system_prompt=_ANALYST_PROMPT,
            tool_names=["read_file", "search_code", "list_files"],
        ),
        "executor": Role(
            name="executor",
            description="执行专家：编写代码、生成内容",
            system_prompt=_EXECUTOR_PROMPT,
            tool_names=["read_file", "write_file", "run_command", "search_code"],
        ),
    }


# ---------------------------------------------------------------------------
# Multi-agent state (LangGraph TypedDict)
# ---------------------------------------------------------------------------


def _merge_results(existing: list, updates: list) -> list:
    """Reducer: merge parallel role results."""
    result = list(existing)
    seen_roles: set[str] = set()
    for item in result:
        if isinstance(item, dict) and "role" in item:
            seen_roles.add(item["role"])
    for update in updates:
        if isinstance(update, dict):
            role = update.get("role", "")
            if role and role not in seen_roles:
                result.append(update)
                seen_roles.add(role)
            elif role and role in seen_roles:
                for i, item in enumerate(result):
                    if isinstance(item, dict) and item.get("role") == role:
                        result[i] = update
                        break
    return result


def _merge_context(existing: dict, updates: dict) -> dict:
    """Reducer: merge context updates."""
    result = dict(existing)
    result.update(updates)
    return result


def _merge_role_errors(existing: list, updates: list) -> list:
    """Reducer: collect role errors."""
    return list(existing) + list(updates)


class MultiAgentState(TypedDict, total=False):
    task: str
    role_plan: list[dict[str, Any]]
    role_results: Annotated[list, _merge_results]
    role_errors: Annotated[list, _merge_role_errors]
    shared_context: Annotated[dict, _merge_context]
    synthesis: str
    started_at: float
    completed_at: float | None
    current_role_name: str
    current_sub_task: str
    barrier_triggered: bool


def _make_initial_state(
    task: str,
    role_plan: list[dict[str, Any]] | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create initial state for the multi-agent graph."""
    return {
        "task": task,
        "role_plan": role_plan or [],
        "role_results": [],
        "role_errors": [],
        "shared_context": context or {},
        "synthesis": "",
        "started_at": time.time(),
        "completed_at": None,
    }


# ---------------------------------------------------------------------------
# Multi-agent graph with parallel execution
# ---------------------------------------------------------------------------


class MultiAgentGraph:
    """LangGraph-based multi-agent collaboration engine.

    Uses LangGraph's Send API for true parallel execution of
    role-based agents, with proper state management via Annotated
    reducers.

    Usage:
        roles = create_default_roles()
        engine = MultiAgentGraph(roles=roles)

        # Or invoke the graph directly:
        result = engine.invoke("分析项目代码结构", context={"path": "/project"})

    Architecture:
        coordinator -> fan-out (Send) -> role1, role2, ... -> fan-in -> synthesizer
    """

    def __init__(
        self,
        roles: dict[str, Role] | None = None,
        llm: Any = None,
        coordinator_llm: Any = None,
    ) -> None:
        self._roles = roles or create_default_roles()
        self._llm = llm
        self._coordinator_llm = coordinator_llm or llm
        self._compiled_graph = self._build_graph()
        logger.info(
            "[MultiAgent] Graph ready with %d roles: %s",
            len(self._roles),
            list(self._roles.keys()),
        )

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        graph = StateGraph(MultiAgentState)

        graph.add_node("coordinator", self._node_coordinator)
        graph.add_node("role_executor", self._node_role_executor)
        graph.add_node("barrier", self._node_barrier)
        graph.add_node("synthesizer", self._node_synthesizer)

        graph.add_edge(START, "coordinator")
        graph.add_edge("role_executor", "barrier")
        graph.add_conditional_edges(
            "barrier",
            self._barrier_router,
            {
                "synthesizer": "synthesizer",
                "end": END,
            },
        )
        graph.add_edge("synthesizer", END)

        return graph.compile()

    # ------------------------------------------------------------------
    # Node implementations
    # ------------------------------------------------------------------

    def _node_coordinator(self, state: dict[str, Any]) -> Command:
        task = state.get("task", "")
        existing_plan = state.get("role_plan", [])

        if existing_plan:
            plan_for_sends = existing_plan
        elif self._coordinator_llm is None:
            plan_for_sends = self._simple_decompose(task).get("role_plan", [])
        else:
            plan_for_sends = self._llm_decompose(task, state)

        if plan_for_sends:
            sends = []
            shared_context = state.get("shared_context", {})
            for item in plan_for_sends:
                role_name = item.get("role", "")
                sub_task = item.get("task", "")
                sends.append(
                    Send(
                        "role_executor",
                        {
                            "current_role_name": role_name,
                            "current_sub_task": sub_task,
                            "shared_context": shared_context,
                            "task": task,
                        },
                    )
                )
            logger.info("[MultiAgent] Coordinator fanning out %d roles", len(sends))
            return Command(goto=sends, update={"role_plan": plan_for_sends})

        logger.info("[MultiAgent] Coordinator: no plan, jumping to synthesizer")
        return Command(goto="synthesizer", update={"role_plan": []})

    def _llm_decompose(self, task: str, state: dict[str, Any]) -> list[dict[str, str]]:
        """LLM-based task decomposition into role plan."""
        prompt = (
            f"作为多智能体协调员，请将以下任务分解为专业角色的子任务。\n\n"
            f"任务: {task}\n\n"
            f"可用角色:\n"
        )
        for role in self._roles.values():
            prompt += f"  - {role.name}: {role.description}\n"
        prompt += (
            "\n请以JSON数组形式返回，每个元素包含 role（角色名）和 task（该角色的子任务）字段。"
            "\n如果任务简单，返回空数组。只返回JSON，不要其他内容。"
        )

        try:
            response = self._coordinator_llm.invoke(prompt)
            plan = self._parse_llm_plan(
                response.content if hasattr(response, "content") else str(response)
            )
            if not plan:
                plan = self._simple_decompose(task).get("role_plan", [])
            return plan
        except Exception as exc:
            logger.warning("[MultiAgent] LLM decomposition failed: %s", exc)
            return self._simple_decompose(task).get("role_plan", [])

    def _simple_decompose(self, task: str) -> dict[str, Any]:
        """Simple heuristic decomposition when LLM is not available."""
        task_lower = task.lower()
        plan: list[dict[str, str]] = []

        if any(kw in task_lower for kw in ["搜索", "查找", "查", "find", "search", "研究", "research"]):
            plan.append({"role": "researcher", "task": f"搜索相关信息: {task}"})

        if any(kw in task_lower for kw in ["分析", "对比", "analyze", "compare", "统计"]):
            plan.append({"role": "analyst", "task": f"分析{task}"})

        if any(kw in task_lower for kw in ["写", "编写", "创建", "生成", "code", "create", "write", "build"]):
            plan.append({"role": "executor", "task": f"执行{task}"})

        if not plan:
            plan.append({"role": "executor", "task": task})

        return {"role_plan": plan}

    @staticmethod
    def _parse_llm_plan(response: str) -> list[dict[str, str]]:
        """Parse LLM response into role plan."""
        import json
        try:
            start = response.find("[")
            end = response.rfind("]") + 1
            if start == -1 or end == 0:
                return []
            data = json.loads(response[start:end])
            if isinstance(data, list):
                return [
                    {"role": item.get("role", "executor"), "task": item.get("task", "")}
                    for item in data
                    if isinstance(item, dict) and item.get("task")
                ]
        except Exception:
            pass
        return []

    # ------------------------------------------------------------------
    # Fan-out / fan-in with Send API
    # ------------------------------------------------------------------

    def _node_role_executor(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute a single role's sub-task (fan-out target).

        Called in parallel by the coordinator's Send API fan-out.
        Reads role identity from state, looks up Role instance,
        returns state update with role_results for the Annotated reducer.
        """
        role_name = state.get("current_role_name", "")
        sub_task = state.get("current_sub_task", "")
        context = state.get("shared_context", {})
        role = self._roles.get(role_name)

        return self._execute_single_role(role_name, sub_task, context, role)

    @staticmethod
    def _execute_single_role(
        role_name: str,
        sub_task: str,
        context: dict[str, Any],
        role: Role | None,
    ) -> dict[str, Any]:
        """Execute a single role's sub-task and return state update."""
        try:
            if role is not None:
                result = role.execute(sub_task, context)
            else:
                result = {
                    "role": role_name,
                    "task": sub_task,
                    "result": "completed",
                    "status": "success",
                }

            if not isinstance(result, dict):
                result = {
                    "role": role_name,
                    "task": sub_task,
                    "result": str(result),
                    "status": "success",
                }

            if "role" not in result:
                result["role"] = role_name
            if "task" not in result:
                result["task"] = sub_task

            return {"role_results": [result]}

        except Exception as exc:
            logger.error("[MultiAgent] Role %s failed: %s", role_name, exc)
            return {
                "role_results": [
                    {
                        "role": role_name,
                        "task": sub_task,
                        "result": f"Error: {exc}",
                        "status": "failed",
                        "error": str(exc),
                    }
                ]
            }

    def _node_barrier(self, state: dict[str, Any]) -> dict[str, Any]:
        results = state.get("role_results", [])
        plan = state.get("role_plan", [])
        logger.info(
            "[MultiAgent] barrier: results_count=%d, plan_count=%d, barrier_triggered=%s",
            len(results), len(plan), state.get("barrier_triggered"),
        )
        return {}

    def _barrier_router(self, state: dict[str, Any]) -> str:
        if state.get("barrier_triggered"):
            logger.info("[MultiAgent] barrier: already triggered, going to end")
            return "end"

        results = state.get("role_results", [])
        plan = state.get("role_plan", [])
        expected = len(plan)

        if expected > 0 and len(results) >= expected:
            logger.info("[MultiAgent] barrier: all results collected (%d/%d), routing to synthesizer", len(results), expected)
            return "synthesizer"

        logger.info("[MultiAgent] barrier: not all results yet (%d/%d), waiting", len(results), expected)
        return "end"

    def _node_synthesizer(self, state: dict[str, Any]) -> Command:
        if state.get("synthesis"):
            logger.info("[MultiAgent] synthesizer: already synthesized, skipping")
            return Command(goto=END)

        results = state.get("role_results", [])
        task = state.get("task", "")

        logger.info("[MultiAgent] synthesizer: processing %d results", len(results))

        if not results:
            synthesis = "没有产生任何结果。"
        elif len(results) == 1:
            synthesis = results[0].get("result", str(results[0]))
        else:
            synthesis = self._synthesize_results(task, results)

        return Command(goto=END, update={
            "synthesis": synthesis,
            "completed_at": time.time(),
            "role_results": results,
            "barrier_triggered": True,
        })

    def _synthesize_results(self, task: str, results: list[dict[str, Any]]) -> str:
        """Synthesize multiple role results into a coherent answer."""
        if self._llm is None:
            return self._simple_synthesize(task, results)

        results_text = "\n\n".join([
            f"[{r.get('role', 'unknown')}]: {r.get('result', '')}"
            for r in results
        ])

        prompt = (
            f"你是一个多智能体合成器。请将以下多个专业角色的执行结果整合成一份完整、连贯的报告。\n\n"
            f"原始任务: {task}\n\n"
            f"各角色执行结果:\n{results_text}\n\n"
            f"请综合以上所有结果，确保信息完整、逻辑清晰。用中文回复。"
        )

        try:
            response = self._llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            return content
        except Exception:
            return self._simple_synthesize(task, results)

    @staticmethod
    def _simple_synthesize(task: str, results: list[dict[str, Any]]) -> str:
        """Simple synthesis when LLM is not available."""
        parts = [f"任务: {task}", "", "各角色执行结果:"]
        for i, r in enumerate(results, 1):
            role = r.get("role", "unknown")
            result = r.get("result", "")
            status = r.get("status", "success")
            parts.append(f"\n{i}. [{role}] ({status}):\n{result}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def compiled_graph(self) -> Any:
        """Expose compiled graph for subgraph composition."""
        return self._compiled_graph

    def invoke(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        role_plan: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Invoke the multi-agent graph for a single task.

        Args:
            task: The main task to decompose and execute.
            context: Shared context passed to all roles.
            role_plan: Optional pre-defined role plan (skips coordinator).

        Returns:
            dict with synthesis, role_results, and metadata.
        """
        initial = _make_initial_state(task, role_plan=role_plan, context=context)

        try:
            result = self._compiled_graph.invoke(initial)
            return {
                "task": task,
                "synthesis": result.get("synthesis", ""),
                "role_results": result.get("role_results", []),
                "role_errors": result.get("role_errors", []),
                "duration": result.get("completed_at", 0) - result.get("started_at", 0),
                "status": "success",
            }
        except Exception as exc:
            logger.error("[MultiAgent] Graph invocation failed: %s", exc, exc_info=True)
            return {
                "task": task,
                "synthesis": f"执行失败: {exc}",
                "role_results": [],
                "role_errors": [{"error": str(exc)}],
                "duration": 0,
                "status": "failed",
            }

    def add_role(self, role: Role) -> None:
        """Dynamically add a new role without rebuilding the graph."""
        self._roles[role.name] = role
        logger.info("[MultiAgent] New role added: %s", role.name)

    def remove_role(self, name: str) -> bool:
        """Remove a role by name."""
        if name in self._roles:
            del self._roles[name]
            return True
        return False

    @property
    def available_roles(self) -> list[str]:
        return list(self._roles.keys())

    @property
    def roles(self) -> dict[str, Role]:
        return dict(self._roles)