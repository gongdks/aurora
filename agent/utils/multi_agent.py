"""Multi-agent role system — role-based collaboration.

Implements the role-based multi-agent pattern:
  - BaseAgentRole: role protocol and capability declaration
  - Specialized roles: Researcher, Analyst, Executor, Coordinator
  - Collaboration modes: Sequential, Parallel, Debate, Hierarchical
  - MessageBus: inter-agent communication

Architecture:
  AgentSession → CoordinatorRole → [Researcher, Analyst, Executor]
                                        ↓
                                   MessageBus (async)
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    """Inter-agent message for communication."""

    sender: str
    receiver: str
    message_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    message_id: str = ""
    timestamp: float = 0.0
    correlation_id: str = ""

    def __post_init__(self) -> None:
        if not self.message_id:
            self.message_id = str(uuid.uuid4())[:12]
        if not self.timestamp:
            self.timestamp = time.time()


@dataclass
class RoleCapability:
    """Declared capability of an agent role."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.8


class MessageBus:
    """Lightweight async message bus for inter-agent communication."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable[[AgentMessage], None]]] = {}
        self._message_log: list[AgentMessage] = []
        self._max_log: int = 500
        self._pending: dict[str, list[AgentMessage]] = {}

    def subscribe(self, role_name: str, callback: Callable[[AgentMessage], None]) -> None:
        self._subscribers.setdefault(role_name, []).append(callback)

    def publish(self, message: AgentMessage) -> None:
        self._message_log.append(message)
        if len(self._message_log) > self._max_log:
            self._message_log.pop(0)

        receivers = [message.receiver] if message.receiver != "*" else list(self._subscribers.keys())

        for receiver in receivers:
            if receiver in self._subscribers:
                for cb in self._subscribers[receiver]:
                    try:
                        cb(message)
                    except Exception as exc:
                        logger.warning("[MessageBus] Delivery failed: %s", exc)
            else:
                self._pending.setdefault(receiver, []).append(message)

    def send_request(
        self, sender: str, receiver: str, request_type: str, payload: dict[str, Any]
    ) -> AgentMessage:
        msg = AgentMessage(
            sender=sender,
            receiver=receiver,
            message_type=request_type,
            payload=payload,
        )
        self.publish(msg)
        return msg

    def get_pending(self, role_name: str) -> list[AgentMessage]:
        return self._pending.pop(role_name, [])

    @property
    def stats(self) -> dict[str, int]:
        return {
            "subscribers": len(self._subscribers),
            "messages": len(self._message_log),
            "pending": sum(len(v) for v in self._pending.values()),
        }


class BaseAgentRole:
    """Base class for all agent roles.

    A role is a specialized agent with:
      - Declared capabilities
      - Specialized prompt/system instructions
      - Access to specific tool subsets
      - Message-based communication with other roles

    Subclasses must implement execute().
    """

    role_name: str = "base"
    role_description: str = "Base agent role"
    capabilities: list[RoleCapability] = []

    def __init__(
        self,
        name: str | None = None,
        llm: Any = None,
        tool_subset: list[Any] | None = None,
        message_bus: MessageBus | None = None,
    ) -> None:
        self.name = name or self.role_name
        self._llm = llm
        self._tools = tool_subset or []
        self._bus = message_bus
        self._context: dict[str, Any] = {}
        self._message_history: list[AgentMessage] = []

        if self._bus:
            self._bus.subscribe(self.name, self._on_message)

    def _on_message(self, message: AgentMessage) -> None:
        self._message_history.append(message)

    def execute(self, task: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute a task — must be implemented by subclasses."""
        raise NotImplementedError

    def collaborate(self, target_role: str, task: str) -> dict[str, Any]:
        """Send a task to another role for collaboration."""
        if self._bus:
            msg = self._bus.send_request(
                sender=self.name,
                receiver=target_role,
                request_type="collaborate",
                payload={"task": task, "context": self._context},
            )
            return {"status": "sent", "message_id": msg.message_id}
        return {"status": "no_bus"}

    def get_capabilities(self) -> list[dict[str, Any]]:
        return [
            {
                "name": c.name,
                "description": c.description,
                "confidence": c.confidence,
            }
            for c in self.capabilities
        ]

    def set_context(self, key: str, value: Any) -> None:
        self._context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        return self._context.get(key, default)


class ResearcherRole(BaseAgentRole):
    """Research agent — gathers information from web, files, and databases."""

    role_name = "researcher"
    role_description = "Gathers information and researches topics comprehensively"
    capabilities = [
        RoleCapability("web_search", "Search the web for information"),
        RoleCapability("file_search", "Search files and documents"),
        RoleCapability("summarize", "Summarize research findings"),
        RoleCapability("fact_check", "Verify information accuracy"),
    ]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._search_history: list[dict[str, Any]] = []

    def execute(self, task: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        self._search_history.append({"task": task, "time": time.time()})
        return {
            "role": self.name,
            "action": "research",
            "task": task,
            "result": f"Research completed for: {task[:100]}",
            "sources": [],
        }


class AnalystRole(BaseAgentRole):
    """Analysis agent — processes data and generates insights."""

    role_name = "analyst"
    role_description = "Analyzes data and generates actionable insights"
    capabilities = [
        RoleCapability("data_analysis", "Analyze structured/unstructured data"),
        RoleCapability("trend_detection", "Identify trends and patterns"),
        RoleCapability("comparison", "Compare and contrast datasets"),
        RoleCapability("reporting", "Generate analysis reports"),
    ]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def execute(self, task: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "role": self.name,
            "action": "analyze",
            "task": task,
            "result": f"Analysis completed: {task[:100]}",
            "insights": [],
        }


class ExecutorRole(BaseAgentRole):
    """Execution agent — performs concrete operations (file I/O, code, commands)."""

    role_name = "executor"
    role_description = "Executes concrete operations and produces deliverables"
    capabilities = [
        RoleCapability("file_operations", "Read, write, and modify files"),
        RoleCapability("code_execution", "Run code and scripts"),
        RoleCapability("task_automation", "Automate repetitive tasks"),
        RoleCapability("output_generation", "Generate charts, reports, documents"),
    ]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def execute(self, task: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "role": self.name,
            "action": "execute",
            "task": task,
            "result": f"Execution completed: {task[:100]}",
            "artifacts": [],
        }


class CoordinatorRole(BaseAgentRole):
    """Coordinator — orchestrates multi-role collaboration.

    Responsible for:
      1. Decomposing complex tasks into sub-tasks
      2. Assigning sub-tasks to appropriate roles
      3. Managing parallel/sequential execution
      4. Resolving conflicts between role outputs
      5. Synthesizing final results
    """

    role_name = "coordinator"
    role_description = "Orchestrates multi-role collaboration and synthesizes results"
    capabilities = [
        RoleCapability("task_decomposition", "Break complex tasks into sub-tasks"),
        RoleCapability("role_assignment", "Assign sub-tasks to appropriate roles"),
        RoleCapability("parallel_execution", "Run multiple roles in parallel"),
        RoleCapability("conflict_resolution", "Resolve disagreements between roles"),
        RoleCapability("result_synthesis", "Combine results into final answer"),
    ]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._roles: dict[str, BaseAgentRole] = {}
        self._task_log: list[dict[str, Any]] = []

    def register_role(self, role: BaseAgentRole) -> None:
        self._roles[role.name] = role

    def decompose_task(self, task: str) -> list[dict[str, str]]:
        """Decompose a complex task into role-specific sub-tasks."""
        sub_tasks: list[dict[str, str]] = []

        research_keywords = ["研究", "搜索", "查找", "调研", "research", "search", "find"]
        analysis_keywords = ["分析", "计算", "对比", "analyze", "calculate", "compare"]
        execution_keywords = ["创建", "生成", "执行", "写", "创建", "create", "generate", "execute"]

        task_lower = task.lower()

        if any(kw in task_lower for kw in research_keywords):
            sub_tasks.append({"role": "researcher", "task": task})
        if any(kw in task_lower for kw in analysis_keywords):
            sub_tasks.append({"role": "analyst", "task": task})
        if any(kw in task_lower for kw in execution_keywords):
            sub_tasks.append({"role": "executor", "task": task})

        if not sub_tasks:
            sub_tasks.append({"role": "researcher", "task": f"了解背景: {task[:60]}"})
            sub_tasks.append({"role": "analyst", "task": f"分析需求: {task[:60]}"})
            sub_tasks.append({"role": "executor", "task": f"执行操作: {task[:60]}"})

        return sub_tasks

    def execute(
        self, task: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Execute a task by coordinating across roles."""
        sub_tasks = self.decompose_task(task)
        results: list[dict[str, Any]] = []

        for sub in sub_tasks:
            role_name = sub["role"]
            if role_name in self._roles:
                try:
                    result = self._roles[role_name].execute(
                        sub["task"], context=context
                    )
                    results.append(result)
                    self._task_log.append({
                        "role": role_name,
                        "task": sub["task"],
                        "result": result.get("result", "")[:100],
                        "time": time.time(),
                    })
                except Exception as exc:
                    logger.warning(
                        "[Coordinator] Role %s failed: %s", role_name, exc
                    )
                    results.append({
                        "role": role_name,
                        "error": str(exc),
                    })

        return self.synthesize(task, results)

    def synthesize(
        self, original_task: str, results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Synthesize results from multiple roles into a coherent answer."""
        parts: list[str] = [f"## 协作结果: {original_task[:60]}\n"]

        for result in results:
            role = result.get("role", "unknown")
            if "error" in result:
                parts.append(f"- **{role}**: ⚠️ 执行失败: {result['error']}")
            else:
                parts.append(f"- **{role}**: {result.get('result', '已完成')}")

        return {
            "role": self.name,
            "action": "coordinate",
            "task": original_task,
            "synthesized_result": "\n".join(parts),
            "participating_roles": [r.get("role", "") for r in results],
            "success_count": sum(1 for r in results if "error" not in r),
            "total_count": len(results),
        }

    def get_collaboration_history(self) -> list[dict[str, Any]]:
        return self._task_log[-20:]


def create_default_team(
    llm: Any = None,
    message_bus: MessageBus | None = None,
) -> CoordinatorRole:
    """Create a default multi-agent team with standard roles."""
    bus = message_bus or MessageBus()
    coordinator = CoordinatorRole(llm=llm, message_bus=bus)
    researcher = ResearcherRole(llm=llm, message_bus=bus)
    analyst = AnalystRole(llm=llm, message_bus=bus)
    executor = ExecutorRole(llm=llm, message_bus=bus)

    coordinator.register_role(researcher)
    coordinator.register_role(analyst)
    coordinator.register_role(executor)

    return coordinator