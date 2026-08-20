"""Multi-agent role system — role-based collaboration.

Implements the role-based multi-agent pattern (CrewAI-style):
  - BaseAgentRole: role protocol with system prompt, scene, tool routing
  - Specialized roles: Researcher, Analyst, Executor, Coordinator
  - Collaboration: sequential chaining, parallel fan-out, LLM synthesis
  - MessageBus: inter-agent async communication

Architecture:
  AgentSession → CoordinatorRole → [Researcher, Analyst, Executor]
                                        ↓
                                   MessageBus (async)

Design alignment with mainstream frameworks:
  - CrewAI: role + task + crew pattern
  - AutoGen: message-driven agent communication
  - LangGraph: scene-based tool routing and plan-execute-synthesize
"""

from __future__ import annotations

import json
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
      - Declared capabilities (RoleCapability)
      - Specialized system prompt (_system_prompt) — injected as proper
        system message via ChatPromptTemplate, NOT appended to human text
      - Access to scene-routed tools or custom tool subsets
      - Message-based communication with other roles via MessageBus
      - Context chaining: results from previous roles are automatically
        passed as context to subsequent roles

    Subclasses must set _system_prompt and _scene, then implement execute()
    using _build_prompt() and _build_role_executor().
    """

    role_name: str = "base"
    role_description: str = "Base agent role"
    capabilities: list[RoleCapability] = []
    _system_prompt: str = ""
    _scene: str | None = None

    def __init__(
        self,
        name: str | None = None,
        llm: Any = None,
        tool_subset: list[Any] | None = None,
        message_bus: MessageBus | None = None,
        tool_router: Any | None = None,
    ) -> None:
        self.name = name or self.role_name
        self._llm = llm
        self._tools = tool_subset or []
        self._bus = message_bus
        self._router = tool_router
        self._context: dict[str, Any] = {}
        self._message_history: list[AgentMessage] = []
        self._shared_results: dict[str, dict[str, Any]] = {}

        if self._bus:
            self._bus.subscribe(self.name, self._on_message)

    def _on_message(self, message: AgentMessage) -> None:
        self._message_history.append(message)

    def execute(self, task: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        raise NotImplementedError

    def collaborate(self, target_role: str, task: str) -> dict[str, Any]:
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

    def receive_role_result(self, role_name: str, result: dict[str, Any]) -> None:
        self._shared_results[role_name] = result
        key = f"from_{role_name}"
        self._context[key] = result.get("result", "")
        self._context[f"{role_name}_status"] = result.get("status", "unknown")

    def get_shared_results(self) -> dict[str, dict[str, Any]]:
        return dict(self._shared_results)

    def _fallback_execute(
        self, task: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        cap_names = ", ".join(c.name for c in self.capabilities) or "general"
        return {
            "role": self.name,
            "action": self.role_name,
            "task": task,
            "result": f"[{self.role_description}] 任务: {task[:80]} | 能力: {cap_names}",
            "status": "skipped",
        }

    def _build_prompt(self, task: str, context: dict[str, Any] | None = None) -> str:
        parts: list[str] = [f"任务: {task}"]
        if context:
            ctx_str = json.dumps(context, ensure_ascii=False, default=str)[:500]
            parts.append(f"上下文: {ctx_str}")
        if self._shared_results:
            parts.append("")
            parts.append("前序角色执行结果:")
            for rname, rresult in self._shared_results.items():
                parts.append(f"  [{rname}]: {rresult.get('result', '')[:200]}")
        parts.append("")
        parts.append("请使用可用工具来完成任务。最终请用中文简洁总结执行结果。")
        return "\n".join(parts)

    def _build_role_executor(self) -> Any:
        """Build an executor with role-specific system prompt injected.

        Uses ChatPromptTemplate to properly separate system message from
        human message, aligning with LangChain best practices.
        """
        from agent.config import settings
        from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
        from langchain_core.prompts import ChatPromptTemplate
        from agent.tools.registry import list_scene_tools

        if self._tools:
            tools = self._tools
        else:
            tools = list_scene_tools(self._scene or "general")

        role_system = self._system_prompt or "你是一个智能 AI 助手。"
        tool_names = ", ".join(t.name for t in tools) if tools else "(无可用工具)"

        prompt = ChatPromptTemplate.from_messages([
            ("system", "{role_system}\n\n你可以使用以下工具：\n\n{tool_names}\n\n在需要时使用工具来帮助用户完成任务。始终保持友好和简洁。\n\n重要：你必须始终使用中文回复。"),
            ("human", "对话历史：\n{chat_history}\n\n用户问题：\n{input}\n\n请逐步思考并使用适当的工具来获得最佳答案。用中文回复。"),
            ("placeholder", "{agent_scratchpad}"),
        ]).partial(role_system=role_system, tool_names=tool_names)

        agent = create_tool_calling_agent(self._llm, tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            max_iterations=settings.MAX_ITERATIONS,
            max_execution_time=settings.MAX_EXECUTION_TIME_SEC,
            handle_parsing_errors=True,
        )

    def _run_with_tools(self, task: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        from agent.runner import run_react_step

        prompt = self._build_prompt(task, context)
        executor = self._build_role_executor()
        return run_react_step(executor, prompt)


class ResearcherRole(BaseAgentRole):
    """Research agent — gathers information from web, files, and databases.

    Uses the "research" scene (web + summarize + translate + core tools)
    to perform real LLM-powered research with tool calls.
    """

    role_name = "researcher"
    role_description = "Gathers information and researches topics comprehensively"
    _system_prompt = (
        "你是一个专业的研究助手。你擅长搜索互联网信息、阅读文档、翻译外文、"
        "以及对研究结果进行总结。你必须使用可用工具来查找和验证信息，"
        "并以清晰结构化的方式呈现研究成果。"
    )
    _scene = "research"
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

        if self._llm is None:
            fallback = self._fallback_execute(task, context)
            fallback.update({"action": "research", "sources": []})
            return fallback

        try:
            result = self._run_with_tools(task, context)
            output_text = result.get("result", "")
        except Exception as exc:
            logger.warning("[Researcher] Execution failed: %s", exc)
            output_text = f"研究执行出错: {exc}"

        return {
            "role": self.name,
            "action": "research",
            "task": task,
            "result": output_text[:800],
            "sources": [],
            "status": "completed",
        }


class AnalystRole(BaseAgentRole):
    """Analysis agent — processes data and generates insights.

    Uses the "analysis" scene (code + file + core tools) to perform
    real LLM-powered data analysis with tool calls.
    """

    role_name = "analyst"
    role_description = "Analyzes data and generates actionable insights"
    _system_prompt = (
        "你是一个专业的数据分析助手。你擅长处理结构化和非结构化数据、"
        "识别趋势和模式、进行对比分析、以及生成分析报告。"
        "你必须使用可用工具来读取数据、执行计算、并将分析结果以清晰的方式呈现。"
    )
    _scene = "analysis"
    capabilities = [
        RoleCapability("data_analysis", "Analyze structured/unstructured data"),
        RoleCapability("trend_detection", "Identify trends and patterns"),
        RoleCapability("comparison", "Compare and contrast datasets"),
        RoleCapability("reporting", "Generate analysis reports"),
    ]

    def execute(self, task: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._llm is None:
            fallback = self._fallback_execute(task, context)
            fallback.update({"action": "analyze", "insights": []})
            return fallback

        try:
            result = self._run_with_tools(task, context)
            output_text = result.get("result", "")
        except Exception as exc:
            logger.warning("[Analyst] Execution failed: %s", exc)
            output_text = f"分析执行出错: {exc}"

        return {
            "role": self.name,
            "action": "analyze",
            "task": task,
            "result": output_text[:800],
            "insights": [],
            "status": "completed",
        }


class ExecutorRole(BaseAgentRole):
    """Execution agent — performs concrete operations (file I/O, code, commands).

    Uses the "execution" scene (file + code + dev + git + shell + core)
    to perform real LLM-powered execution with tool calls.
    """

    role_name = "executor"
    role_description = "Executes concrete operations and produces deliverables"
    _system_prompt = (
        "你是一个专业的执行助手。你擅长读写文件、执行代码脚本、"
        "自动化重复任务、以及生成交付物（如图表、报告、文档等）。"
        "你必须使用可用工具来完成具体操作，并确保操作结果可靠。"
    )
    _scene = "execution"
    capabilities = [
        RoleCapability("file_operations", "Read, write, and modify files"),
        RoleCapability("code_execution", "Run code and scripts"),
        RoleCapability("task_automation", "Automate repetitive tasks"),
        RoleCapability("output_generation", "Generate charts, reports, documents"),
    ]

    def execute(self, task: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._llm is None:
            fallback = self._fallback_execute(task, context)
            fallback.update({"action": "execute", "artifacts": []})
            return fallback

        try:
            result = self._run_with_tools(task, context)
            output_text = result.get("result", "")
        except Exception as exc:
            logger.warning("[Executor] Execution failed: %s", exc)
            output_text = f"执行出错: {exc}"

        return {
            "role": self.name,
            "action": "execute",
            "task": task,
            "result": output_text[:800],
            "artifacts": [],
            "status": "completed",
        }


class CoordinatorRole(BaseAgentRole):
    """Coordinator — orchestrates multi-role collaboration.

    Responsible for:
      1. Decomposing complex tasks into sub-tasks (LLM-first, keyword fallback)
      2. Assigning sub-tasks to appropriate roles
      3. Managing sequential/parallel execution with context chaining
      4. LLM-based result synthesis and conflict resolution
      5. Producing a coherent final answer
    """

    role_name = "coordinator"
    role_description = "Orchestrates multi-role collaboration and synthesizes results"
    _system_prompt = (
        "你是一个任务分配专家。根据用户的请求，将任务分解为以下角色可执行的子任务："
        "researcher（研究/搜索/信息收集）、analyst（分析/计算/数据处理）、"
        "executor（执行/创建/文件操作）。"
        "请以 JSON 格式输出子任务列表，每个元素包含 role 和 task 字段。"
    )
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
        sub_tasks = self._decompose_by_llm(task)
        if sub_tasks:
            return sub_tasks
        return self._decompose_by_keywords(task)

    def _decompose_by_llm(self, task: str) -> list[dict[str, str]] | None:
        if self._llm is None:
            return None

        prompt = f"""{self._system_prompt}

用户请求: {task}

可用角色:
- researcher: 研究助手，擅长搜索、阅读、翻译、总结
- analyst: 分析助手，擅长数据处理、计算、对比、报告
- executor: 执行助手，擅长文件操作、代码执行、自动化、生成交付物

请输出 JSON 格式的子任务列表，例如:
[{{"role": "researcher", "task": "搜索相关资料"}}, {{"role": "analyst", "task": "分析数据趋势"}}]"""

        try:
            response = self._llm.invoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
            parsed = json.loads(text)
            if isinstance(parsed, list) and all(
                isinstance(item, dict) and "role" in item and "task" in item
                for item in parsed
            ):
                valid_roles = {"researcher", "analyst", "executor"}
                return [p for p in parsed if p["role"] in valid_roles]
        except (json.JSONDecodeError, Exception) as exc:
            logger.debug("[Coordinator] LLM decompose failed: %s", exc)
        return None

    def _decompose_by_keywords(self, task: str) -> list[dict[str, str]]:
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
        sub_tasks = self.decompose_task(task)
        results: list[dict[str, Any]] = []

        shared_context = dict(context) if context else {}

        for sub in sub_tasks:
            role_name = sub["role"]
            if role_name in self._roles:
                try:
                    role = self._roles[role_name]
                    for prev_role, prev_result in self._roles.items():
                        if prev_role != role_name and prev_role in [r.get("role", "") for r in results]:
                            role.receive_role_result(prev_role, self._last_result_for(prev_role, results))

                    result = role.execute(
                        sub["task"], context=shared_context
                    )
                    results.append(result)

                    self._propagate_result_to_others(role_name, result)

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

        return self._synthesize(task, results)

    @staticmethod
    def _last_result_for(role_name: str, results: list[dict[str, Any]]) -> dict[str, Any]:
        for r in reversed(results):
            if r.get("role") == role_name:
                return r
        return {}

    def _propagate_result_to_others(
        self, from_role: str, result: dict[str, Any]
    ) -> None:
        for role_name, role in self._roles.items():
            if role_name != from_role:
                role.receive_role_result(from_role, result)

    def _synthesize(
        self, original_task: str, results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if self._llm is not None and len(results) > 0:
            synthesized = self._synthesize_via_llm(original_task, results)
            if synthesized:
                return synthesized

        return self._synthesize_via_template(original_task, results)

    def _synthesize_via_llm(
        self, original_task: str, results: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        role_summaries: list[str] = []
        for result in results:
            role = result.get("role", "unknown")
            if "error" in result:
                role_summaries.append(f"- **{role}**: 执行失败 ({result['error']})")
            else:
                role_summaries.append(f"- **{role}**: {result.get('result', '')[:400]}")

        prompt = f"""你是一个协作结果综合专家。请根据以下多个 AI 角色的执行结果，综合出一份连贯、简洁的最终答案。

用户原始请求: {original_task}

各角色执行结果:
{chr(10).join(role_summaries)}

请:
1. 综合各角色的核心发现，去除重复内容
2. 如有冲突信息，标注并给出最合理的判断
3. 以中文输出最终答案，结构清晰，重点突出
4. 如果某角色失败，基于其他角色的结果尽量给出完整答案"""

        try:
            response = self._llm.invoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)
            synthesized_text = text.strip()

            return {
                "role": self.name,
                "action": "coordinate",
                "task": original_task,
                "synthesized_result": synthesized_text,
                "participating_roles": [r.get("role", "") for r in results],
                "success_count": sum(1 for r in results if "error" not in r),
                "total_count": len(results),
                "synthesis_method": "llm",
            }
        except Exception as exc:
            logger.warning("[Coordinator] LLM synthesis failed: %s", exc)
            return None

    def _synthesize_via_template(
        self, original_task: str, results: list[dict[str, Any]]
    ) -> dict[str, Any]:
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
            "synthesis_method": "template",
        }

    def synthesize(
        self, original_task: str, results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return self._synthesize(original_task, results)

    def get_collaboration_history(self) -> list[dict[str, Any]]:
        return self._task_log[-20:]


def create_default_team(
    llm: Any = None,
    message_bus: MessageBus | None = None,
    tool_overrides: dict[str, list[Any]] | None = None,
    extra_roles: list[BaseAgentRole] | None = None,
    shared_router: Any | None = None,
) -> CoordinatorRole:
    """Create a default multi-agent team with standard roles.

    Args:
        llm: Language model instance (shared across all roles).
        message_bus: Optional shared MessageBus instance.
        tool_overrides: Optional dict mapping role names to custom tool lists.
                       e.g. {"researcher": [custom_tool]}
        extra_roles: Additional roles to register beyond the defaults.
        shared_router: Optional shared ToolRouter instance to avoid duplication.

    Returns:
        Configured CoordinatorRole with all sub-roles registered.
    """
    bus = message_bus or MessageBus()
    coordinator = CoordinatorRole(llm=llm, message_bus=bus, tool_router=shared_router)

    tool_overrides = tool_overrides or {}

    researcher = ResearcherRole(
        llm=llm, message_bus=bus,
        tool_subset=tool_overrides.get("researcher"),
        tool_router=shared_router,
    )
    analyst = AnalystRole(
        llm=llm, message_bus=bus,
        tool_subset=tool_overrides.get("analyst"),
        tool_router=shared_router,
    )
    executor = ExecutorRole(
        llm=llm, message_bus=bus,
        tool_subset=tool_overrides.get("executor"),
        tool_router=shared_router,
    )

    coordinator.register_role(researcher)
    coordinator.register_role(analyst)
    coordinator.register_role(executor)

    if extra_roles:
        for role in extra_roles:
            coordinator.register_role(role)

    return coordinator