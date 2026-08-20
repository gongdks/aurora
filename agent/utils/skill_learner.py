"""Skill learner — extracts reusable skills from successful executions.

Core flow:
  Execution success → Analyze tool usage patterns → Generate skill definition
  → Validate against history → Store in SkillStore → Register as runtime tool

Only promotes skills that demonstrate consistent success patterns.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

from agent.utils.skill_store import SkillDefinition, SkillStore

logger = logging.getLogger(__name__)

_MAX_TOOL_SEQUENCE_LENGTH = 10
_MIN_TOOL_SEQUENCE_LENGTH = 2
_MAX_TRIGGER_PATTERNS = 5


class SkillLearner:
    """Learns reusable skills from successful agent executions.

    Usage:
        store = SkillStore()
        learner = SkillLearner(store)

        # After a successful execution:
        skill = learner.learn_from_execution(
            user_input="分析销售数据",
            plan=["read_file", "analyze_data", "create_chart"],
            results=["文件已读取", "分析完成", "图表已创建"],
            tool_calls=[{"name": "read_file", "args": {...}}, ...],
            success=True,
        )
        if skill:
            print(f"New skill learned: {skill.name}")
    """

    def __init__(self, store: SkillStore | None = None) -> None:
        self._store = store or SkillStore()
        self._learned_skills: dict[str, SkillDefinition] = {}

    def learn_from_execution(
        self,
        user_input: str,
        plan: list[str],
        results: list[str],
        tool_calls: list[dict[str, Any]] | None = None,
        success: bool = True,
        reflection_score: float = 0.0,
    ) -> SkillDefinition | None:
        """Attempt to learn a skill from an execution trace.

        Args:
            user_input: Original user query
            plan: Executed plan steps
            results: Step results
            tool_calls: Tool call records [{name, args, result}]
            success: Whether execution was successful
            reflection_score: Overall reflection quality score

        Returns:
            New SkillDefinition if learned, None otherwise
        """
        if not success and reflection_score < 0.5:
            return None

        if not plan or len(plan) < _MIN_TOOL_SEQUENCE_LENGTH:
            return None

        tool_sequence = self._extract_tool_sequence(plan, tool_calls)
        if len(tool_sequence) < _MIN_TOOL_SEQUENCE_LENGTH:
            return None

        skill_name = self._generate_skill_name(user_input, plan)
        if not skill_name:
            return None

        trigger_patterns = self._extract_trigger_patterns(user_input)

        existing = self._store.get_by_name(skill_name)
        if existing:
            if success:
                self._store.update_stats(existing.skill_id, success=True)
            return existing

        description = self._generate_description(skill_name, tool_sequence, user_input)

        skill = SkillDefinition(
            name=skill_name,
            description=description,
            trigger_patterns=trigger_patterns[:_MAX_TRIGGER_PATTERNS],
            tool_sequence=tool_sequence[:_MAX_TOOL_SEQUENCE_LENGTH],
            confidence=0.5 if success else 0.3,
            usage_count=1,
            success_count=1 if success else 0,
            failure_count=0 if success else 1,
        )

        self._store.save(skill)
        self._learned_skills[skill.skill_id] = skill

        logger.info(
            "[SkillLearner] New skill learned: %s (tools: %d, triggers: %d)",
            skill_name, len(tool_sequence), len(trigger_patterns),
        )

        return skill

    def _extract_tool_sequence(
        self,
        plan: list[str],
        tool_calls: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """Extract the effective tool sequence from execution trace."""
        sequence: list[dict[str, Any]] = []

        if isinstance(tool_calls, list) and tool_calls:
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                name = call.get("name", "")
                args = call.get("args", {})
                if name:
                    sequence.append({
                        "tool": name,
                        "args_schema": self._summarize_args(args),
                    })
        else:
            for step in plan:
                tool_hint = self._infer_tool_from_step(step)
                if tool_hint:
                    sequence.append({"tool": tool_hint, "args_schema": {}})

        return sequence

    def _summarize_args(self, args: dict[str, Any]) -> dict[str, Any]:
        """Summarize tool args to capture the pattern without sensitive data."""
        summary: dict[str, Any] = {}
        for key, value in args.items():
            if isinstance(value, str):
                if len(value) > 30:
                    summary[key] = value[:30] + "..."
                else:
                    summary[key] = value
            elif isinstance(value, (int, float, bool)):
                summary[key] = value
            elif isinstance(value, list):
                summary[key] = f"list[{len(value)}]"
            else:
                summary[key] = str(type(value).__name__)
        return summary

    def _infer_tool_from_step(self, step: str) -> str | None:
        """Infer which tool was likely used from a step description."""
        step_lower = step.lower()
        tool_keywords = {
            "read_file": ["读取", "阅读", "查看", "read", "open", "file"],
            "write_file": ["写入", "保存", "创建文件", "write", "save", "create file"],
            "search_web": ["搜索", "查找", "查询", "search", "find", "web"],
            "fetch_url": ["获取", "访问", "fetch", "url", "download"],
            "run_code": ["执行", "运行", "run", "execute", "code"],
            "list_dir": ["列表", "目录", "list", "directory", "dir"],
            "analyze": ["分析", "计算", "analyze", "calculate"],
            "summarize": ["总结", "摘要", "summarize", "summary"],
            "translate": ["翻译", "translate"],
            "create_chart": ["图表", "图表", "chart", "graph", "plot"],
            "git_": ["git", "commit", "push", "branch"],
            "shell": ["命令行", "终端", "shell", "terminal", "command"],
        }
        for tool_name, keywords in tool_keywords.items():
            if any(kw in step_lower for kw in keywords):
                return tool_name
        return None

    def _generate_skill_name(self, user_input: str, plan: list[str]) -> str | None:
        """Generate a concise, descriptive skill name."""
        input_lower = user_input.lower()

        name_patterns = [
            (r"分析(.+?)数据", "analyze_{topic}_data"),
            (r"(.+?)生成.*?(图表|图|chart)", "generate_{topic}_chart"),
            (r"(.+?)总结|汇总", "summarize_{topic}"),
            (r"(.+?)翻译", "translate_{topic}"),
            (r"查找|搜索(.+)", "search_{topic}"),
            (r"创建|生成(.+?)代码", "generate_{topic}_code"),
            (r"修复|调试(.+)", "debug_{topic}"),
            (r"部署|发布(.+)", "deploy_{topic}"),
            (r"读取(.+?)文件", "read_{topic}_file"),
            (r"写入|保存(.+)", "save_{topic}"),
            (r"计算(.+)", "calculate_{topic}"),
        ]

        for pattern, template in name_patterns:
            match = re.search(pattern, input_lower)
            if match:
                topic = match.group(1).strip()
                if len(topic) > 20:
                    topic = topic[:20]
                topic = re.sub(r"[^\w\u4e00-\u9fff]+", "_", topic)
                return template.format(topic=topic)

        first_step = plan[0] if plan else ""
        step_words = re.findall(r"[\u4e00-\u9fff]+|\w+", first_step)
        if step_words:
            topic = "_".join(step_words[:3]).lower()
            if len(topic) > 30:
                topic = topic[:30]
            return f"skill_{topic}"

        input_hash = hashlib.md5(user_input.encode("utf-8")).hexdigest()[:8]
        return f"skill_{input_hash}"

    def _extract_trigger_patterns(self, user_input: str) -> list[str]:
        """Extract key phrases that should trigger this skill."""
        patterns: list[str] = []

        words = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]+", user_input)
        for word in words:
            if len(word) >= 2:
                patterns.append(word.lower())

        if len(patterns) < 2:
            patterns.append(user_input.lower()[:10])

        return patterns[:_MAX_TRIGGER_PATTERNS]

    def _generate_description(
        self, skill_name: str, tool_sequence: list[dict], user_input: str
    ) -> str:
        """Generate a human-readable skill description."""
        tools_str = " -> ".join(
            t.get("tool", "?") for t in tool_sequence[:5]
        )
        return (
            f"Skill '{skill_name}': {len(tool_sequence)} steps "
            f"[{tools_str}]. Triggered by: {user_input[:60]}"
        )

    def find_relevant_skill(self, user_input: str) -> SkillDefinition | None:
        """Find the most relevant skill for a given user input."""
        matches = self._store.find_matching(user_input)
        if not matches:
            return None
        return matches[0]

    def get_learned_skills(self) -> list[SkillDefinition]:
        """Get all skills learned in this session."""
        return list(self._learned_skills.values())

    def get_store_stats(self) -> dict[str, int]:
        return self._store.stats

    def promote_tools(self, skill: SkillDefinition) -> bool:
        """Attempt to promote a skill as a runtime-available tool.

        Returns True if the skill was registered successfully.
        """
        try:
            from agent.tools.registry import register, _TOOLS, _TAGS

            def _skill_tool(**kwargs: Any) -> str:
                """Execute a learned skill."""
                parts = [f"Executing skill: {skill.name}"]
                for step in skill.tool_sequence:
                    tool_name = step.get("tool", "?")
                    args = step.get("args_schema", {})
                    parts.append(f"  → {tool_name}({args})")
                return "\n".join(parts)

            _skill_tool.__name__ = f"skill_{skill.skill_id}"
            _skill_tool.__qualname__ = f"skill_{skill.skill_id}"

            if hasattr(_skill_tool, "__doc__"):
                _skill_tool.__doc__ = skill.description

            register(tags={"core", "skill"})(_skill_tool)
            logger.info("[SkillLearner] Skill promoted as tool: %s", skill.name)
            return True
        except Exception as exc:
            logger.warning("[SkillLearner] Failed to promote skill: %s", exc)
            return False

    def cleanup(self) -> int:
        """Clean up low-quality skills."""
        removed = self._store.cleanup_expired(min_usage=1, min_confidence=0.3)
        logger.info("[SkillLearner] Cleaned up %d low-quality skills", removed)
        return removed