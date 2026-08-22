"""Graph-based skill learner using LangGraph StateGraph.

Replaces the procedural skill_learner.py with a stateful graph that models
skill extraction, validation, and storage as explicit nodes with conditional
routing and checkpoint support.
"""
from __future__ import annotations

import logging
import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .skill_store import SkillDefinition, SkillStore, _DEFAULT_SKILLS_DIR
import os

logger = logging.getLogger(__name__)

_MAX_TOOL_SEQUENCE_LENGTH = 8
_MAX_TRIGGER_PATTERNS = 5


class SkillLearnerState(TypedDict, total=False):
    user_input: str
    plan: list[str]
    results: list[str]
    tool_calls: list[dict[str, Any]] | None
    success: bool
    reflection_score: float
    tool_sequence: list[dict[str, Any]]
    trigger_patterns: list[str]
    tool_schemas: list[dict[str, Any]]
    confidence: float
    should_store: bool
    stored_skill: dict[str, Any] | None
    error: str | None
    existing_skill_names: list[str]
    derived_skill_name: str


class SkillLearnerGraph:
    """LangGraph-based skill learner that extracts and stores reusable skills.

    Replaces the procedural approach with a stateful graph that:
    - Extracts tool sequences from successful executions
    - Derives trigger patterns from user input
    - Validates and scores skill candidates
    - Stores validated skills with checkpoint persistence

    Nodes:
        extract_sequence -> derive_patterns -> validate -> (store or skip) -> finalize

    Usage:
        learner = SkillLearnerGraph()
        skill = learner.learn_from_execution(user_input, plan, results, tool_calls)
    """

    def __init__(
        self,
        store: SkillStore | None = None,
    ):
        self._store = store or SkillStore()
        self._graph = self._build_graph()

    @property
    def compiled_graph(self) -> Any:
        """Expose compiled graph for subgraph composition."""
        return self._graph

    def _build_graph(self):
        graph = StateGraph(SkillLearnerState)
        graph.add_node("extract_sequence", self._node_extract_sequence)
        graph.add_node("derive_patterns", self._node_derive_patterns)
        graph.add_node("validate", self._node_validate)
        graph.add_node("store", self._node_store)
        graph.add_node("skip", self._node_skip)
        graph.add_node("finalize", self._node_finalize)
        graph.add_edge(START, "extract_sequence")
        graph.add_edge("extract_sequence", "derive_patterns")
        graph.add_edge("derive_patterns", "validate")
        graph.add_conditional_edges(
            "validate",
            self._route_after_validate,
            {
                "store": "store",
                "skip": "skip",
            },
        )
        graph.add_edge("store", "finalize")
        graph.add_edge("skip", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile()

    def learn_from_execution(
        self,
        user_input: str,
        plan: list[str],
        results: list[str],
        tool_calls: list[dict[str, Any]] | None = None,
        success: bool = True,
        reflection_score: float = 0.0,
    ) -> SkillDefinition | None:
        try:
            existing_names = [sk.name for sk in self._store.list_all()]

            initial_state: SkillLearnerState = {
                "user_input": user_input,
                "plan": plan,
                "results": results,
                "tool_calls": tool_calls,
                "success": success,
                "reflection_score": reflection_score,
                "existing_skill_names": existing_names,
            }
            result = self._graph.invoke(initial_state)

            if result.get("stored_skill"):
                skill_def = SkillDefinition(**result["stored_skill"])
                self._store.save(skill_def)
                logger.info(f"Saved skill '{skill_def.name}' with confidence {skill_def.confidence}")
                return skill_def
            return None
        except Exception as e:
            logger.error(f"Skill learning graph failed: {e}")
            return None

    def _node_extract_sequence(self, state: SkillLearnerState) -> dict[str, Any]:
        tool_calls = state.get("tool_calls")
        if not tool_calls:
            if state.get("success"):
                return {"error": "No tool calls to learn from"}
            return {}

        seen: set[str] = set()
        sequence: list[dict[str, Any]] = []
        for call in tool_calls:
            tool_name = call.get("name", call.get("tool", ""))
            if not tool_name or tool_name in seen:
                continue
            seen.add(tool_name)
            sequence.append({
                "tool": tool_name,
                "args": self._extract_args(call),
            })
            if len(sequence) >= _MAX_TOOL_SEQUENCE_LENGTH:
                break

        logger.info(f"Extracted tool sequence with {len(sequence)} steps")
        return {"tool_sequence": sequence}

    def _node_derive_patterns(self, state: SkillLearnerState) -> dict[str, Any]:
        user_input = state.get("user_input", "").strip()
        if not user_input:
            return {}

        patterns: list[str] = []
        words = user_input.split()
        if len(words) <= 6:
            patterns.append(user_input.lower())
        else:
            patterns.append(" ".join(words[:4]).lower())
            patterns.append(" ".join(words[:3]).lower())

        prefixes = [
            "我想知道", "请告诉我", "帮我查一下", "帮我看看",
            "什么是", "如何", "怎么", "为什么",
            "查询", "搜索", "查找", "获取", "生成",
            "分析", "比较", "评估", "总结", "预测",
            "写", "创建", "生成", "构建", "设计",
        ]
        lower_input = user_input.lower()
        for prefix in prefixes:
            if prefix in lower_input:
                idx = lower_input.find(prefix) + len(prefix)
                remainder = user_input[idx:].strip()
                if remainder:
                    patterns.append(f"{prefix}{{subject}}".lower())

        return {"trigger_patterns": patterns[:_MAX_TRIGGER_PATTERNS]}

    def _node_validate(self, state: SkillLearnerState) -> dict[str, Any]:
        if state.get("error"):
            return {"should_store": False}

        if not state.get("success") and state.get("reflection_score", 0) < 0.3:
            return {"should_store": False}

        if not state.get("tool_sequence"):
            return {"should_store": False}

        if state.get("success") and state.get("reflection_score", 0) >= 0.7:
            confidence = 0.8
        elif state.get("success"):
            confidence = 0.5
        elif state.get("reflection_score", 0) >= 0.5:
            confidence = 0.35
        else:
            confidence = 0.2

        return {
            "confidence": confidence,
            "should_store": confidence >= 0.2,
        }

    def _node_store(self, state: SkillLearnerState) -> dict[str, Any]:
        if not state.get("should_store") or not state.get("tool_sequence"):
            return {"stored_skill": None}

        skill_name = self._derive_skill_name(state)

        skill = SkillDefinition(
            name=skill_name,
            description=f"Skill: {state.get('user_input', '')[:80]}",
            trigger_patterns=state.get("trigger_patterns", []),
            tool_sequence=state.get("tool_sequence", []),
            confidence=state.get("confidence", 0.0),
            usage_count=1,
            success_count=1 if state.get("success") else 0,
            failure_count=0 if state.get("success") else 1,
        )

        return {
            "stored_skill": skill.to_dict(),
            "derived_skill_name": skill_name,
        }

    def _node_skip(self, state: SkillLearnerState) -> dict[str, Any]:
        logger.debug(f"Skipping skill storage: {state.get('error') or 'low confidence or no tools'}")
        return {"stored_skill": None}

    def _node_finalize(self, state: SkillLearnerState) -> dict[str, Any]:
        return {}

    def _route_after_validate(
        self, state: SkillLearnerState
    ) -> str:
        if state.get("should_store"):
            return "store"
        return "skip"

    def _derive_skill_name(self, state: SkillLearnerState) -> str:
        input_text = state.get("user_input", "").strip()
        words = input_text.split()
        if len(words) >= 2:
            base = "".join(w for w in words[:3] if w.isalnum())
        elif words:
            base = words[0]
        else:
            base = "skill"
        base = base[:32]
        existing_names = state.get("existing_skill_names", [])
        used_nums: set[int] = set()
        for name in existing_names:
            if name.startswith(base):
                suffix = name[len(base):]
                if suffix.isdigit():
                    used_nums.add(int(suffix))
        n = 1
        while n in used_nums:
            n += 1
        return f"{base}{n}"

    @staticmethod
    def _extract_args(call: dict[str, Any]) -> dict[str, Any]:
        args = call.get("args", call.get("arguments", {}))
        if isinstance(args, str):
            try:
                import json
                args = json.loads(args)
            except Exception:
                args = {"_raw": args}
        if not isinstance(args, dict):
            args = {"_raw": str(args)}
        return {k: str(v)[:100] for k, v in args.items()}