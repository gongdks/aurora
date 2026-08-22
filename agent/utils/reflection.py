"""Reflection engine — deep self-evaluation and strategy adjustment.

Implements the Reflexion pattern using LangGraph StateGraph:
  Act → Observe → Reflect → Adjust → Re-act

Supports both standalone invocation and subgraph composition.
When used as a subgraph, shares the parent's checkpointer.

Key LangGraph features:
  - StateGraph with typed state for clean data flow
  - Conditional edges to route based on score thresholds
  - Shared checkpointer for unified state persistence
  - Subgraph composition support for embedding in parent graphs
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)


@dataclass
class ReflectionScore:
    goal_completeness: float = 0.0
    output_quality: float = 0.0
    process_efficiency: float = 0.0
    tool_selection: float = 0.0
    overall: float = 0.0

    def __post_init__(self) -> None:
        if self.overall == 0.0:
            scores = [
                self.goal_completeness,
                self.output_quality,
                self.process_efficiency,
                self.tool_selection,
            ]
            non_zero = [s for s in scores if s > 0]
            self.overall = sum(non_zero) / len(non_zero) if non_zero else 0.0


@dataclass
class ReflectionResult:
    scores: ReflectionScore = field(default_factory=ReflectionScore)
    went_well: list[str] = field(default_factory=list)
    went_wrong: list[str] = field(default_factory=list)
    strategy_adjustments: list[str] = field(default_factory=list)
    should_replan: bool = False
    confidence: float = 0.0
    summary: str = ""


class ReflectionState(TypedDict, total=False):
    user_input: str
    plan: list[str]
    results: list[str]
    execution_metrics: dict[str, Any]
    previous_feedback: str
    scores: dict[str, float]
    went_well: list[str]
    went_wrong: list[str]
    strategy_adjustments: list[str]
    should_replan: bool
    confidence: float
    summary: str
    learning_key: str
    failure_pattern_data: dict[str, Any] | None
    success_strategy_data: dict[str, Any] | None


def _extract_keywords(text: str) -> set[str]:
    text = text.lower().strip()
    words = set(text.split())
    if len(words) < 2:
        for i in range(len(text) - 1):
            bigram = text[i : i + 2]
            if any("\u4e00" <= c <= "\u9fff" for c in bigram):
                words.add(bigram)
    return words


class ReflectionEngine:
    """Core reflexion engine — LangGraph-native evaluation and strategy adjustment.

    Supports both standalone invocation and subgraph composition.
    When used as a subgraph, shares the parent's checkpointer and
    can be embedded directly via compiled_graph property.

    Usage (standalone):
        engine = ReflectionEngine()
        result = engine.reflect(user_input="...", plan=[...], results=[...])

    Usage (as subgraph):
        main_graph.add_node("reflection", engine.compiled_graph)
    """

    def __init__(self, llm: Any = None, checkpointer: Any | None = None) -> None:
        self._llm = llm
        self._failure_patterns: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._success_strategies: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._max_patterns: int = 64
        self._pattern_cache_ttl: float = 3600.0

        if checkpointer is None:
            checkpointer = MemorySaver()
        self._checkpointer = checkpointer
        self._graph = self._build_graph()

    @property
    def compiled_graph(self) -> Any:
        """Expose compiled graph for subgraph composition."""
        return self._graph

    def set_llm(self, llm: Any) -> None:
        self._llm = llm

    def _build_graph(self) -> Any:
        graph = StateGraph(ReflectionState)

        graph.add_node("evaluate_scores", self._node_evaluate_scores)
        graph.add_node("analyze_steps", self._node_analyze_steps)
        graph.add_node("build_insights", self._node_build_insights)
        graph.add_node("build_strategy", self._node_build_strategy)
        graph.add_node("store_learning", self._node_store_learning)
        graph.add_node("build_summary", self._node_build_summary)

        graph.add_edge(START, "evaluate_scores")
        graph.add_edge("evaluate_scores", "analyze_steps")
        graph.add_edge("analyze_steps", "build_insights")
        graph.add_edge("build_insights", "build_strategy")

        graph.add_conditional_edges(
            "build_strategy",
            self._route_after_strategy,
            {
                "store": "store_learning",
                "skip": "build_summary",
            },
        )

        graph.add_edge("store_learning", "build_summary")
        graph.add_edge("build_summary", END)

        return graph.compile(checkpointer=self._checkpointer)

    def reflect(
        self,
        user_input: str,
        plan: list[str],
        results: list[str],
        execution_metrics: dict[str, Any] | None = None,
        previous_feedback: str = "",
        config: dict[str, Any] | None = None,
    ) -> ReflectionResult:
        execution_metrics = execution_metrics or {}

        initial_state: ReflectionState = {
            "user_input": user_input,
            "plan": plan,
            "results": results,
            "execution_metrics": execution_metrics,
            "previous_feedback": previous_feedback,
            "scores": {},
            "went_well": [],
            "went_wrong": [],
            "strategy_adjustments": [],
            "should_replan": False,
            "confidence": 0.0,
            "summary": "",
            "learning_key": "",
        }

        run_config = config or {
            "configurable": {
                "thread_id": f"reflection_{hash(user_input) & 0xFFFFFFFF}",
            }
        }

        try:
            final = self._graph.invoke(initial_state, config=run_config)
        except Exception as exc:
            logger.error("[Reflection] Graph error: %s", exc, exc_info=True)
            return ReflectionResult(
                scores=ReflectionScore(overall=0.5),
                should_replan=True,
                summary=f"反思执行出错: {exc}",
            )

        failure_data = final.get("failure_pattern_data")
        success_data = final.get("success_strategy_data")
        if failure_data:
            self._failure_patterns[failure_data.get("learning_key", "")] = failure_data
            self._evict_patterns(self._failure_patterns)
        if success_data:
            self._success_strategies[success_data.get("learning_key", "")] = success_data
            self._evict_patterns(self._success_strategies)

        scores_dict = final.get("scores", {})
        scores = ReflectionScore(
            goal_completeness=scores_dict.get("goal_completeness", 0.0),
            output_quality=scores_dict.get("output_quality", 0.0),
            process_efficiency=scores_dict.get("process_efficiency", 0.0),
            tool_selection=scores_dict.get("tool_selection", 0.0),
            overall=scores_dict.get("overall", 0.0),
        )

        return ReflectionResult(
            scores=scores,
            went_well=final.get("went_well", []),
            went_wrong=final.get("went_wrong", []),
            strategy_adjustments=final.get("strategy_adjustments", []),
            should_replan=final.get("should_replan", False),
            confidence=final.get("confidence", 0.0),
            summary=final.get("summary", ""),
        )

    def invoke(
        self,
        user_input: str,
        plan: list[str],
        results: list[str],
        execution_metrics: dict[str, Any] | None = None,
        previous_feedback: str = "",
        config: dict[str, Any] | None = None,
    ) -> ReflectionResult:
        """Invoke reflection engine with optional parent config for subgraph integration.

        When config is provided from the parent graph, the checkpointer is shared,
        enabling unified state persistence across the graph hierarchy.
        """
        return self.reflect(
            user_input=user_input,
            plan=plan,
            results=results,
            execution_metrics=execution_metrics,
            previous_feedback=previous_feedback,
            config=config,
        )

    def _node_evaluate_scores(self, state: ReflectionState) -> dict[str, Any]:
        plan = state.get("plan", [])
        results = state.get("results", [])
        metrics = state.get("execution_metrics", {})

        completed_steps = sum(
            1 for r in results if r and r.strip() not in ("", "未执行", "⏹ 已停止")
        )
        total_steps = max(len(plan), 1)

        goal_completeness = completed_steps / total_steps if total_steps > 0 else 0.0

        result_lengths = [len(r.strip()) for r in results if r]
        avg_length = sum(result_lengths) / max(len(result_lengths), 1) if result_lengths else 0
        output_quality = min(1.0, avg_length / 30)

        elapsed = metrics.get("time", 0)
        tool_calls = metrics.get("tool_calls", 0)
        if elapsed > 0 and tool_calls > 0:
            efficiency = min(1.0, 30.0 / elapsed) if elapsed > 30 else 1.0
            process_efficiency = efficiency * (1.0 - min(1.0, tool_calls / 20))
        else:
            process_efficiency = 0.5

        tool_selection = 0.6
        if completed_steps == total_steps:
            tool_selection = 0.8

        scores = {
            "goal_completeness": goal_completeness,
            "output_quality": output_quality,
            "process_efficiency": process_efficiency,
            "tool_selection": tool_selection,
            "overall": 0.0,
        }

        non_zero = [v for v in [goal_completeness, output_quality, process_efficiency, tool_selection] if v > 0]
        scores["overall"] = sum(non_zero) / len(non_zero) if non_zero else 0.0

        return {"scores": scores}

    def _node_analyze_steps(self, state: ReflectionState) -> dict[str, Any]:
        plan = state.get("plan", [])
        results = state.get("results", [])

        went_well: list[str] = []
        went_wrong: list[str] = []

        for i, (step, result) in enumerate(zip(plan, results)):
            if not result or result.strip() in ("", "未执行", "⏹ 已停止"):
                went_wrong.append(f"步骤 {i + 1} 未执行: {step[:80]}")
            elif len(result.strip()) < 4:
                went_wrong.append(f"步骤 {i + 1} 结果过短: {step[:60]}")
            else:
                went_well.append(f"步骤 {i + 1} 成功: {step[:60]}")

        if not went_well:
            went_well.append("执行流程完成")

        return {
            "went_well": went_well,
            "went_wrong": went_wrong,
        }

    def _node_build_insights(self, state: ReflectionState) -> dict[str, Any]:
        scores = state.get("scores", {})
        went_well = state.get("went_well", [])
        went_wrong = state.get("went_wrong", [])
        strategy_adjustments: list[str] = []

        overall = scores.get("overall", 0.0)
        goal_completeness = scores.get("goal_completeness", 0.0)

        if goal_completeness == 0 or overall < 0.35:
            strategy_adjustments.append("整体质量过低，建议完全重新规划")
        elif overall < 0.5:
            strategy_adjustments.append("部分步骤需要改进，建议针对性补充执行")

        metrics = state.get("execution_metrics", {})
        elapsed = metrics.get("time", 0)
        tool_calls = metrics.get("tool_calls", 0)

        if elapsed > 0 and tool_calls > 0:
            efficiency = tool_calls / max(elapsed, 1)
            if efficiency > 3.0:
                went_wrong.append(f"工具调用效率偏低 ({tool_calls}次/{elapsed:.1f}s)")
                strategy_adjustments.append("减少不必要的工具调用，合并相关操作")

        previous_feedback = state.get("previous_feedback", "")
        if previous_feedback:
            fb_lower = previous_feedback.lower()
            if any(kw in fb_lower for kw in ("不完整", "缺少", "未完成", "incomplete", "missing", "not done")):
                strategy_adjustments.append("根据反馈补充缺失内容")

        should_replan = overall < 0.45 or any(
            w for w in went_wrong if "未执行" in w
        )

        confidence = min(1.0, overall + 0.1 * len(went_well) - 0.05 * len(went_wrong))

        learning_key = hashlib.md5(
            state.get("user_input", "").encode("utf-8")
        ).hexdigest()[:16]

        return {
            "strategy_adjustments": strategy_adjustments,
            "should_replan": should_replan,
            "confidence": confidence,
            "learning_key": learning_key,
        }

    def _node_build_strategy(self, state: ReflectionState) -> dict[str, Any]:
        return {}

    def _node_store_learning(self, state: ReflectionState) -> dict[str, Any]:
        scores = state.get("scores", {})
        went_well = state.get("went_well", [])
        went_wrong = state.get("went_wrong", [])
        adjustments = state.get("strategy_adjustments", [])
        user_input = state.get("user_input", "")
        learning_key = state.get("learning_key", "")
        now = time.time()

        overall = scores.get("overall", 0.0)
        goal_completeness = scores.get("goal_completeness", 0.0)
        output_quality = scores.get("output_quality", 0.0)

        failure_pattern = None
        success_strategy = None

        if (overall < 0.5 or goal_completeness < 1.0 or output_quality < 0.2) and went_wrong:
            failure_pattern = {
                "user_input": user_input[:200],
                "scores": {
                    "goal": goal_completeness,
                    "quality": output_quality,
                    "efficiency": scores.get("process_efficiency", 0.0),
                },
                "went_wrong": went_wrong[:5],
                "adjustments": adjustments[:5],
                "timestamp": now,
                "learning_key": learning_key,
            }

        if overall >= 0.7 and went_well:
            success_strategy = {
                "user_input": user_input[:200],
                "scores": {
                    "goal": goal_completeness,
                    "quality": output_quality,
                    "efficiency": scores.get("process_efficiency", 0.0),
                },
                "went_well": went_well[:5],
                "timestamp": now,
                "learning_key": learning_key,
            }

        return {
            "failure_pattern_data": failure_pattern,
            "success_strategy_data": success_strategy,
        }

    def _node_build_summary(self, state: ReflectionState) -> dict[str, Any]:
        scores = state.get("scores", {})
        went_well = state.get("went_well", [])
        went_wrong = state.get("went_wrong", [])
        adjustments = state.get("strategy_adjustments", [])

        parts = ["## 🤔 深度反思\n"]

        parts.append(f"**综合评分**: {scores.get('overall', 0.0):.2f}")
        parts.append(f"- 目标完整性: {scores.get('goal_completeness', 0.0):.2f}")
        parts.append(f"- 输出质量: {scores.get('output_quality', 0.0):.2f}")
        parts.append(f"- 执行效率: {scores.get('process_efficiency', 0.0):.2f}")
        parts.append("")

        if went_well:
            parts.append("**✅ 做得好的**:")
            for item in went_well[:5]:
                parts.append(f"  - {item}")
            parts.append("")

        if went_wrong:
            parts.append("**❌ 需要改进**:")
            for item in went_wrong[:5]:
                parts.append(f"  - {item}")
            parts.append("")

        if adjustments:
            parts.append("**🔧 策略调整**:")
            for item in adjustments[:5]:
                parts.append(f"  - {item}")
            parts.append("")

        return {"summary": "\n".join(parts)}

    @staticmethod
    def _route_after_strategy(state: ReflectionState) -> str:
        scores = state.get("scores", {})
        overall = scores.get("overall", 0.0)
        if overall > 0:
            return "store"
        return "skip"

    def _evict_patterns(self, store: OrderedDict[str, dict[str, Any]]) -> None:
        now = time.time()
        expired = [k for k, v in store.items() if now - v.get("timestamp", 0) > self._pattern_cache_ttl]
        for k in expired:
            del store[k]
        while len(store) > self._max_patterns:
            store.popitem(last=False)

    def get_failure_patterns(self) -> list[dict[str, Any]]:
        return list(self._failure_patterns.values())

    def get_success_strategies(self) -> list[dict[str, Any]]:
        return list(self._success_strategies.values())

    def get_relevant_strategy(self, user_input: str) -> dict[str, Any] | None:
        input_hash = hashlib.md5(user_input.encode("utf-8")).hexdigest()[:16]
        if input_hash in self._success_strategies:
            return self._success_strategies[input_hash]

        input_keywords = _extract_keywords(user_input)
        for key, strategy in self._success_strategies.items():
            stored_input = strategy.get("user_input", "").lower()
            stored_keywords = _extract_keywords(stored_input)
            overlap = len(input_keywords & stored_keywords)
            if overlap >= 2:
                return strategy
        return None

    def clear(self) -> None:
        self._failure_patterns.clear()
        self._success_strategies.clear()

    @property
    def stats(self) -> dict[str, int]:
        return {
            "failure_patterns": len(self._failure_patterns),
            "success_strategies": len(self._success_strategies),
        }