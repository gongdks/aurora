"""Reflection engine — deep self-evaluation and strategy adjustment.

Implements the Reflexion pattern:
  Act → Observe → Reflect → Adjust → Re-act

Three-layer reflection:
  1. Result-level: Did the output meet quality thresholds?
  2. Process-level: Were the right tools/strategies used?
  3. Strategy-level: What should change for the next attempt?

Integrates with long-term memory to store failure patterns
and successful strategies for cross-session learning.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ReflectionScore:
    """Multi-dimensional evaluation scores."""

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
    """Complete reflection output with scores and strategy."""

    scores: ReflectionScore = field(default_factory=ReflectionScore)
    went_well: list[str] = field(default_factory=list)
    went_wrong: list[str] = field(default_factory=list)
    strategy_adjustments: list[str] = field(default_factory=list)
    should_replan: bool = False
    confidence: float = 0.0
    summary: str = ""


class ReflectionEngine:
    """Core reflexion engine — evaluates, reflects, and adjusts strategies.

    Usage:
        engine = ReflectionEngine()
        result = engine.reflect(
            user_input="分析销售数据",
            plan=["加载数据", "计算增长率", "生成图表"],
            results=["数据已加载", "增长率计算完成", "图表已生成"],
            execution_metrics={"steps": 3, "tool_calls": 5, "time": 45.2},
        )
        if result.should_replan:
            # Re-plan with strategy adjustments
            print(result.strategy_adjustments)
    """

    def __init__(self, llm: Any = None) -> None:
        self._llm = llm
        self._failure_patterns: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._success_strategies: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._max_patterns: int = 64
        self._pattern_cache_ttl: float = 3600.0

    def set_llm(self, llm: Any) -> None:
        self._llm = llm

    def reflect(
        self,
        user_input: str,
        plan: list[str],
        results: list[str],
        execution_metrics: dict[str, Any] | None = None,
        previous_feedback: str = "",
    ) -> ReflectionResult:
        """Perform deep reflection on execution results.

        Args:
            user_input: Original user goal
            plan: Steps that were executed
            results: Results from each step
            execution_metrics: {steps, tool_calls, time, tokens, ...}
            previous_feedback: Prior verification or reflection feedback

        Returns:
            ReflectionResult with scores, insights, and adjustment plan
        """
        execution_metrics = execution_metrics or {}

        scores = self._evaluate(user_input, plan, results, execution_metrics)

        went_well: list[str] = []
        went_wrong: list[str] = []
        strategy_adjustments: list[str] = []

        for i, (step, result) in enumerate(zip(plan, results)):
            if not result or result.strip() in ("", "未执行", "⏹ 已停止"):
                went_wrong.append(f"步骤 {i + 1} 未执行: {step[:80]}")
                strategy_adjustments.append(f"重新设计步骤 {i + 1} 的执行策略")
            elif len(result.strip()) < 4:
                went_wrong.append(f"步骤 {i + 1} 结果过短: {step[:60]}")
                strategy_adjustments.append(f"步骤 {i + 1} 需要更详细的执行")
            else:
                went_well.append(f"步骤 {i + 1} 成功: {step[:60]}")

        total_steps = execution_metrics.get("steps", len(plan))
        elapsed = execution_metrics.get("time", 0)
        tool_calls = execution_metrics.get("tool_calls", 0)

        if scores.goal_completeness == 0 or scores.overall < 0.35:
            strategy_adjustments.append("整体质量过低，建议完全重新规划")
        elif scores.overall < 0.5:
            strategy_adjustments.append("部分步骤需要改进，建议针对性补充执行")

        if elapsed > 0 and tool_calls > 0:
            efficiency = tool_calls / max(elapsed, 1)
            if efficiency > 3.0:
                went_wrong.append(f"工具调用效率偏低 ({tool_calls}次/{elapsed:.1f}s)")
                strategy_adjustments.append("减少不必要的工具调用，合并相关操作")

        if previous_feedback:
            fb_lower = previous_feedback.lower()
            if any(kw in fb_lower for kw in ("不完整", "缺少", "未完成", "incomplete", "missing", "not done")):
                strategy_adjustments.append("根据反馈补充缺失内容")

        if not went_well:
            went_well.append("执行流程完成")

        should_replan = scores.overall < 0.45 or any(
            w for w in went_wrong if "未执行" in w
        )

        confidence = min(1.0, scores.overall + 0.1 * len(went_well) - 0.05 * len(went_wrong))

        summary = self._build_summary(
            user_input, scores, went_well, went_wrong, strategy_adjustments
        )

        self._store_learning(user_input, scores, went_well, went_wrong, strategy_adjustments)

        return ReflectionResult(
            scores=scores,
            went_well=went_well,
            went_wrong=went_wrong,
            strategy_adjustments=strategy_adjustments,
            should_replan=should_replan,
            confidence=confidence,
            summary=summary,
        )

    def _evaluate(
        self,
        user_input: str,
        plan: list[str],
        results: list[str],
        metrics: dict[str, Any],
    ) -> ReflectionScore:
        """Multi-dimensional evaluation of execution quality."""
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

        return ReflectionScore(
            goal_completeness=goal_completeness,
            output_quality=output_quality,
            process_efficiency=process_efficiency,
            tool_selection=tool_selection,
        )

    def _build_summary(
        self,
        user_input: str,
        scores: ReflectionScore,
        went_well: list[str],
        went_wrong: list[str],
        adjustments: list[str],
    ) -> str:
        """Build a human-readable reflection summary."""
        parts = ["## 🤔 深度反思\n"]

        parts.append(f"**综合评分**: {scores.overall:.2f}")
        parts.append(f"- 目标完整性: {scores.goal_completeness:.2f}")
        parts.append(f"- 输出质量: {scores.output_quality:.2f}")
        parts.append(f"- 执行效率: {scores.process_efficiency:.2f}")
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

        return "\n".join(parts)

    def _store_learning(
        self,
        user_input: str,
        scores: ReflectionScore,
        went_well: list[str],
        went_wrong: list[str],
        adjustments: list[str],
    ) -> None:
        """Store reflection results for cross-session learning."""
        import hashlib

        key = hashlib.md5(user_input.encode("utf-8")).hexdigest()[:16]
        now = time.time()

        if (
            scores.overall < 0.5
            or scores.goal_completeness < 1.0
            or scores.output_quality < 0.2
        ) and went_wrong:
            self._failure_patterns[key] = {
                "user_input": user_input[:200],
                "scores": {
                    "goal": scores.goal_completeness,
                    "quality": scores.output_quality,
                    "efficiency": scores.process_efficiency,
                },
                "went_wrong": went_wrong[:5],
                "adjustments": adjustments[:5],
                "timestamp": now,
            }
            self._evict_patterns(self._failure_patterns)

        if scores.overall >= 0.7 and went_well:
            self._success_strategies[key] = {
                "user_input": user_input[:200],
                "scores": {
                    "goal": scores.goal_completeness,
                    "quality": scores.output_quality,
                    "efficiency": scores.process_efficiency,
                },
                "went_well": went_well[:5],
                "timestamp": now,
            }
            self._evict_patterns(self._success_strategies)

    def _evict_patterns(self, store: OrderedDict[str, dict[str, Any]]) -> None:
        """Evict expired or overflow entries."""
        now = time.time()
        expired = [k for k, v in store.items() if now - v.get("timestamp", 0) > self._pattern_cache_ttl]
        for k in expired:
            del store[k]
        while len(store) > self._max_patterns:
            store.popitem(last=False)

    def get_failure_patterns(self) -> list[dict[str, Any]]:
        """Retrieve stored failure patterns for similar queries."""
        return list(self._failure_patterns.values())

    def get_success_strategies(self) -> list[dict[str, Any]]:
        """Retrieve stored success strategies."""
        return list(self._success_strategies.values())

    @staticmethod
    def _extract_keywords(text: str) -> set[str]:
        """Extract keywords from text, handling both English words and CJK chars.

        For English: splits on whitespace.
        For CJK (Chinese/Japanese/Korean): uses character n-grams (bigrams).
        """
        text = text.lower().strip()
        words = set(text.split())
        if len(words) < 2:
            for i in range(len(text) - 1):
                bigram = text[i : i + 2]
                if any("\u4e00" <= c <= "\u9fff" for c in bigram):
                    words.add(bigram)
        return words

    def get_relevant_strategy(self, user_input: str) -> dict[str, Any] | None:
        """Find a relevant past strategy for similar user input."""
        import hashlib

        input_hash = hashlib.md5(user_input.encode("utf-8")).hexdigest()[:16]
        if input_hash in self._success_strategies:
            return self._success_strategies[input_hash]

        input_keywords = self._extract_keywords(user_input)
        for key, strategy in self._success_strategies.items():
            stored_input = strategy.get("user_input", "").lower()
            stored_keywords = self._extract_keywords(stored_input)
            overlap = len(input_keywords & stored_keywords)
            if overlap >= 2:
                return strategy
        return None

    def clear(self) -> None:
        """Clear all stored patterns and strategies."""
        self._failure_patterns.clear()
        self._success_strategies.clear()

    @property
    def stats(self) -> dict[str, int]:
        """Return reflection engine statistics."""
        return {
            "failure_patterns": len(self._failure_patterns),
            "success_strategies": len(self._success_strategies),
        }