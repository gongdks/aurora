"""Query classifier — hybrid keyword + LLM + adaptive feedback with confidence scoring and cache.

Three-tier adaptive design:
  Tier 1: Learned corrections (feedback from past runs)
  Tier 2: Keyword weight scoring with conflict resolution
  Tier 3: LLM-based classification (fallback)
  Runtime: Budget monitoring auto-upgrades/downgrades paths.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    label: str
    confidence: float
    source: str
    detail: str = ""


# ---------------------------------------------------------------------------
# Learned classifications — feedback from runtime execution
# ---------------------------------------------------------------------------

class _LearnedClassifications:
    """Stores learned corrections when runtime proves classification wrong.

    E.g. a query classified as "simple" that took >30s and used 5 tool calls
    gets reclassified as "complex" here, avoiding repeated misrouting.
    """

    def __init__(self) -> None:
        self._upgrades: OrderedDict[str, ClassificationResult] = OrderedDict()
        self._downgrades: OrderedDict[str, ClassificationResult] = OrderedDict()
        self._max_size: int = 128

    def get(self, key: str) -> ClassificationResult | None:
        entry = self._upgrades.get(key) or self._downgrades.get(key)
        if entry is not None:
            logger.debug("[Learned] hit: %s -> %s (from %s)", key[:40], entry.label, entry.source)
        return entry

    def record_upgrade(self, key: str, reason: str) -> None:
        result = ClassificationResult(
            label="complex",
            confidence=0.9,
            source="learned_upgrade",
            detail=reason,
        )
        self._upgrades[key] = result
        self._upgrades.move_to_end(key)
        while len(self._upgrades) > self._max_size:
            self._upgrades.popitem(last=False)
        self._downgrades.pop(key, None)
        logger.info("[Learned] upgraded '%s' -> complex (reason: %s)", key[:40], reason)

    def record_downgrade(self, key: str, reason: str) -> None:
        result = ClassificationResult(
            label="simple",
            confidence=0.9,
            source="learned_downgrade",
            detail=reason,
        )
        self._downgrades[key] = result
        self._downgrades.move_to_end(key)
        while len(self._downgrades) > self._max_size:
            self._downgrades.popitem(last=False)
        self._upgrades.pop(key, None)
        logger.info("[Learned] downgraded '%s' -> simple (reason: %s)", key[:40], reason)

    def clear(self) -> None:
        self._upgrades.clear()
        self._downgrades.clear()

    @property
    def size(self) -> int:
        return len(self._upgrades) + len(self._downgrades)


class _ClassifierCache:
    _instance: _ClassifierCache | None = None

    def __new__(cls) -> _ClassifierCache:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._cache: OrderedDict[str, ClassificationResult] = OrderedDict()
            cls._instance._max_size = 256
        return cls._instance

    def get(self, key: str) -> ClassificationResult | None:
        entry = self._cache.get(key)
        if entry is not None:
            self._cache.move_to_end(key)
        return entry

    def put(self, key: str, value: ClassificationResult) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()


class _AdaptiveBudget:
    """Runtime budget thresholds for auto-upgrade/downgrade decisions."""

    MAX_SIMPLE_ITERATIONS: int = 10
    MAX_SIMPLE_TOOL_CALLS: int = 4
    MAX_SIMPLE_TIME_SEC: float = 30.0
    MAX_COMPLEX_STEPS: int = 5
    MIN_COMPLEX_STEPS: int = 2

    @classmethod
    def exceeded_simple_budget(cls, iterations: int, tool_calls: int, elapsed: float) -> bool:
        return (
            iterations >= cls.MAX_SIMPLE_ITERATIONS
            or tool_calls >= cls.MAX_SIMPLE_TOOL_CALLS
            or elapsed >= cls.MAX_SIMPLE_TIME_SEC
        )

    @classmethod
    def can_downgrade_complex(cls, steps_executed: int, elapsed: float) -> bool:
        return steps_executed <= 1 and elapsed <= 10.0


_LEARNED = _LearnedClassifications()
_CACHE = _ClassifierCache()
_BUDGET = _AdaptiveBudget()


# ---------------------------------------------------------------------------
# Keyword rules
# ---------------------------------------------------------------------------

_COMPLEX_RULES: list[tuple[str, int]] = [
    ("plan", 3), ("step by step", 3), ("分步", 3), ("步骤", 3),
    ("analyze", 3), ("分析", 3), ("compare", 3), ("对比", 3),
    ("research", 3), ("研究", 3), ("investigate", 3), ("查资料", 3),
    ("build", 3), ("create", 3), ("制作", 3), ("创建", 3), ("开发", 3),
    ("debug", 3), ("fix", 3), ("修复", 3), ("troubleshoot", 3),
    ("summarize", 2), ("总结", 2), ("compile", 2), ("汇总", 2),
    ("download", 2), ("下载", 2), ("scrape", 3), ("crawl", 3), ("爬虫", 3),
    ("organize", 2), ("整理", 2), ("structure", 2), ("结构化", 2),
    ("transform", 2), ("convert", 2), ("转换", 2),
    ("automate", 3), ("自动", 3), ("workflow", 3), ("流程", 3),
    ("deploy", 3), ("部署", 3), ("install", 2), ("安装", 2),
    ("monitor", 3), ("监控", 3), ("test", 2), ("测试", 2),
    ("写代码", 3), ("generate code", 3),
    ("api", 3), ("rest", 3), ("endpoint", 3), ("服务", 2), ("接口", 2),
    ("database", 3), ("sql", 3),
    ("实现", 2), ("编写", 2),
    ("排查", 2), ("定位", 2),
    ("improve", 2), ("优化", 2), ("refactor", 2),
    ("integrate", 2), ("集成", 2), ("connect", 2), ("对接", 2),
    ("troubleshoot", 3), ("issue", 2), ("problem", 2),
    ("recommend", 2), ("suggest", 2), ("建议", 2), ("推荐", 2),
]

_SIMPLE_RULES: list[tuple[str, int]] = [
    ("what is", 2), ("who is", 2), ("define", 2), ("meaning of", 2),
    ("how does", 2), ("explain", 2), ("translate", 2), ("翻译", 2),
    ("calculate", 2), ("compute", 2), ("count", 1),
    ("search for", 1), ("find", 1), ("look up", 1), ("查找", 1), ("搜索", 1),
    ("read file", 1), ("open file", 1), ("show", 1), ("display", 1),
    ("tell me", 1), ("give me", 1), ("list", 1),
    ("yes or no", 2), ("true or false", 2),
    ("什么是", 2), ("什么", 1), ("怎么", 2), ("为什么", 2),
    ("解释", 2), ("说明", 2), ("介绍", 2), ("了解", 1),
    ("阅读", 1), ("看看", 1), ("读取", 1),
    ("你好", 1), ("hello", 1), ("hi", 1),
    ("多少", 1), ("几", 1),
    ("是谁", 2), ("哪里", 1), ("哪个", 1),
    ("介绍一下", 2), ("帮我看看", 1), ("帮我查", 1),
    ("简单", 1), ("直接", 1),
]


# ---------------------------------------------------------------------------
# Query length / structure heuristics
# ---------------------------------------------------------------------------

def _length_heuristic(text: str) -> ClassificationResult | None:
    """Short queries with no action verbs are almost always simple."""
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    total_chars = len(text.replace(" ", ""))

    if total_chars <= 6 and chinese_chars <= 4:
        return ClassificationResult(
            label="simple",
            confidence=0.8,
            source="heuristic_short",
            detail=f"len={total_chars}",
        )
    return None


def _multi_signal_detection(text: str) -> ClassificationResult | None:
    """Detect queries that imply multiple actions without explicit keywords.

    E.g. "help me search the web and then summarize" — no complex keyword,
    but "and then" signals multi-step intent.
    """
    multi_step_markers = [
        "然后", "之后", "接着", "再", "并且", "同时",
        " and ", " then ", " after that ", "next,",
        "first", "先", "subsequently",
    ]
    count = sum(1 for m in multi_step_markers if m in text)
    if count >= 2:
        return ClassificationResult(
            label="complex",
            confidence=0.7,
            source="heuristic_multi",
            detail=f"multi_step_markers={count}",
        )
    return None


# ---------------------------------------------------------------------------
# Core classification logic
# ---------------------------------------------------------------------------

def _keyword_score(text: str, rules: list[tuple[str, int]]) -> tuple[int, list[str]]:
    score = 0
    matched: list[str] = []
    for kw, weight in rules:
        if kw in text:
            score += weight
            matched.append(kw)
    return score, matched


def _cache_key(text: str) -> str:
    return " ".join(text.lower().split())


def _classify_via_keywords(text: str) -> ClassificationResult | None:
    complex_score, complex_hits = _keyword_score(text, _COMPLEX_RULES)
    simple_score, simple_hits = _keyword_score(text, _SIMPLE_RULES)

    if complex_score > 0 and simple_score == 0:
        conf = min(1.0, complex_score / 6.0)
        return ClassificationResult(
            label="complex",
            confidence=conf,
            source="keyword",
            detail=f"complex[{complex_score}] hits={complex_hits}",
        )

    if simple_score > 0 and complex_score == 0:
        conf = min(1.0, simple_score / 4.0)
        return ClassificationResult(
            label="simple",
            confidence=conf,
            source="keyword",
            detail=f"simple[{simple_score}] hits={simple_hits}",
        )

    if complex_score > simple_score:
        conf = min(0.9, complex_score / (complex_score + simple_score + 1.0))
        return ClassificationResult(
            label="complex",
            confidence=conf,
            source="keyword",
            detail=f"complex[{complex_score}] vs simple[{simple_score}], hits={complex_hits}",
        )

    if simple_score > complex_score:
        conf = min(0.85, simple_score / (simple_score + complex_score + 1.0))
        return ClassificationResult(
            label="simple",
            confidence=conf,
            source="keyword",
            detail=f"simple[{simple_score}] vs complex[{complex_score}], hits={simple_hits}",
        )

    if complex_score > 0 and simple_score > 0:
        return ClassificationResult(
            label="complex",
            confidence=0.5,
            source="keyword_tie",
            detail=f"tie complex[{complex_score}] vs simple[{simple_score}]",
        )

    return None


def _classify_via_llm(llm: Any, user_input: str) -> ClassificationResult:
    t0 = time.perf_counter()
    try:
        prompt = f"""You are a routing classifier. Decide if this user query is "simple" or "complex".

Simple = answerable in one step: a single tool call, a direct lookup, or knowledge-based response.
Complex = needs multiple steps, planning, multiple tool calls, or verification.

Query: {user_input}

Reply with exactly one word: simple or complex."""

        response = llm.invoke(prompt)
        answer = response.content.strip().lower()
        elapsed = time.perf_counter() - t0

        if "simple" in answer:
            label = "simple"
        elif "complex" in answer:
            label = "complex"
        else:
            label = "complex"

        logger.info("[Classifier] LLM decision: %s (%.1fs, query=%s)", label, elapsed, user_input[:60])
        return ClassificationResult(
            label=label,
            confidence=0.7,
            source="llm",
            detail=f"llm_elapsed={elapsed:.2f}s",
        )
    except Exception as exc:
        logger.warning("[Classifier] LLM failed, defaulting to complex: %s", exc)
        return ClassificationResult(
            label="complex",
            confidence=0.3,
            source="fallback",
            detail=f"llm_error={exc}",
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_query(llm: Any, user_input: str) -> str:
    """Classify query complexity with three-tier adaptive strategy.

    Tier 1: Learned corrections (feedback from past runs)
    Tier 2: Keyword + heuristic scoring
    Tier 3: LLM classification (fallback)
    """
    key = _cache_key(user_input)

    learned = _LEARNED.get(key)
    if learned is not None:
        logger.info("[Classifier] learned: %s (src=%s)", learned.label, learned.source)
        return learned.label

    cached = _CACHE.get(key)
    if cached is not None:
        logger.debug("[Classifier] cache hit: %s (conf=%.2f)", cached.label, cached.confidence)
        return cached.label

    text = user_input.strip().lower()

    heuristic_result = _length_heuristic(text)
    if heuristic_result is not None:
        _CACHE.put(key, heuristic_result)
        logger.info(
            "[Classifier] heuristic: %s conf=%.2f (%s)",
            heuristic_result.label, heuristic_result.confidence, heuristic_result.detail,
        )
        return heuristic_result.label

    multi_result = _multi_signal_detection(text)
    if multi_result is not None:
        _CACHE.put(key, multi_result)
        logger.info(
            "[Classifier] multi-signal: %s conf=%.2f",
            multi_result.label, multi_result.confidence,
        )
        return multi_result.label

    kw_result = _classify_via_keywords(text)
    if kw_result is not None and kw_result.confidence >= 0.5:
        _CACHE.put(key, kw_result)
        logger.info(
            "[Classifier] keyword: %s conf=%.2f hits=%s",
            kw_result.label, kw_result.confidence, kw_result.detail,
        )
        return kw_result.label

    llm_result = _classify_via_llm(llm, user_input)
    _CACHE.put(key, llm_result)
    return llm_result.label


def record_feedback(
    user_input: str,
    original_label: str,
    iterations: int = 0,
    tool_calls: int = 0,
    elapsed: float = 0.0,
    steps_executed: int = 0,
) -> str | None:
    """Report runtime execution metrics to learn better classifications.

    Returns the new label if reclassified, or None if no change needed.
    """
    key = _cache_key(user_input)
    changed = False
    new_label: str | None = None

    if original_label == "simple" and _BUDGET.exceeded_simple_budget(iterations, tool_calls, elapsed):
        reason = f"budget_exceeded: iter={iterations} tools={tool_calls} time={elapsed:.1f}s"
        _LEARNED.record_upgrade(key, reason)
        _CACHE.put(key, ClassificationResult(
            label="complex", confidence=0.9, source="learned_upgrade", detail=reason,
        ))
        new_label = "complex"
        changed = True

    elif original_label == "complex" and _BUDGET.can_downgrade_complex(steps_executed, elapsed):
        reason = f"under_budget: steps={steps_executed} time={elapsed:.1f}s"
        _LEARNED.record_downgrade(key, reason)
        _CACHE.put(key, ClassificationResult(
            label="simple", confidence=0.9, source="learned_downgrade", detail=reason,
        ))
        new_label = "simple"
        changed = True

    if changed:
        logger.info(
            "[Classifier] feedback: %s -> %s (orig=%s, iter=%d, tools=%d, time=%.1fs, steps=%d)",
            user_input[:40], new_label, original_label, iterations, tool_calls, elapsed, steps_executed,
        )
        return new_label

    return None


def get_budget() -> _AdaptiveBudget:
    """Get current adaptive budget thresholds for tuning."""
    return _BUDGET


def set_budget(
    max_simple_iterations: int | None = None,
    max_simple_tool_calls: int | None = None,
    max_simple_time_sec: float | None = None,
    min_complex_steps: int | None = None,
) -> None:
    """Override adaptive budget thresholds at runtime."""
    if max_simple_iterations is not None:
        _BUDGET.MAX_SIMPLE_ITERATIONS = max_simple_iterations
    if max_simple_tool_calls is not None:
        _BUDGET.MAX_SIMPLE_TOOL_CALLS = max_simple_tool_calls
    if max_simple_time_sec is not None:
        _BUDGET.MAX_SIMPLE_TIME_SEC = max_simple_time_sec
    if min_complex_steps is not None:
        _BUDGET.MIN_COMPLEX_STEPS = min_complex_steps


def clear_classifier() -> None:
    """Clear all caches and learned classifications."""
    _CACHE.clear()
    _LEARNED.clear()
    logger.info("[Classifier] All caches cleared.")