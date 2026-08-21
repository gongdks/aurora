"""Query classifier — LangGraph-native hybrid keyword + heuristic + adaptive feedback.

Three-tier adaptive design as a LangGraph StateGraph:

  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
  │ check_learned│───→│  check_cache │───→│  apply_heur  │───→│   END    │
  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘    └──────────┘
         │ miss               │ miss               │ low conf
         ▼                    ▼                    ▼
  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
  │ keyword_score│───→│  heuristic   │───→│  feedback    │───→│   END    │
  └──────────────┘    │  _fallback   │    │    _route    │    └──────────┘
                      └──────────────┘    └──────────────┘

Key LangGraph features leveraged:
  - StateGraph with typed state for clean data flow
  - Conditional edges for tier-based routing (learned → cache → heuristic → keyword → fallback)
  - Checkpointer for classification state persistence
  - Separation of each classification tier into independent nodes
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    label: str
    confidence: float
    source: str
    detail: str = ""


class ClassifierState(TypedDict, total=False):
    user_input: str
    normalized_text: str
    cache_key: str
    label: str
    confidence: float
    source: str
    detail: str
    iterations: int
    tool_calls: int
    elapsed: float
    steps_executed: int


# ---------------------------------------------------------------------------
# Learned classifications
# ---------------------------------------------------------------------------

class _LearnedClassifications:
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
    ("issue", 2), ("problem", 2),
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
# ClassifierGraph — LangGraph-native classification pipeline
# ---------------------------------------------------------------------------

class ClassifierGraph:
    """LangGraph-native query classifier with tiered adaptive routing.

    Usage:
        classifier = ClassifierGraph()
        label = classifier.classify("分析销售数据并生成图表")
        # → "complex"
    """

    def __init__(self, checkpointer: Any | None = None) -> None:
        if checkpointer is None:
            checkpointer = MemorySaver()
        self._checkpointer = checkpointer
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        graph = StateGraph(ClassifierState)

        graph.add_node("normalize", self._node_normalize)
        graph.add_node("check_learned", self._node_check_learned)
        graph.add_node("check_cache", self._node_check_cache)
        graph.add_node("apply_heuristics", self._node_apply_heuristics)
        graph.add_node("keyword_score", self._node_keyword_score)
        graph.add_node("heuristic_fallback", self._node_heuristic_fallback)

        graph.add_edge(START, "normalize")
        graph.add_edge("normalize", "check_learned")

        graph.add_conditional_edges(
            "check_learned",
            self._route_after_learned,
            {
                "hit": END,
                "miss": "check_cache",
            },
        )

        graph.add_conditional_edges(
            "check_cache",
            self._route_after_cache,
            {
                "hit": END,
                "miss": "apply_heuristics",
            },
        )

        graph.add_conditional_edges(
            "apply_heuristics",
            self._route_after_heuristics,
            {
                "hit": END,
                "miss": "keyword_score",
            },
        )

        graph.add_conditional_edges(
            "keyword_score",
            self._route_after_keyword,
            {
                "hit": END,
                "low_conf": "heuristic_fallback",
            },
        )

        graph.add_edge("heuristic_fallback", END)

        return graph.compile(checkpointer=self._checkpointer)

    def classify(self, user_input: str) -> str:
        initial: ClassifierState = {
            "user_input": user_input,
            "normalized_text": "",
            "cache_key": "",
            "label": "",
            "confidence": 0.0,
            "source": "",
            "detail": "",
        }

        config = {"configurable": {"thread_id": f"classify_{hash(user_input) & 0xFFFFFFFF}"}}

        try:
            final = self._graph.invoke(initial, config=config)
        except Exception as exc:
            logger.error("[Classifier] Graph error: %s", exc, exc_info=True)
            return "complex"

        label = final.get("label", "")
        if not label:
            return "complex"
        return label

    def record_feedback(
        self,
        user_input: str,
        original_label: str,
        iterations: int = 0,
        tool_calls: int = 0,
        elapsed: float = 0.0,
        steps_executed: int = 0,
    ) -> str | None:
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

    # ------------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------------

    @staticmethod
    def _node_normalize(state: ClassifierState) -> dict[str, Any]:
        user_input = state.get("user_input", "")
        normalized = " ".join(user_input.lower().split())
        cache_key = normalized
        return {
            "normalized_text": normalized,
            "cache_key": cache_key,
        }

    @staticmethod
    def _node_check_learned(state: ClassifierState) -> dict[str, Any]:
        key = state.get("cache_key", "")
        entry = _LEARNED.get(key)
        if entry is not None:
            return {
                "label": entry.label,
                "confidence": entry.confidence,
                "source": entry.source,
                "detail": entry.detail,
            }
        return {}

    @staticmethod
    def _node_check_cache(state: ClassifierState) -> dict[str, Any]:
        key = state.get("cache_key", "")
        entry = _CACHE.get(key)
        if entry is not None:
            logger.debug("[Classifier] cache hit: %s (conf=%.2f)", entry.label, entry.confidence)
            return {
                "label": entry.label,
                "confidence": entry.confidence,
                "source": entry.source,
                "detail": entry.detail,
            }
        return {}

    @staticmethod
    def _node_apply_heuristics(state: ClassifierState) -> dict[str, Any]:
        text = state.get("normalized_text", "")

        result = _length_heuristic(text)
        if result is not None:
            return {
                "label": result.label,
                "confidence": result.confidence,
                "source": result.source,
                "detail": result.detail,
            }

        result = _multi_signal_detection(text)
        if result is not None:
            return {
                "label": result.label,
                "confidence": result.confidence,
                "source": result.source,
                "detail": result.detail,
            }

        return {}

    @staticmethod
    def _node_keyword_score(state: ClassifierState) -> dict[str, Any]:
        text = state.get("normalized_text", "")
        result = _classify_via_keywords(text)
        if result is not None and result.confidence >= 0.5:
            return {
                "label": result.label,
                "confidence": result.confidence,
                "source": result.source,
                "detail": result.detail,
            }
        return {}

    @staticmethod
    def _node_heuristic_fallback(state: ClassifierState) -> dict[str, Any]:
        text = state.get("normalized_text", "")
        result = _heuristic_fallback(text)
        return {
            "label": result.label,
            "confidence": result.confidence,
            "source": result.source,
            "detail": result.detail,
        }

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    @staticmethod
    def _route_after_learned(state: ClassifierState) -> str:
        if state.get("label"):
            _CACHE.put(state["cache_key"], ClassificationResult(
                label=state["label"], confidence=state.get("confidence", 0.0),
                source=state.get("source", ""), detail=state.get("detail", ""),
            ))
            return "hit"
        return "miss"

    @staticmethod
    def _route_after_cache(state: ClassifierState) -> str:
        if state.get("label"):
            return "hit"
        return "miss"

    @staticmethod
    def _route_after_heuristics(state: ClassifierState) -> str:
        if state.get("label"):
            _CACHE.put(state["cache_key"], ClassificationResult(
                label=state["label"], confidence=state.get("confidence", 0.0),
                source=state.get("source", ""), detail=state.get("detail", ""),
            ))
            return "hit"
        return "miss"

    @staticmethod
    def _route_after_keyword(state: ClassifierState) -> str:
        if state.get("label") and state.get("confidence", 0.0) >= 0.5:
            _CACHE.put(state["cache_key"], ClassificationResult(
                label=state["label"], confidence=state.get("confidence", 0.0),
                source=state.get("source", ""), detail=state.get("detail", ""),
            ))
            return "hit"
        if state.get("label"):
            _CACHE.put(state["cache_key"], ClassificationResult(
                label=state["label"], confidence=state.get("confidence", 0.0),
                source=state.get("source", ""), detail=state.get("detail", ""),
            ))
            return "hit"
        return "low_conf"


# ---------------------------------------------------------------------------
# Module-level singleton and backward-compatible API
# ---------------------------------------------------------------------------

_classifier_graph: ClassifierGraph | None = None


def _get_classifier() -> ClassifierGraph:
    global _classifier_graph
    if _classifier_graph is None:
        _classifier_graph = ClassifierGraph()
    return _classifier_graph


def _cache_key(text: str) -> str:
    return " ".join(text.lower().split())


def _length_heuristic(text: str) -> ClassificationResult | None:
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


def _keyword_score(text: str, rules: list[tuple[str, int]]) -> tuple[int, list[str]]:
    score = 0
    matched: list[str] = []
    for kw, weight in rules:
        if kw in text:
            score += weight
            matched.append(kw)
    return score, matched


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


def _heuristic_fallback(text: str) -> ClassificationResult:
    action_verbs = [
        "分析", "对比", "研究", "查", "调查", "开发", "创建", "构建",
        "修复", "调试", "优化", "重构", "集成", "部署", "安装",
        "monitor", "test", "write", "build", "create", "debug",
        "fix", "install", "deploy", "integrate", "refactor",
        "research", "investigate", "analyze", "compare",
        "write code", "generate", "implement",
    ]
    verb_count = sum(1 for v in action_verbs if v in text)

    connectors = ["然后", "之后", "接着", "再", "并且", "同时", "先", "然后",
                   " and ", " then ", " after ", "next", "subsequently"]
    connector_count = sum(1 for c in connectors if c in text)

    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    total_chars = len(text.replace(" ", ""))

    if connector_count >= 2 or verb_count >= 2:
        return ClassificationResult(
            label="complex",
            confidence=0.7,
            source="heuristic_fallback",
            detail=f"connectors={connector_count} verbs={verb_count}",
        )

    if total_chars <= 10 and verb_count == 0 and connector_count == 0:
        return ClassificationResult(
            label="simple",
            confidence=0.65,
            source="heuristic_fallback",
            detail=f"short_query len={total_chars}",
        )

    if chinese_chars >= 15 and verb_count >= 1:
        return ClassificationResult(
            label="complex",
            confidence=0.6,
            source="heuristic_fallback",
            detail=f"long_query_with_verb chars={chinese_chars} verbs={verb_count}",
        )

    return ClassificationResult(
        label="complex",
        confidence=0.5,
        source="heuristic_fallback",
        detail="default_complex_safe",
    )


# ---------------------------------------------------------------------------
# Public API (delegates to ClassifierGraph)
# ---------------------------------------------------------------------------

def classify_query(user_input: str) -> str:
    classifier = _get_classifier()
    return classifier.classify(user_input)


def record_feedback(
    user_input: str,
    original_label: str,
    iterations: int = 0,
    tool_calls: int = 0,
    elapsed: float = 0.0,
    steps_executed: int = 0,
) -> str | None:
    classifier = _get_classifier()
    return classifier.record_feedback(
        user_input=user_input,
        original_label=original_label,
        iterations=iterations,
        tool_calls=tool_calls,
        elapsed=elapsed,
        steps_executed=steps_executed,
    )


def get_budget() -> _AdaptiveBudget:
    return _BUDGET


def set_budget(
    max_simple_iterations: int | None = None,
    max_simple_tool_calls: int | None = None,
    max_simple_time_sec: float | None = None,
    min_complex_steps: int | None = None,
) -> None:
    if max_simple_iterations is not None:
        _BUDGET.MAX_SIMPLE_ITERATIONS = max_simple_iterations
    if max_simple_tool_calls is not None:
        _BUDGET.MAX_SIMPLE_TOOL_CALLS = max_simple_tool_calls
    if max_simple_time_sec is not None:
        _BUDGET.MAX_SIMPLE_TIME_SEC = max_simple_time_sec
    if min_complex_steps is not None:
        _BUDGET.MIN_COMPLEX_STEPS = min_complex_steps


def clear_classifier() -> None:
    _CACHE.clear()
    _LEARNED.clear()
    logger.info("[Classifier] All caches cleared.")