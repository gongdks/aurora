"""Query classifier — hybrid keyword + LLM with confidence scoring and cache."""

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


class ClassifierCache:
    _instance: ClassifierCache | None = None

    def __new__(cls) -> ClassifierCache:
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
]


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
        conf = 0.5
        return ClassificationResult(
            label="complex",
            confidence=conf,
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


def classify_query(llm: Any, user_input: str) -> str:
    cache = ClassifierCache()
    key = _cache_key(user_input)

    cached = cache.get(key)
    if cached is not None:
        logger.debug("[Classifier] cache hit: %s (conf=%.2f)", cached.label, cached.confidence)
        return cached.label

    text = user_input.strip().lower()

    kw_result = _classify_via_keywords(text)
    if kw_result is not None and kw_result.confidence >= 0.5:
        cache.put(key, kw_result)
        logger.info(
            "[Classifier] keyword: %s conf=%.2f hits=%s",
            kw_result.label, kw_result.confidence, kw_result.detail,
        )
        return kw_result.label

    llm_result = _classify_via_llm(llm, user_input)
    cache.put(key, llm_result)
    return llm_result.label