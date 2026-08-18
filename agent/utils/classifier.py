"""Query classifier — determines if a query is simple or complex."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def classify_query(llm: Any, user_input: str) -> str:
    """Classify a query as 'simple' or 'complex'.

    Simple: questions that can be answered in one ReAct loop
        (single tool call or just LLM knowledge).
    Complex: multi-step tasks requiring planning, multiple tool
        calls, or verification.

    Args:
        llm: LangChain chat model instance.
        user_input: The user's query text.

    Returns:
        "simple" or "complex".
    """
    text = user_input.strip().lower()

    # Quick keyword heuristics for common patterns
    complex_keywords = [
        "plan", "step by step", "step-by-step", "分步", "步骤",
        "analyze", "分析", "compare", "对比", "evaluate",
        "research", "研究", "investigate", "查资料",
        "build", "create", "制作", "创建", "develop", "开发",
        "debug", "fix", "修复", "troubleshoot",
        "summarize", "总结", "compile", "汇总",
        "download", "下载", "scrape", "crawl", "爬虫",
        "organize", "整理", "structure", "结构化",
        "transform", "convert", "转换",
        "automate", "自动", "workflow", "流程",
        "deploy", "部署", "install", "安装",
        "monitor", "监控", "test", "测试",
        "write code", "写代码", "generate code",
        "api", "rest", "endpoint", "服务", "接口",
        "database", "sql", "query",
    ]

    simple_keywords = [
        "what is", "who is", "define", "meaning of",
        "how does", "explain", "translate", "翻译",
        "calculate", "compute", "count",
        "search for", "find", "look up", "查找", "搜索",
        "read file", "open file", "show", "display",
        "tell me", "give me", "list",
        "yes/no", "true/false",
    ]

    for kw in complex_keywords:
        if kw in text:
            return "complex"

    for kw in simple_keywords:
        if kw in text:
            return "simple"

    # Use LLM for ambiguous cases
    try:
        prompt = f"""Classify this user query as 'simple' or 'complex'.

Simple = can be answered with a single tool call or direct knowledge.
Complex = requires multiple steps, planning, or verification.

Query: "{user_input}"

Respond with exactly one word: simple or complex."""

        response = llm.invoke(prompt)
        answer = response.content.strip().lower()

        if "simple" in answer:
            return "simple"
        return "complex"

    except Exception:
        logger.warning("LLM classification failed, defaulting to 'complex'")
        return "complex"