"""Query classifier — determines if a user request is 'simple' or 'complex'."""

from __future__ import annotations

import logging
from typing import Any

from agent.utils.retry import llm_invoke_with_guard

logger = logging.getLogger(__name__)

_QUERY_CLASSIFIER_PROMPT = """Classify this user request. Reply with exactly one word: "simple" or "complex".

Rules:
- "simple": a single-step task — math, factual lookup, translation, brief Q&A, read a file, simple web search.
- "complex": anything requiring multiple steps, planning, chaining tools (e.g. "search X then summarize Y"), analyzing and then acting, or involving conditional logic.

Request: {user_input}

Classification (simple/complex):"""

_CLASSIFIER_TIMEOUT = 5.0
_CLASSIFIER_MAX_RETRIES = 1


def classify_query(llm: Any, user_input: str) -> str:
    """Use the LLM to classify a query as 'simple' or 'complex'.

    Args:
        llm: LangChain BaseChatModel instance.
        user_input: The user's raw message.

    Returns:
        "simple" or "complex". Falls back to "complex" on any error
        (over-planning is safer than silently skipping steps).
    """
    prompt = _QUERY_CLASSIFIER_PROMPT.format(user_input=user_input)
    try:
        response = llm_invoke_with_guard(
            llm, [{"role": "user", "content": prompt}],
            timeout=_CLASSIFIER_TIMEOUT,
            max_retries=_CLASSIFIER_MAX_RETRIES,
        )
        result = response.content.strip().lower()
        first_word = result.split("\n")[0].strip().split()[0] if result else ""
        if first_word in ("simple", "complex"):
            return first_word
        if "simple" in result and "complex" not in result:
            return "simple"
        return "complex"
    except Exception:
        logger.warning("Query classifier failed, defaulting to complex path")
        return "complex"