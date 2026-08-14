"""Shared message extraction utilities for AutoGen GroupChat output."""

from __future__ import annotations


def extract_answer(messages: list) -> str:
    """Extract the final answer from GroupChat messages.

    Walks backward through messages, skipping system/internal messages
    and stripping TERMINATE markers.
    """
    for msg in reversed(messages):
        content = msg.get("content", "").strip()
        name = msg.get("name", "")
        if name in ("user_proxy", "chat_manager", "") or not content:
            continue
        if content.endswith("TERMINATE"):
            content = content[: -len("TERMINATE")].strip()
        if content:
            return content
    for msg in reversed(messages):
        content = msg.get("content", "").strip()
        if content:
            return content
    return "已完成，但未产生文本输出。"


def extract_plan(messages: list) -> str:
    """Extract the planner's plan from GroupChat messages."""
    for msg in messages:
        content = msg.get("content", "").strip()
        name = msg.get("name", "")
        if name == "planner" and content:
            return content
    return ""