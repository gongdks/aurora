"""Memory manager — short-term conversation window + SQLite long-term storage.

ShortTermMemory: sliding window of recent exchanges (in-memory, per-session)
LongTermMemory: SQLite-backed facts, user preferences, conversation summaries
"""

from __future__ import annotations

import logging
from typing import Any

from agent.config import settings
from agent.memory.short_term import ShortTermMemory
from agent.memory.long_term import LongTermMemory

logger = logging.getLogger(__name__)


class MemoryManager:
    """Memory manager with short-term sliding window and long-term SQLite storage.

    Short-term: in-memory, recent N exchanges, lost on restart.
    Long-term: SQLite-backed, persists across sessions, searchable by keyword.
    """

    def __init__(self) -> None:
        self.short_term = ShortTermMemory()
        self._long_term: LongTermMemory | None = None

    @property
    def long_term(self) -> LongTermMemory:
        """Lazy-init long-term memory on first access."""
        if self._long_term is None:
            self._long_term = LongTermMemory()
        return self._long_term

    def format_short_term(self, chat_history: list | None = None) -> str:
        return self.short_term.format_for_prompt(chat_history)

    def format_short_term_messages(self, chat_history: list | None = None) -> list:
        return self.short_term.format_as_messages(chat_history)

    def add_interaction(self, user_msg: str, assistant_msg: str) -> None:
        """Record an interaction in both short-term and long-term memory."""
        self.short_term.add(user_msg, assistant_msg)
        # Auto-summarize to long-term memory
        try:
            self.long_term.summarize_conversation(
                user_msg[:500], assistant_msg[:500],
                summary=f"Q: {user_msg[:120]} → A: {assistant_msg[:120]}",
            )
        except Exception as exc:
            logger.warning("Failed to persist conversation summary: %s", exc)

    def remember_fact(self, key: str, content: str, category: str = "general") -> None:
        """Store a fact in long-term memory."""
        self.long_term.remember(key, content, category)

    def recall_fact(self, key: str) -> str | None:
        """Recall a fact by key from long-term memory."""
        return self.long_term.recall(key)

    def search_memories(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search long-term memory by keyword."""
        return self.long_term.search(query, limit=limit)

    def recent_conversations(self, limit: int = 5) -> list[dict[str, Any]]:
        """Get recent conversation summaries from long-term memory."""
        return self.long_term.recent_summaries(limit)

    def format_long_term_context(self, user_input: str, limit: int = 5) -> str:
        """Format relevant long-term memories as context for the current query."""
        memories = self.long_term.search(user_input, limit=limit)
        summaries = self.long_term.recent_summaries(limit=3)
        if not memories and not summaries:
            return ""

        lines: list[str] = ["## Long-term Context"]
        if memories:
            lines.append("\n### Relevant Memories")
            for m in memories:
                lines.append(f"- [{m['category']}] {m['key']}: {m['content'][:200]}")
        if summaries:
            lines.append("\n### Recent Conversations")
            for s in summaries:
                lines.append(f"- {s['summary'][:200]}")

        return "\n".join(lines) + "\n"

    def clear_short_term(self) -> None:
        self.short_term.clear()

    def clear_long_term(self) -> str:
        """Clear all long-term memories. Returns status message."""
        return self.long_term.clear_all()

    @property
    def memory_stats(self) -> dict[str, int]:
        """Return combined memory statistics."""
        return {
            "short_term_entries": len(self.short_term._history),
            **self.long_term.stats,
        }
