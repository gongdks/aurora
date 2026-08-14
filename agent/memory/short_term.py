"""短期对话滑动窗口。

纯内存结构，保留最近 N 轮对话，同时支持文本和消息列表两种输出格式。
"""

from langchain_core.messages import AIMessage, HumanMessage

from agent.config import settings


class ShortTermMemory:
    """短期记忆：最近 N 轮对话的滑动窗口。

    不持久化，重启即丢失。
    提供 format_for_prompt()（文本，供 ReAct 模板）和
    format_as_messages()（消息列表，供 LangChain 消息链）。
    """

    def __init__(self, max_rounds: int | None = None) -> None:
        self.max_rounds = (
            max_rounds if max_rounds is not None else settings.MAX_SHORT_TERM_ROUNDS
        )
        self.max_messages = self.max_rounds * 2
        self._history: list[dict] = []

    def add(self, user_msg: str, assistant_msg: str) -> None:
        """添加一轮对话，超出容量时自动裁剪。"""
        self._history.append({"role": "user", "content": user_msg})
        self._history.append({"role": "assistant", "content": assistant_msg})
        if len(self._history) > self.max_messages:
            self._history = self._history[-self.max_messages:]

    def clear(self) -> None:
        """清空短期记忆。"""
        self._history.clear()

    def format_for_prompt(self, chat_history: list | None = None) -> str:
        """将聊天历史格式化为提示词可用的纯文本。

        Args:
            chat_history: Chat history 列表（可选）。
                          如果不传，使用内部 _history。

        Returns:
            格式化的历史文本，以「[用户]」「[智能体]」为前缀的行。
        """
        source = chat_history if chat_history else self._history
        if not source:
            return "（无近期对话）"

        recent = source[-self.max_rounds * 2:]
        parts: list[str] = []
        for item in recent:
            role = "[用户]" if item.get("role") == "user" else "[智能体]"
            content = item.get("content", "")
            content = self._normalize_content(content)
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def format_as_messages(
        self, chat_history: list | None = None,
    ) -> list:
        """将聊天历史转换为 LangChain 消息列表。

        保留 role 结构，用于需要消息格式的场景。

        Args:
            chat_history: Chat history 列表（可选）。

        Returns:
            HumanMessage / AIMessage 列表
        """
        source = chat_history if chat_history else self._history
        if not source:
            return []

        recent = source[-self.max_rounds * 2:]
        messages: list = []
        for item in recent:
            content = self._normalize_content(item.get("content", ""))
            if item.get("role") == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))
        return messages

    @staticmethod
    def _normalize_content(content) -> str:
        """规范化消息内容，兼容 str 和 list 格式。"""
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        elif isinstance(content, str):
            content = content.strip()
        else:
            content = str(content)
        return content