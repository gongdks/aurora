"""持久笔记管理工具。

智能体可用它保存、读取、列出笔记，笔记以文本文件形式存于本地目录。
数据在多次会话间持久保留。
"""

import os

from langchain.tools import tool

from agent.config import settings
from agent.tools.registry import register


def _ensure_notes_dir() -> str:
    """确保笔记目录存在，返回绝对路径。"""
    path = os.path.abspath(settings.NOTES_DIR)
    os.makedirs(path, exist_ok=True)
    return path


def _safe_note_path(name: str) -> str:
    """生成安全的笔记文件路径，防止路径穿越。"""
    if "\x00" in name:
        raise ValueError(f"笔记名称包含非法字符：{name!r}")
    safe_name = "".join(c for c in name if c.isalnum() or c in "._- ")
    safe_name = safe_name.strip() or "untitled"
    if len(safe_name) > 80:
        safe_name = safe_name[:80]
    return os.path.join(_ensure_notes_dir(), f"{safe_name}.txt")


@register
@tool
def note_save(content: str) -> str:
    """保存一条笔记。

    Args:
        content: 笔记内容，格式 "标题: 正文内容"（标题和正文用冒号空格分隔）。
                 标题将作为文件名。
    """
    if len(content) > 50_000:
        return f"笔记内容过大（{len(content)} > 50000 字符），拒绝保存"

    try:
        if ": " in content:
            title, body = content.split(": ", 1)
        else:
            title = content[:30]
            body = content

        path = _safe_note_path(title.strip())
        with open(path, "w", encoding="utf-8") as f:
            f.write(body.strip())
        return f"笔记已保存：{os.path.basename(path)}"
    except OSError as exc:
        return f"保存笔记失败：{exc}"


@register
@tool
def note_read(title: str) -> str:
    """读取一条已保存的笔记。

    Args:
        title: 笔记标题（不含扩展名）
    """
    try:
        path = _safe_note_path(title.strip())
        if not os.path.isfile(path):
            return f"笔记不存在：{title}"
        with open(path, encoding="utf-8") as f:
            content = f.read()
        return f"笔记 [{title}]：\n\n{content}"
    except OSError as exc:
        return f"读取笔记失败：{exc}"


@register
@tool
def note_list() -> str:
    """列出所有已保存的笔记标题。"""
    try:
        notes_dir = _ensure_notes_dir()
        files = sorted(
            [f[:-4] for f in os.listdir(notes_dir) if f.endswith(".txt")],
            key=str.lower,
        )
        if not files:
            return "暂无笔记。"
        numbered = [f"{i}. {name}" for i, name in enumerate(files, 1)]
        return f"共 {len(files)} 条笔记：\n" + "\n".join(numbered)
    except OSError as exc:
        return f"列出笔记失败：{exc}"