"""读取本地文件（沙盒路径）。

限制在配置的根目录内，防止路径穿越攻击。
"""

import os

from langchain.tools import tool

from agent.config import settings
from agent.tools.registry import register
from agent.utils.path_guard import safe_resolve


_EXTERNAL_KEYWORDS = {"桌面", "文档", "下载", "图片", "音乐", "视频", "收藏",
                      "desktop", "documents", "downloads", "pictures", "music", "videos", "favorites"}


def _detect_external_path(path: str) -> str | None:
    """检测是否为外部路径（如桌面/文档），如果是则返回建议信息。"""
    path_lower = path.lower().replace("\\", "/")
    for keyword in _EXTERNAL_KEYWORDS:
        if keyword in path_lower:
            return (
                f"路径 '{path}' 包含外部目录名 '{keyword}'。"
                f"file_reader 只能读取项目目录内的文件。"
                f"如果您想打开此文件，请使用 file_opener 工具。"
            )
    if ":" in path and not path.startswith(("http", "https")):
        drive = path[0].upper()
        if drive in "CDEFGHIJKLMNOPQRSTUVWXYZ":
            return (
                f"路径 '{path}' 看起来是绝对路径。"
                f"file_reader 只能读取项目目录内的文件。"
                f"如果您想打开此文件，请使用 file_opener 工具。"
            )
    return None


@register
@tool
def text_reader(filename: str) -> str:
    """读取项目目录内的纯文本文件内容，返回带行号的文本（仅限项目根目录内）。

    此工具是"读取文件内容"，NOT "打开文件"！
    - 当用户说"打开"文件 → 使用 file_opener（启动系统默认应用）
    - 当用户说"读取/查看内容"且文件在项目目录内 → 使用此工具
    - 需要带行号的文本阅读场景（如代码审查）

    限制：
    - 只能读取项目目录内的文件
    - 不能读取桌面、文档、下载等外部目录
    - 如果文件在外部目录，会自动提示使用 file_opener

    支持 .txt、.py、.md、.json、.csv 等文本文件。
    返回内容带行号，最多读取 500 行。

    Args:
        filename: 相对于项目根目录的路径，例如 "README.md"、"data/input.csv"
    """
    external_hint = _detect_external_path(filename)
    if external_hint:
        return external_hint

    try:
        safe = safe_resolve(filename)
        if not os.path.isfile(safe):
            return f"文件不存在：{filename}。如果此文件在桌面、文档等外部目录，请使用 file_opener 工具打开，而不是 file_reader。"
        if os.path.getsize(safe) > 1_000_000:
            return f"文件过大（>1MB），拒绝读取：{filename}"

        with open(safe, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()

        max_lines = 500
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            truncated = True
        else:
            truncated = False

        numbered = [f"{i + 1:>4}: {line.rstrip()}" for i, line in enumerate(lines)]
        result = "\n".join(numbered)
        if truncated:
            result += f"\n\n（文件共超过 {max_lines} 行，仅显示前 {max_lines} 行）"
        return result
    except ValueError as exc:
        return f"路径错误：{exc}"
    except OSError as exc:
        return f"读取失败：{exc}"