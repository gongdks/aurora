"""网页抓取工具。

给定 URL，返回页面正文文本（提取 <p>、<h1>-<h6>、<li> 等可读内容）。
"""

import re
from urllib.parse import urlparse

from langchain.tools import tool

from agent.tools.registry import register
from agent.utils.http_fetcher import http_fetch

# 简单的 HTML 标签剥离
_TAG_RE = re.compile(r"<[^>]+>")


def _extract_text(html: str) -> str:
    """简单地从 HTML 中提取可读文本。"""
    # 移除 script 和 style
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)

    # 将块级元素换行
    html = re.sub(r"</?(p|div|h[1-6]|li|tr|br|hr)[^>]*>", "\n", html, flags=re.IGNORECASE)

    # 剥离所有标签
    text = _TAG_RE.sub("", html)

    # 清理空白
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines[:200])  # 限制 200 行


@register(tags={"web"})
@tool
def web_fetcher(url: str) -> str:
    """抓取网页内容，提取正文文本。"""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return f"不支持的协议：{parsed.scheme}"

        html, status_code, content_type = http_fetch(
            url,
            extra_headers={
                "Accept": "text/html,application/xhtml+xml",
            },
        )

        if status_code != 200:
            return f"抓取失败，状态码：{status_code}"

        if "text/html" not in content_type and "text/plain" not in content_type:
            return f"非文本内容类型：{content_type}"

        text = _extract_text(html)
        if not text.strip():
            return "页面无可提取的文本内容（可能是纯 JavaScript 渲染页面）。"

        return f"网页内容 ({url})：\n\n{text[:3000]}"

    except (OSError, ValueError, RuntimeError) as exc:
        return f"抓取失败：{exc}"