"""Markdown renderer — converts Markdown text to HTML using mistune."""

from __future__ import annotations

import html as _html
from typing import Any

try:
    import mistune

    _HAS_MISTUNE = True
except ImportError:
    _HAS_MISTUNE = False


def _make_markdown() -> Any:
    """Create a markdown renderer with common extensions."""
    if not _HAS_MISTUNE:
        return None

    return mistune.create_markdown(renderer="html")


_md = _make_markdown()


def render_markdown(text: str) -> str:
    """Render markdown text to HTML.

    Falls back to simple HTML escaping + newlines if mistune is unavailable.
    """
    if _md is None:
        escaped = _html.escape(text)
        return f'<div style="white-space: pre-wrap;">{escaped}</div>'

    rendered = _md(text)
    return rendered


def render_markdown_in_bubble(text: str, bubble_style: str) -> str:
    """Render markdown inside a styled bubble container.

    Args:
        text: Raw markdown text.
        bubble_style: Inline style string for the bubble outer div.
    """
    inner_html = render_markdown(text)
    return f"""
    <div style="{bubble_style}">
        <div class="md-body" style="font-size: 14px; line-height: 1.7;">
            {inner_html}
        </div>
    </div>"""