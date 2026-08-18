"""Chat HTML rendering — converts agent events and messages to HTML fragments."""

from __future__ import annotations

import html
from typing import Any

from agent.ui.markdown_renderer import render_markdown
from agent.ui.styles import COLORS


def _escape(text: str) -> str:
    """HTML-escape user-visible text."""
    return html.escape(text)


def thinking_html() -> str:
    """Render a thinking/typing indicator on the assistant side."""
    return f"""
    <div style="margin: 12px 0; text-align: left;" data-thinking="1">
        <div style="display: inline-block; background-color: {COLORS['assistant_bubble']};
             border: 1px solid {COLORS['border']}; border-radius: 12px;
             padding: 14px 18px; line-height: 1.7; max-width: 85%; text-align: left;">
            <div style="display: flex; align-items: center; gap: 8px; color: {COLORS['muted']}; font-size: 14px;">
                <span style="font-size: 16px;">🤖</span>
                <span>思考中</span>
                <span style="display: inline-flex; gap: 3px;">
                    <span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background-color: {COLORS['muted']}; opacity: 0.3;"></span>
                    <span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background-color: {COLORS['muted']}; opacity: 0.6;"></span>
                    <span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background-color: {COLORS['muted']}; opacity: 0.9;"></span>
                </span>
            </div>
        </div>
    </div>"""


def user_message_html(text: str) -> str:
    """Render a user message bubble (right side)."""
    escaped = _escape(text)
    return f"""
    <table align="right" cellspacing="0" cellpadding="0" style="margin: 8px 0;">
    <tr><td style="background-color: {COLORS['user_bubble']};
         border: 1px solid {COLORS['border']}; border-radius: 12px;
         padding: 10px 16px;">
        <div style="color: {COLORS['text']}; text-align: left;">{escaped}</div>
    </td></tr></table>"""


def assistant_header_html() -> str:
    """Render the assistant message header (left side)."""
    return f"""
    <div style="margin: 12px 0 4px 0; display: flex; align-items: center; gap: 8px; text-align: left;">
        <span style="font-size: 16px;">🤖</span>
        <span style="font-weight: 600; font-size: 13px; color: {COLORS['text_secondary']};">AI Agent</span>
    </div>"""


def tool_html(name: str, input_str: str, output: str | None = None) -> str:
    """Render a tool call block (compact, always-visible for QTextBrowser)."""
    escaped_name = _escape(name)
    escaped_input = _escape(input_str[:200])
    escaped_output = _escape((output or "")[:500])
    input_truncated = len(input_str) > 200
    output_truncated = (len(output or "") > 500) if output else False

    if output:
        input_line = escaped_input + ("..." if input_truncated else "")
        output_line = escaped_output + ("..." if output_truncated else "")
        return f"""
        <div class="tool-block" style="margin: 6px 0; background-color: {COLORS['tool_bubble']};
             border: 1px solid {COLORS['border']}; border-radius: 8px; padding: 10px 14px;">
            <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 6px;">
                <span style="color: {COLORS['accent']}; font-weight: 600; font-size: 12px;">🔧 {escaped_name}</span>
            </div>
            <div style="color: {COLORS['text_secondary']}; font-size: 12px; margin-bottom: 4px;">
                <span style="color: {COLORS['muted']};">Input:</span> {input_line}
            </div>
            <div style="color: {COLORS['green']}; font-size: 12px; font-family: Consolas, Monaco, monospace;">
                {output_line}
            </div>
        </div>"""
    else:
        short = escaped_input[:100] + ("..." if len(input_str) > 100 else "")
        return f"""
        <div class="tool-block" style="margin: 6px 0; background-color: {COLORS['tool_bubble']};
             border: 1px solid {COLORS['border']}; border-radius: 8px; padding: 10px 14px;">
            <div style="display: flex; align-items: center; gap: 6px;">
                <span style="color: {COLORS['accent']}; font-weight: 600; font-size: 12px;">🔧 {escaped_name}</span>
                <span style="color: {COLORS['muted']}; font-size: 11px;">{short}</span>
            </div>
        </div>"""


def error_html(text: str) -> str:
    """Render an error message block."""
    escaped = _escape(text[:500])
    return f"""
    <div style="margin: 8px 0; background-color: rgba(207, 34, 46, 0.08);
         border: 1px solid {COLORS['red']}; border-radius: 8px; padding: 10px 14px;">
        <div style="color: {COLORS['red']}; font-size: 13px; font-weight: 600;">Error</div>
        <div style="color: {COLORS['text']}; font-size: 13px; margin-top: 4px;">{escaped}</div>
    </div>"""


def cancelled_html() -> str:
    """Render a cancelled indicator."""
    return f"""
    <div style="margin: 8px 0; text-align: center;">
        <span style="color: {COLORS['yellow']}; font-size: 13px;">⏹ 已停止</span>
    </div>"""


def final_answer_html(answer: str, elapsed: float, tool_count: int) -> str:
    """Render the final answer with metadata (left side) — with Markdown support."""
    rendered = render_markdown(answer)
    sec = f"{elapsed:.1f}s" if elapsed < 60 else f"{int(elapsed // 60)}m{int(elapsed % 60)}s"
    return f"""
    <div style="margin: 12px 0; text-align: left;">
        <div style="display: inline-block; background-color: {COLORS['assistant_bubble']};
             border: 1px solid {COLORS['border']}; border-radius: 12px;
             padding: 14px 18px; line-height: 1.7; max-width: 85%; text-align: left;">
            <div class="md-body" style="color: {COLORS['text']}; font-size: 14px;">
                {rendered}
            </div>
        </div>
        <!-- <div style="margin-top: 6px; display: flex; gap: 16px; font-size: 11px; color: {COLORS['muted']};">
            <span>⏱ {sec}</span>
            <span>🔧 {tool_count} 次工具调用</span>
        </div> -->
    </div>"""


def status_for_log_message(message: str) -> str:
    """Determine icon color for a log message."""
    msg_lower = message.lower()
    if any(k in msg_lower for k in ("error", "❌", "failed")):
        return COLORS["red"]
    if any(k in msg_lower for k in ("complete", "✅", "done", "achieved")):
        return COLORS["green"]
    if any(k in msg_lower for k in ("plan", "📋")):
        return COLORS["accent"]
    return COLORS["text_secondary"]


def event_to_html(event: dict[str, Any]) -> str:
    """Convert a progress event to an HTML fragment."""
    event_type = event.get("type", "")

    if event_type == "tool":
        name = event.get("name", "unknown")
        input_str = event.get("input", "")
        output = event.get("output")
        if output:
            return tool_html(name, input_str, output)
        return tool_html(name, input_str)

    if event_type == "log":
        message = event.get("message", "")
        _HIDDEN_LOG_PREFIXES = (
            "🤔 **Analyzing",
            "⚡ **Route:",
            "⚡ **Running fast",
            "💭",
        )
        if any(message.startswith(p) for p in _HIDDEN_LOG_PREFIXES):
            return ""
        color = status_for_log_message(message)
        escaped = _escape(message)
        return f"""
        <div style="margin: 4px 0; font-size: 13px; color: {color}; line-height: 1.5;">
            {escaped}
        </div>"""

    if event_type == "error":
        message = event.get("message", "")
        return error_html(message)

    if event_type == "done":
        answer = event.get("answer", "")
        rendered = render_markdown(answer[:2000])
        return assistant_header_html() + f"""
        <div style="margin: 8px 0; text-align: left;">
            <div class="md-body" style="color: {COLORS['text']}; font-size: 14px;">
                {rendered}
            </div>
        </div>"""

    if event_type == "streaming_token":
        token = event.get("token", "")
        return f'<span class="streaming-token" style="color: {COLORS["text"]};">{_escape(token)}</span>'

    return ""