"""Web chat HTML template — full-page HTML for QWebEngineView."""

from __future__ import annotations

import html
from typing import Any

from agent.ui.markdown_renderer import render_markdown
from agent.ui.styles import COLORS

WEB_CSS = """
* { box-sizing: border-box; }
body {
    margin: 0;
    padding: 16px;
    background-color: #f5f6f8;
    color: #1f2328;
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    font-size: 14px;
    line-height: 1.7;
}
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 4px;
    min-height: 100%;
}
.msg-user {
    display: flex;
    justify-content: flex-end;
    margin: 8px 0;
}
.msg-user .bubble {
    background-color: #dce8ff;
    border: 1px solid #d9dce1;
    border-radius: 12px;
    padding: 10px 16px;
    max-width: 80%;
    color: #1f2328;
    word-wrap: break-word;
    white-space: pre-wrap;
}
.msg-assistant {
    display: flex;
    justify-content: flex-start;
    margin: 8px 0;
}
.msg-assistant .bubble {
    background-color: #f0f2f5;
    border: 1px solid #d9dce1;
    border-radius: 12px;
    padding: 14px 18px;
    max-width: 85%;
    color: #1f2328;
}
.msg-assistant .avatar {
    font-size: 16px;
    margin-right: 6px;
}
.msg-tool {
    margin: 6px 0;
    background-color: #fff8e1;
    border: 1px solid #d9dce1;
    border-radius: 8px;
    padding: 10px 14px;
}
.msg-tool .tool-name {
    color: #4a6cf7;
    font-weight: 600;
    font-size: 12px;
    margin-bottom: 4px;
}
.msg-tool .tool-input {
    color: #656d76;
    font-size: 12px;
    margin-bottom: 4px;
    word-break: break-all;
}
.msg-tool .tool-output {
    color: #28a745;
    font-size: 12px;
    font-family: Consolas, Monaco, monospace;
    white-space: pre-wrap;
    word-break: break-all;
}
.msg-thinking {
    display: flex;
    justify-content: flex-start;
    margin: 12px 0;
}
.msg-thinking .bubble {
    background-color: #f0f2f5;
    border: 1px solid #d9dce1;
    border-radius: 12px;
    padding: 14px 18px;
    color: #8b949e;
    font-size: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.msg-thinking .dots {
    display: inline-flex;
    gap: 3px;
}
.msg-thinking .dots span {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background-color: #8b949e;
}
.msg-error {
    margin: 8px 0;
    background-color: rgba(207, 34, 46, 0.08);
    border: 1px solid #cf222e;
    border-radius: 8px;
    padding: 10px 14px;
    color: #cf222e;
    font-size: 13px;
}
.msg-cancelled {
    margin: 8px 0;
    text-align: center;
    color: #d29922;
    font-size: 13px;
}
.msg-log {
    margin: 4px 0;
    font-size: 13px;
    line-height: 1.5;
}
.msg-done-preview {
    margin: 8px 0;
}

.md-body {
    color: #1f2328;
    line-height: 1.7;
}
.md-body h1 {
    font-size: 20px;
    font-weight: 700;
    margin: 16px 0 10px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #d9dce1;
}
.md-body h2 {
    font-size: 18px;
    font-weight: 700;
    margin: 14px 0 8px 0;
    padding-bottom: 4px;
    border-bottom: 1px solid #d9dce1;
}
.md-body h3 {
    font-size: 16px;
    font-weight: 700;
    margin: 12px 0 6px 0;
}
.md-body h4, .md-body h5, .md-body h6 {
    font-size: 14px;
    font-weight: 600;
    margin: 10px 0 4px 0;
    color: #656d76;
}
.md-body p {
    margin: 8px 0;
}
.md-body ul, .md-body ol {
    margin: 8px 0;
    padding-left: 24px;
}
.md-body li {
    margin: 4px 0;
}
.md-body blockquote {
    margin: 10px 0;
    padding: 8px 14px;
    border-left: 3px solid #4a6cf7;
    background-color: #f0f2f5;
    color: #656d76;
    border-radius: 0 6px 6px 0;
}
.md-body code {
    background-color: #f6f8fa;
    color: #24292f;
    border-radius: 4px;
    padding: 2px 6px;
    font-family: Consolas, Monaco, monospace;
    font-size: 13px;
}
.md-body pre {
    background-color: #f6f8fa;
    border: 1px solid #d9dce1;
    border-radius: 8px;
    padding: 12px;
    margin: 10px 0;
    font-family: Consolas, Monaco, monospace;
    font-size: 13px;
    white-space: pre-wrap;
    word-wrap: break-word;
    overflow-x: auto;
}
.md-body pre code {
    background-color: transparent;
    padding: 0;
    border-radius: 0;
    border: none;
}
.md-body table {
    border-collapse: collapse;
    margin: 12px 0;
    width: 100%;
    font-size: 13px;
}
.md-body th {
    background-color: #f0f2f5;
    border: 1px solid #d9dce1;
    padding: 8px 12px;
    font-weight: 600;
    color: #1f2328;
    text-align: left;
}
.md-body td {
    border: 1px solid #d9dce1;
    padding: 6px 12px;
}
.md-body tr:nth-child(even) {
    background-color: #f9fafb;
}
.md-body hr {
    border: none;
    border-top: 1px solid #d9dce1;
    margin: 16px 0;
}
.md-body strong {
    font-weight: 700;
}
.md-body em {
    font-style: italic;
}
.md-body del {
    color: #8b949e;
}
.md-body a {
    color: #4a6cf7;
    text-decoration: underline;
}
.msg-streaming {
    display: flex;
    justify-content: flex-start;
    margin: 12px 0;
}
.msg-streaming .bubble {
    background-color: #f0f2f5;
    border: 1px solid #d9dce1;
    border-radius: 12px;
    padding: 14px 18px;
    max-width: 85%;
    color: #1f2328;
    font-size: 14px;
    line-height: 1.7;
    white-space: pre-wrap;
    word-wrap: break-word;
}
.msg-streaming .cursor {
    display: inline-block;
    width: 2px;
    height: 1em;
    background-color: #4a6cf7;
    margin-left: 2px;
    animation: blink 0.8s infinite;
    vertical-align: text-bottom;
}
@keyframes blink {
    0%, 50% { opacity: 1; }
    51%, 100% { opacity: 0; }
}
"""


def _wrap_page(body_html: str) -> str:
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>{WEB_CSS}</style>
</head>
<body>
<div class="chat-container" id="chat">{body_html}</div>
<script>
function scrollToBottom() {{
    var chat = document.getElementById('chat');
    if (chat) {{
        window.scrollTo(0, document.body.scrollHeight);
    }}
}}
</script>
</body>
</html>"""


def build_user_msg(text: str, msg_id: str = "") -> str:
    escaped = html.escape(text)
    id_attr = f' id="{msg_id}"' if msg_id else ""
    return f'<div class="msg-user"{id_attr}><div class="bubble">{escaped}</div></div>'


def build_assistant_msg(rendered_md: str, msg_id: str = "") -> str:
    id_attr = f' id="{msg_id}"' if msg_id else ""
    return (
        f'<div class="msg-assistant"{id_attr}>'
        f'<div class="bubble">'
        f'<div class="md-body">{rendered_md}</div>'
        f'</div></div>'
    )


def build_tool_msg(name: str, input_str: str, output: str | None = None, msg_id: str = "") -> str:
    escaped_name = html.escape(name)
    escaped_input = html.escape(input_str[:200])
    escaped_output = html.escape((output or "")[:500])
    input_truncated = len(input_str) > 200
    output_truncated = (len(output or "") > 500) if output else False
    id_attr = f' id="{msg_id}"' if msg_id else ""

    if output:
        return (
            f'<div class="msg-tool"{id_attr}>'
            f'<div class="tool-name">🔧 {escaped_name}</div>'
            f'<div class="tool-input"><span style="color:#8b949e">Input:</span> '
            f'{escaped_input}{"..." if input_truncated else ""}</div>'
            f'<div class="tool-output">'
            f'{escaped_output}{"..." if output_truncated else ""}</div>'
            f'</div>'
        )
    else:
        short = escaped_input[:100] + ("..." if len(input_str) > 100 else "")
        return (
            f'<div class="msg-tool"{id_attr}>'
            f'<div class="tool-name">🔧 {escaped_name}</div>'
            f'<div class="tool-input" style="color:#8b949e;font-size:11px;">{short}</div>'
            f'</div>'
        )


def build_thinking(msg_id: str = "") -> str:
    id_attr = f' id="{msg_id}"' if msg_id else ""
    return (
        f'<div class="msg-thinking"{id_attr}>'
        f'<div class="bubble">'
        f'<span class="avatar">🤖</span>'
        f'<span>思考中</span>'
        f'<span class="dots"><span style="opacity:0.3"></span><span style="opacity:0.6"></span><span style="opacity:0.9"></span></span>'
        f'</div></div>'
    )


def build_streaming(text: str, msg_id: str = "") -> str:
    id_attr = f' id="{msg_id}"' if msg_id else ""
    escaped = html.escape(text)
    return (
        f'<div class="msg-streaming"{id_attr}>'
        f'<div class="bubble">{escaped}<span class="cursor"></span></div>'
        f'</div>'
    )


def build_error(text: str, msg_id: str = "") -> str:
    escaped = html.escape(text[:500])
    id_attr = f' id="{msg_id}"' if msg_id else ""
    return f'<div class="msg-error"{id_attr}>{escaped}</div>'


def build_cancelled(msg_id: str = "") -> str:
    id_attr = f' id="{msg_id}"' if msg_id else ""
    return f'<div class="msg-cancelled"{id_attr}>⏹ 已停止</div>'


def build_log(text: str, color: str = "#656d76", msg_id: str = "") -> str:
    escaped = html.escape(text)
    id_attr = f' id="{msg_id}"' if msg_id else ""
    return f'<div class="msg-log"{id_attr} style="color:{color}">{escaped}</div>'


def build_event_html(event: dict[str, Any]) -> str:
    event_type = event.get("type", "")
    if event_type == "tool":
        name = event.get("name", "unknown")
        input_str = event.get("input", "")
        output = event.get("output")
        if output:
            return build_tool_msg(name, input_str, output)
        return build_tool_msg(name, input_str)

    if event_type == "log":
        message = event.get("message", "")
        hidden_prefixes = ("🤔 **Analyzing", "⚡ **Route:", "⚡ **Running fast", "💭")
        if any(message.startswith(p) for p in hidden_prefixes):
            return ""
        return build_log(message, "#656d76")

    if event_type == "error":
        return build_error(event.get("message", ""))

    if event_type == "done":
        answer = event.get("answer", "")
        rendered = render_markdown(answer[:2000])
        return build_assistant_msg(rendered)

    if event_type == "streaming_token":
        token = event.get("token", "")
        return html.escape(token)

    return ""


def wrap_page(body_html: str) -> str:
    return _wrap_page(body_html)