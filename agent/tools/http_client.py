"""HTTP 请求工具 — 支持 GET/POST/PUT/DELETE 等方法。

用于调用 API、测试接口、获取网页数据等。
支持自定义 Headers、请求体、超时控制。
"""

from __future__ import annotations

import json
import re
from urllib.parse import urlparse

from langchain.tools import tool

from agent.tools.registry import register

_MAX_RESPONSE_SIZE = 50_000
_DEFAULT_TIMEOUT = 15
_MAX_URL_LEN = 500


def _validate_url(url: str) -> str | None:
    url = url.strip()
    if len(url) > _MAX_URL_LEN:
        return f"URL 过长（{len(url)} > {_MAX_URL_LEN}）"
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"仅支持 http/https 协议，当前：{parsed.scheme or '无协议'}"
    if not parsed.netloc:
        return "URL 缺少主机名"
    return None


def _parse_headers(headers_str: str) -> dict[str, str]:
    """解析 Headers 字符串，支持 JSON 格式或 Key: Value 格式。"""
    headers: dict[str, str] = {}
    if not headers_str.strip():
        return headers

    stripped = headers_str.strip()
    if stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return {str(k): str(v) for k, v in parsed.items()}
        except json.JSONDecodeError:
            pass

    for line in stripped.split("\n"):
        line = line.strip()
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if key:
                headers[key] = value

    return headers


def _parse_body(body: str, content_type: str) -> bytes | str | None:
    """解析请求体。"""
    if not body or not body.strip():
        return None

    if "json" in content_type.lower():
        try:
            json.loads(body)
            return body.encode("utf-8")
        except json.JSONDecodeError:
            return body.encode("utf-8")

    return body.encode("utf-8")


def _truncate_response(text: str, max_size: int = _MAX_RESPONSE_SIZE) -> str:
    if len(text) <= max_size:
        return text
    return text[:max_size] + f"\n\n（响应过长，已截断 {len(text)} → {max_size} 字符）"


def _try_format_json(text: str) -> str:
    """尝试格式化 JSON 响应。"""
    text = text.strip()
    if not text:
        return text
    try:
        obj = json.loads(text)
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except (json.JSONDecodeError, ValueError):
        return text


@register
@tool
def http_request(
    url: str,
    method: str = "GET",
    headers: str = "",
    body: str = "",
    content_type: str = "application/json",
    timeout: int = 15,
) -> str:
    """发送 HTTP 请求，返回响应状态码、Headers 和 Body。

    支持 GET、POST、PUT、PATCH、DELETE、HEAD、OPTIONS 方法。
    可用于调用 REST API、测试后端接口、获取网页数据等。

    Args:
        url: 请求 URL，如 "https://api.example.com/users"
        method: HTTP 方法，可选：GET、POST、PUT、PATCH、DELETE、HEAD、OPTIONS（默认 GET）
        headers: 请求 Headers，支持两种格式：
                 - JSON 格式：'{"Authorization": "Bearer token"}'
                 - 每行一个：'Authorization: Bearer token\\nContent-Type: application/json'
        body: 请求体内容（POST/PUT/PATCH 时使用）
        content_type: 请求体 Content-Type，默认 "application/json"
        timeout: 超时时间（秒），默认 15 秒
    """
    err = _validate_url(url)
    if err:
        return f"[错误] {err}"

    method = method.strip().upper()
    valid_methods = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
    if method not in valid_methods:
        return f"[错误] 无效的 HTTP 方法：{method}，可选：{', '.join(sorted(valid_methods))}"

    timeout = max(1, min(timeout, 60))
    request_headers = _parse_headers(headers)

    import urllib.request
    import urllib.error

    data = _parse_body(body, content_type)
    if data is not None and method in ("GET", "HEAD"):
        return f"[错误] {method} 请求不应包含请求体，请改用 POST/PUT/PATCH"

    if data is not None:
        request_headers.setdefault("Content-Type", content_type)

    req = urllib.request.Request(
        url=url.strip(),
        data=data,
        headers=request_headers,
        method=method,
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status_code = resp.status
            response_headers = dict(resp.headers)
            raw_body = resp.read().decode("utf-8", errors="replace")

            is_json = "json" in response_headers.get("Content-Type", "").lower()
            formatted_body = _try_format_json(raw_body) if is_json else raw_body
            formatted_body = _truncate_response(formatted_body)

            lines = [
                f"✅ HTTP {status_code} {resp.reason}",
                f"URL: {url}",
                "",
                "--- Response Headers ---",
            ]
            for key, value in response_headers.items():
                if key.lower() not in ("transfer-encoding", "connection"):
                    lines.append(f"  {key}: {value}")

            if method not in ("HEAD",):
                lines.extend(["", "--- Response Body ---", formatted_body])

            return "\n".join(lines)

    except urllib.error.HTTPError as exc:
        raw_body = exc.read().decode("utf-8", errors="replace")
        formatted_body = _truncate_response(raw_body)
        return (
            f"❌ HTTP {exc.code} {exc.reason}\n"
            f"URL: {url}\n\n"
            f"--- Response Body ---\n{formatted_body}"
        )
    except urllib.error.URLError as exc:
        return f"❌ 连接失败：{exc}"
    except TimeoutError:
        return f"❌ 请求超时（{timeout} 秒）：{url}"
    except ValueError as exc:
        return f"❌ URL 格式错误：{exc}"
    except Exception as exc:
        return f"❌ 请求失败：{exc}"


@register
@tool
def http_get(url: str, headers: str = "", timeout: int = 15) -> str:
    """发送 GET 请求的快捷方式。

    Args:
        url: 请求 URL
        headers: 请求 Headers（可选）
        timeout: 超时时间（秒）
    """
    return http_request(url=url, method="GET", headers=headers, timeout=timeout)


@register
@tool
def http_post(url: str, body: str = "", headers: str = "", content_type: str = "application/json", timeout: int = 15) -> str:
    """发送 POST 请求的快捷方式。

    Args:
        url: 请求 URL
        body: 请求体内容
        headers: 请求 Headers（可选）
        content_type: Content-Type，默认 application/json
        timeout: 超时时间（秒）
    """
    return http_request(url=url, method="POST", headers=headers, body=body, content_type=content_type, timeout=timeout)