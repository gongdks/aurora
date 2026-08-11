"""Shared HTTP fetching utilities — used by web_search and web_fetcher."""

from __future__ import annotations

import logging
from urllib.request import Request, urlopen
from urllib.parse import urlparse
from typing import Tuple

logger = logging.getLogger(__name__)

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

_ALLOWED_SCHEMES = frozenset({"http", "https"})


def http_fetch(
    url: str,
    timeout: int = 15,
    extra_headers: dict | None = None,
) -> Tuple[str, int, str]:
    """Fetch a URL and return (decoded_text, status_code, content_type).

    Tries multiple encodings (utf-8, gbk, gb2312, latin-1) for robustness.
    Only allows http/https schemes to prevent SSRF.
    """
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"不允许的 URL scheme: {parsed.scheme}，仅支持 http/https")

    headers = dict(_DEFAULT_HEADERS)
    if extra_headers:
        headers.update(extra_headers)

    req = Request(url, headers=headers)
    with urlopen(req, timeout=timeout) as resp:
        status_code = resp.status
        content_type = resp.headers.get("Content-Type", "")
        raw = resp.read()

        for encoding in ("utf-8", "gbk", "gb2312", "latin-1"):
            try:
                html = raw.decode(encoding)
                return html, status_code, content_type
            except UnicodeDecodeError:
                continue

        html = raw.decode("utf-8", errors="replace")
        return html, status_code, content_type