"""增强版网页内容抓取工具。

优先使用 requests + BeautifulSoup 抓取指定 URL，
提取结构化内容：标题、meta 描述、层级标题、正文段落、链接等。
若未安装 requests/beautifulsoup4，则回退到 Python 标准库实现。
"""

import re
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

from langchain.tools import tool

from agent.tools.registry import register
from agent.utils.http_fetcher import http_fetch

_HAS_REQUESTS = False
_HAS_BS4 = False

try:
    import requests
    from bs4 import BeautifulSoup
    _HAS_REQUESTS = True
    _HAS_BS4 = True
except ImportError:
    pass


def _resolve_url(base_url: str, href: str) -> str:
    try:
        return urljoin(base_url, href)
    except Exception:
        return href


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0].rstrip() + "..."


# ─── BeautifulSoup 路径 ────────────────────────────────────────────

def _fetch_with_requests(url: str) -> tuple[str, str, int]:
    """使用 requests 获取页面内容。返回 (html, content_type, status_code)。"""
    resp = requests.get(url, headers=_DEFAULT_HEADERS, timeout=20)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text, resp.headers.get("Content-Type", ""), resp.status_code


def _extract_bs4(html: str, base_url: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
        tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    meta_desc = ""
    meta_tag = soup.find("meta", attrs={"name": "description"})
    if meta_tag and meta_tag.get("content"):
        meta_desc = meta_tag["content"].strip()
    if not meta_desc:
        meta_tag = soup.find("meta", attrs={"property": "og:description"})
        if meta_tag and meta_tag.get("content"):
            meta_desc = meta_tag["content"].strip()

    og_title = ""
    og_tag = soup.find("meta", attrs={"property": "og:title"})
    if og_tag and og_tag.get("content"):
        og_title = og_tag["content"].strip()

    canonical = ""
    link_tag = soup.find("link", attrs={"rel": "canonical"})
    if link_tag and link_tag.get("href"):
        canonical = _resolve_url(base_url, link_tag["href"])

    headings: list[dict] = []
    for level in range(1, 7):
        for h in soup.find_all(f"h{level}"):
            text = h.get_text(strip=True)
            if text:
                headings.append({"level": level, "text": _truncate(text, 120)})

    paragraphs: list[str] = []
    for p in soup.find_all("p"):
        text = p.get_text(strip=True)
        if len(text) >= 10:
            paragraphs.append(_truncate(text, 300))

    main_text = "\n\n".join(paragraphs[:30])

    links: list[dict] = []
    seen_urls = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        abs_url = _resolve_url(base_url, href)
        if abs_url in seen_urls:
            continue
        seen_urls.add(abs_url)
        link_text = a.get_text(strip=True) or "(无文字)"
        links.append({
            "text": _truncate(link_text, 80),
            "url": abs_url,
        })
        if len(links) >= 30:
            break

    images: list[dict] = []
    for img in soup.find_all("img", src=True):
        src = img["src"].strip()
        if not src or src.startswith("data:"):
            continue
        abs_src = _resolve_url(base_url, src)
        alt = img.get("alt", "").strip()
        images.append({
            "alt": _truncate(alt, 60),
            "url": abs_src,
        })
        if len(images) >= 20:
            break

    return {
        "title": title,
        "og_title": og_title,
        "meta_description": meta_desc,
        "canonical_url": canonical,
        "headings": headings,
        "main_text": main_text,
        "paragraphs": paragraphs[:15],
        "links": links,
        "images": images,
    }


# ─── 标准库回退路径 ────────────────────────────────────────────────

class _SimpleHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.title = ""
        self.in_title = False
        self.meta_desc = ""
        self.og_title = ""
        self.canonical = ""
        self.headings: list[dict] = []
        self.paragraphs: list[str] = []
        self.links: list[dict] = []
        self.images: list[dict] = []
        self._current_tag = ""
        self._current_href = ""
        self._current_text = ""
        self._skip = False
        self._skip_tags = {"script", "style", "noscript", "iframe", "svg"}
        self._heading_level = 0
        self._in_paragraph = False
        self._paragraph_text = ""

    def handle_starttag(self, tag, attrs):
        if tag in self._skip_tags:
            self._skip = True
            return
        self._current_tag = tag
        attrs_dict = dict(attrs)
        if tag == "title":
            self.in_title = True
        elif tag == "meta":
            name = attrs_dict.get("name", "")
            prop = attrs_dict.get("property", "")
            content = attrs_dict.get("content", "")
            if name == "description" and content:
                self.meta_desc = content.strip()
            elif prop == "og:description" and content and not self.meta_desc:
                self.meta_desc = content.strip()
            elif prop == "og:title" and content:
                self.og_title = content.strip()
        elif tag == "link":
            rel = attrs_dict.get("rel", "")
            href = attrs_dict.get("href", "")
            if "canonical" in rel and href:
                self.canonical = href
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._heading_level = int(tag[1])
            self._current_text = ""
        elif tag == "a":
            self._current_href = attrs_dict.get("href", "")
            self._current_text = ""
        elif tag == "img":
            src = attrs_dict.get("src", "")
            if src and not src.startswith("data:"):
                alt = attrs_dict.get("alt", "")
                self.images.append({
                    "alt": _truncate(alt.strip(), 60),
                    "url": src,
                })
        elif tag == "p":
            self._in_paragraph = True
            self._paragraph_text = ""

    def handle_endtag(self, tag):
        if tag in self._skip_tags:
            self._skip = False
            return
        if tag == "title":
            self.in_title = False
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            if self._current_text.strip():
                self.headings.append({
                    "level": self._heading_level,
                    "text": _truncate(self._current_text.strip(), 120),
                })
            self._heading_level = 0
            self._current_text = ""
        elif tag == "a":
            if self._current_href and self._current_text.strip():
                href = self._current_href.strip()
                if not href.startswith(("#", "javascript:", "mailto:", "tel:")):
                    self.links.append({
                        "text": _truncate(self._current_text.strip(), 80),
                        "url": href,
                    })
            self._current_href = ""
            self._current_text = ""
        elif tag == "p":
            self._in_paragraph = False
            if self._paragraph_text.strip() and len(self._paragraph_text.strip()) >= 10:
                self.paragraphs.append(_truncate(self._paragraph_text.strip(), 300))
            self._paragraph_text = ""

    def handle_data(self, data):
        if self._skip:
            return
        if self.in_title:
            self.title += data
        if self._heading_level > 0:
            self._current_text += data
        if self._current_tag == "a":
            self._current_text += data
        if self._in_paragraph:
            self._paragraph_text += data


def _fetch_with_urllib(url: str) -> tuple[str, str, int]:
    """使用共享 HTTP 工具获取页面内容。返回 (html, content_type, status_code)。"""
    html, status_code, content_type = http_fetch(url, timeout=20)
    return html, content_type, status_code


def _extract_fallback(html: str, base_url: str) -> dict:
    parser = _SimpleHTMLParser()
    parser.feed(html)

    main_text = "\n\n".join(parser.paragraphs[:30])

    return {
        "title": parser.title.strip(),
        "og_title": parser.og_title,
        "meta_description": parser.meta_desc,
        "canonical_url": _resolve_url(base_url, parser.canonical) if parser.canonical else "",
        "headings": parser.headings,
        "main_text": main_text,
        "paragraphs": parser.paragraphs[:15],
        "links": parser.links[:30],
        "images": parser.images[:20],
    }


# ─── 格式化输出 ────────────────────────────────────────────────────

def _format_output(data: dict, url: str, extract_mode: str) -> str:
    lines: list[str] = []
    lines.append(f"=== 网页抓取结果 ===")
    lines.append(f"URL: {url}")
    lines.append("")

    if extract_mode in ("meta", "all"):
        if data["title"]:
            lines.append(f"【页面标题】{data['title']}")
        if data["og_title"] and data["og_title"] != data["title"]:
            lines.append(f"【OG标题】{data['og_title']}")
        if data["meta_description"]:
            lines.append(f"【Meta描述】{data['meta_description']}")
        if data["canonical_url"]:
            lines.append(f"【规范链接】{data['canonical_url']}")
        lines.append("")

    if extract_mode in ("headings", "all"):
        if data["headings"]:
            lines.append("【层级标题】")
            for h in data["headings"]:
                indent = "  " * (h["level"] - 1)
                lines.append(f"{indent}H{h['level']}: {h['text']}")
            lines.append("")

    if extract_mode in ("content", "all"):
        if data["main_text"]:
            lines.append("【正文内容】")
            lines.append(data["main_text"][:4000])
            lines.append("")

    if extract_mode in ("links", "all"):
        if data["links"]:
            lines.append(f"【链接】(共 {len(data['links'])} 个，显示前 20 个)")
            for i, lnk in enumerate(data["links"][:20], 1):
                lines.append(f"  {i}. {lnk['text']}")
                lines.append(f"     {lnk['url']}")
            lines.append("")

    if extract_mode in ("images", "all"):
        if data["images"]:
            lines.append(f"【图片】(共 {len(data['images'])} 个，显示前 10 个)")
            for i, img in enumerate(data["images"][:10], 1):
                alt_info = f" [{img['alt']}]" if img["alt"] else ""
                lines.append(f"  {i}.{alt_info} {img['url']}")
            lines.append("")

    if extract_mode in ("content", "all") and not data["main_text"]:
        lines.append("(页面无可提取的正文内容，可能是纯 JavaScript 渲染页面)")

    return "\n".join(lines)


# ─── 主入口 ────────────────────────────────────────────────────────

@register(tags={"web"})
@tool
def web_content_fetcher(url: str, extract_mode: str = "all") -> str:
    """抓取网页内容，返回结构化信息（标题/正文/链接/图片等）。"""
    try:
        parsed = urlparse(url.strip())
        if parsed.scheme not in ("http", "https"):
            return f"[ERR] 不支持的协议：{parsed.scheme}，仅支持 http/https"
        if not parsed.netloc:
            return f"[ERR] 无效的 URL：{url}"

        mode = extract_mode.strip().lower()
        valid_modes = {"all", "meta", "headings", "content", "links", "images"}
        if mode not in valid_modes:
            return f"[ERR] 无效的 extract_mode：{extract_mode}，可选值：{', '.join(sorted(valid_modes))}"

        target_url = url.strip()

        if _HAS_REQUESTS and _HAS_BS4:
            html, content_type, status_code = _fetch_with_requests(target_url)
            if "text/html" not in content_type and "text/plain" not in content_type:
                return f"[ERR] 非文本/html 内容类型：{content_type}，状态码：{status_code}"
            if len(html) > 5_000_000:
                html = html[:5_000_000]
            data = _extract_bs4(html, target_url)
        else:
            html, content_type, status_code = _fetch_with_urllib(target_url)
            if "text/html" not in content_type and "text/plain" not in content_type:
                return f"[ERR] 非文本/html 内容类型：{content_type}，状态码：{status_code}"
            if len(html) > 5_000_000:
                html = html[:5_000_000]
            data = _extract_fallback(html, target_url)

        if not any([
            data["title"], data["meta_description"],
            data["headings"], data["main_text"], data["links"],
        ]):
            return (
                f"[ERR] 页面抓取成功（状态码 {status_code}），"
                f"但无可提取的结构化内容。可能是纯 JavaScript 渲染页面，"
                f"当前不支持此类页面。"
            )

        return _format_output(data, target_url, mode)

    except Exception as exc:
        return f"[ERR] 抓取失败：{exc}"