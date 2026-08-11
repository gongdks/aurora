"""多引擎搜索工具。

支持百度、必应、搜狗、360、Google 五个搜索引擎。
支持网页搜索和资讯（新闻）搜索两种模式。
统一返回标题、摘要、来源和链接。
"""

import re
from urllib.parse import quote, urljoin

from langchain.tools import tool

from agent.tools.registry import register
from agent.utils.cache import tool_cache
from agent.utils.http_fetcher import http_fetch

_HAS_BS4 = False
try:
    from bs4 import BeautifulSoup
    _HAS_BS4 = True
except ImportError:
    pass

_ENGINES = {
    "baidu": {
        "label": "百度",
        "web_url": "https://www.baidu.com/s?wd={q}",
        "news_url": "https://www.baidu.com/s?tn=news&rtt=1&bsst=1&cl=2&wd={q}",
        "base": "https://www.baidu.com",
    },
    "bing": {
        "label": "必应",
        "web_url": "https://www.bing.com/search?q={q}",
        "news_url": "https://www.bing.com/news/search?q={q}",
        "base": "https://www.bing.com",
    },
    "sogou": {
        "label": "搜狗",
        "web_url": "https://www.sogou.com/web?query={q}",
        "news_url": "https://news.sogou.com/news?query={q}",
        "base": "https://www.sogou.com",
    },
    "360": {
        "label": "360",
        "web_url": "https://www.so.com/s?q={q}",
        "news_url": "https://news.so.com/ns?q={q}",
        "base": "https://www.so.com",
    },
    "google": {
        "label": "Google",
        "web_url": "https://www.google.com/search?q={q}&hl=zh-CN",
        "news_url": "https://www.google.com/search?q={q}&tbm=nws&hl=zh-CN",
        "base": "https://www.google.com",
    },
}


def _fetch_search_page(url: str) -> tuple[str, int, str]:
    """抓取搜索结果页面，返回 (html, status_code, content_type)。"""
    return http_fetch(url)


def _parse_baidu(html: str, base: str) -> list[dict]:
    results: list[dict] = []
    if _HAS_BS4:
        soup = BeautifulSoup(html, "html.parser")
        for item in soup.select("div.result, div.result-op, div.c-container"):
            link_tag = item.select_one("h3 a")
            if not link_tag:
                continue
            title = link_tag.get_text(strip=True)
            href = link_tag.get("href", "")
            if not title or not href:
                continue
            if href.startswith("/"):
                href = urljoin(base, href)
            abs_tag = item.select_one(
                "div.c-abstract, span.c-color-text, div.c-span-last, "
                "div[class*='content'], div[class*='abstract']"
            )
            abstract = abs_tag.get_text(strip=True) if abs_tag else ""
            src_tag = item.select_one("span.c-color-gray, p.c-author, span[class*='source']")
            source = src_tag.get_text(strip=True) if src_tag else ""
            results.append({
                "title": title, "url": href,
                "abstract": abstract[:200], "source": source[:80],
            })
            if len(results) >= 20:
                break
    else:
        for m in re.finditer(
            r'<h3[^>]*>.*?<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
            html, re.DOTALL,
        ):
            href = m.group(1).strip()
            title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
            if href.startswith("/"):
                href = urljoin(base, href)
            if title and len(title) > 2:
                results.append({"title": title, "url": href, "abstract": "", "source": ""})
            if len(results) >= 20:
                break
    return results


def _parse_bing(html: str, base: str) -> list[dict]:
    results: list[dict] = []
    if _HAS_BS4:
        soup = BeautifulSoup(html, "html.parser")
        for item in soup.select("li.b_algo, div.news-card, div.newsitem"):
            link_tag = item.select_one("h2 a, a.title")
            if not link_tag:
                continue
            title = link_tag.get_text(strip=True)
            href = link_tag.get("href", "")
            if not title or not href:
                continue
            if href.startswith("/"):
                href = urljoin(base, href)
            abs_tag = item.select_one("p, div.snippet, div.news_desc")
            abstract = abs_tag.get_text(strip=True) if abs_tag else ""
            src_tag = item.select_one("span.source, cite, .news_source")
            source = src_tag.get_text(strip=True) if src_tag else ""
            results.append({
                "title": title, "url": href,
                "abstract": abstract[:200], "source": source[:80],
            })
            if len(results) >= 20:
                break
    else:
        for m in re.finditer(
            r'<h2[^>]*>.*?<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
            html, re.DOTALL,
        ):
            href = m.group(1).strip()
            title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
            if href.startswith("/"):
                href = urljoin(base, href)
            if title and len(title) > 2:
                results.append({"title": title, "url": href, "abstract": "", "source": ""})
            if len(results) >= 20:
                break
    return results


def _parse_sogou(html: str, base: str) -> list[dict]:
    results: list[dict] = []
    if _HAS_BS4:
        soup = BeautifulSoup(html, "html.parser")
        for item in soup.select("div.results div.vrwrap, div.news-list li, div.rb"):
            link_tag = item.select_one("h3 a, a.news-title")
            if not link_tag:
                continue
            title = link_tag.get_text(strip=True)
            href = link_tag.get("href", "")
            if not title or not href:
                continue
            if href.startswith("/"):
                href = urljoin(base, href)
            abs_tag = item.select_one("div.ft, p.star-wiki, div[class*='content']")
            abstract = abs_tag.get_text(strip=True) if abs_tag else ""
            src_tag = item.select_one("span[class*='source'], p[class*='news-from']")
            source = src_tag.get_text(strip=True) if src_tag else ""
            results.append({
                "title": title, "url": href,
                "abstract": abstract[:200], "source": source[:80],
            })
            if len(results) >= 20:
                break
    else:
        for m in re.finditer(
            r'<h3[^>]*>.*?<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
            html, re.DOTALL,
        ):
            href = m.group(1).strip()
            title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
            if href.startswith("/"):
                href = urljoin(base, href)
            if title and len(title) > 2:
                results.append({"title": title, "url": href, "abstract": "", "source": ""})
            if len(results) >= 20:
                break
    return results


def _parse_360(html: str, base: str) -> list[dict]:
    results: list[dict] = []
    if _HAS_BS4:
        soup = BeautifulSoup(html, "html.parser")
        for item in soup.select("li.res-list, div.news-item, div.result"):
            link_tag = item.select_one("h3 a, a.news-title")
            if not link_tag:
                continue
            title = link_tag.get_text(strip=True)
            href = link_tag.get("href", "")
            if not title or not href:
                continue
            if href.startswith("/"):
                href = urljoin(base, href)
            abs_tag = item.select_one("p.res-desc, div[class*='content'], div[class*='desc']")
            abstract = abs_tag.get_text(strip=True) if abs_tag else ""
            src_tag = item.select_one("span[class*='source'], cite")
            source = src_tag.get_text(strip=True) if src_tag else ""
            results.append({
                "title": title, "url": href,
                "abstract": abstract[:200], "source": source[:80],
            })
            if len(results) >= 20:
                break
    else:
        for m in re.finditer(
            r'<h3[^>]*>.*?<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
            html, re.DOTALL,
        ):
            href = m.group(1).strip()
            title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
            if href.startswith("/"):
                href = urljoin(base, href)
            if title and len(title) > 2:
                results.append({"title": title, "url": href, "abstract": "", "source": ""})
            if len(results) >= 20:
                break
    return results


def _parse_google(html: str, base: str) -> list[dict]:
    results: list[dict] = []
    if _HAS_BS4:
        soup = BeautifulSoup(html, "html.parser")
        for item in soup.select("div.g, div.dbsr, div.SoBEf"):
            link_tag = item.select_one("a[href]")
            if not link_tag:
                continue
            href = link_tag.get("href", "")
            if not href or href.startswith("/"):
                continue
            if "google.com" in href and "/search?" in href:
                continue
            title_tag = item.select_one("h3")
            if not title_tag:
                continue
            title = title_tag.get_text(strip=True)
            if not title:
                continue
            abs_tag = item.select_one(
                "div.VwiC3b, div[data-sncf], span.st, div[class*='content']"
            )
            abstract = abs_tag.get_text(strip=True) if abs_tag else ""
            results.append({
                "title": title, "url": href,
                "abstract": abstract[:200], "source": "",
            })
            if len(results) >= 20:
                break
    else:
        for m in re.finditer(
            r'<a[^>]*href="(https?://[^"]*)"[^>]*>\s*<h3[^>]*>(.*?)</h3>',
            html, re.DOTALL,
        ):
            href = m.group(1).strip()
            title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
            if title and len(title) > 2:
                results.append({"title": title, "url": href, "abstract": "", "source": ""})
            if len(results) >= 20:
                break
    return results


_PARSERS = {
    "baidu": _parse_baidu,
    "bing": _parse_bing,
    "sogou": _parse_sogou,
    "360": _parse_360,
    "google": _parse_google,
}


def _format_results(results: list[dict], query: str, engine: str, search_type: str) -> str:
    engine_label = _ENGINES[engine]["label"]
    type_label = "资讯" if search_type == "news" else "网页"
    lines: list[str] = [
        f"=== {engine_label}{type_label}搜索结果 ===",
        f"查询: {query}",
        f"共找到 {len(results)} 条结果\n",
    ]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}")
        if r.get("source"):
            lines.append(f"   来源: {r['source']}")
        if r.get("abstract"):
            lines.append(f"   摘要: {r['abstract']}")
        lines.append(f"   链接: {r['url']}")
        lines.append("")
    if not results:
        lines.append("未找到相关搜索结果。")
    return "\n".join(lines)


@register
@tool
@tool_cache(key_func=lambda *a, **kw: f"{a[0] if a else kw.get('query','')}:{kw.get('engine', 'baidu')}:{kw.get('search_type', 'web')}", ttl=600)
def web_search(query: str, engine: str = "baidu", search_type: str = "web") -> str:
    """使用指定搜索引擎搜索信息，返回标题、摘要和链接。

    支持 5 个搜索引擎：百度、必应、搜狗、360、Google。
    支持网页搜索和资讯（新闻）搜索两种模式。

    Args:
        query: 搜索关键词，例如 "今天的 AI 新闻"
        engine: 搜索引擎，可选值：
            - "baidu": 百度（默认）
            - "bing": 必应
            - "sogou": 搜狗
            - "360": 360 搜索
            - "google": Google
        search_type: 搜索类型，可选值：
            - "web": 网页搜索（默认）
            - "news": 资讯/新闻搜索
    """
    try:
        q = query.strip()
        if not q:
            return "[ERR] 搜索关键词不能为空"

        eng = engine.strip().lower()
        if eng not in _ENGINES:
            return f"[ERR] 无效的 engine：{engine}，可选值：{', '.join(sorted(_ENGINES.keys()))}"

        st = search_type.strip().lower()
        if st not in ("web", "news"):
            return f"[ERR] 无效的 search_type：{search_type}，可选值：web, news"

        cfg = _ENGINES[eng]
        url_template = cfg["news_url"] if st == "news" else cfg["web_url"]
        target_url = url_template.format(q=quote(q))

        html, status_code, content_type = _fetch_search_page(target_url)

        if status_code != 200:
            return f"[ERR] {cfg['label']}搜索请求失败，状态码：{status_code}"

        if "text/html" not in content_type and "text/plain" not in content_type:
            return f"[ERR] 非文本内容类型：{content_type}"

        parser_fn = _PARSERS[eng]
        results = parser_fn(html, cfg["base"])

        if not results:
            return (
                f"{cfg['label']}搜索成功（状态码 {status_code}），但未解析到结果。"
                f"可能需要使用 `web_content_fetcher` 直接访问相关网站。"
            )

        return _format_results(results, q, eng, st)

    except Exception as exc:
        return f"[ERR] 搜索失败：{exc}"