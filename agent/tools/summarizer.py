"""文本摘要工具。

使用 TextRank 算法实现的自动摘要，支持中英文混合文本。
配合搜索/抓取工具，快速总结长文核心内容。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from heapq import nlargest

from langchain.tools import tool

from agent.tools.registry import register

_MAX_INPUT_LEN = 50_000
_DEFAULT_RATIO = 0.3
_MIN_SENTENCES = 3
_MAX_SENTENCES = 15


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []

    sentences = re.split(
        r'(?<=[。！？.!?])\s*',
        text,
    )
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    words = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+|\d+', text)
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "shall", "should", "may", "might", "can", "could",
        "i", "me", "my", "we", "our", "you", "your", "he", "him",
        "she", "her", "it", "its", "they", "them", "their",
        "this", "that", "these", "those", "am", "of", "in", "to",
        "for", "with", "on", "at", "from", "by", "about", "into",
        "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further",
        "then", "once", "and", "but", "or", "nor", "not", "so",
        "yet", "both", "either", "neither", "each", "every", "all",
        "any", "few", "more", "most", "other", "some", "such",
        "no", "only", "own", "same", "than", "too", "very",
        "just", "because", "if", "when", "where", "how", "what",
        "which", "who", "whom", "whose",
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
        "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
        "你", "会", "着", "没有", "看", "好", "自己", "这",
        "那", "但", "而", "与", "或", "及", "等", "对", "为",
        "以", "从", "被", "把", "让", "使", "将", "于", "之",
    }
    return [w for w in words if w not in stopwords and (len(w) > 1 or '\u4e00' <= w <= '\u9fff')]


def _textrank_score(sentences: list[str]) -> list[float]:
    n = len(sentences)
    if n == 0:
        return []

    tokenized = [set(_tokenize(s)) for s in sentences]

    scores = [0.0] * n
    window_size = min(3, n)

    for i in range(n):
        for j in range(max(0, i - window_size), min(n, i + window_size + 1)):
            if i == j:
                continue
            overlap = len(tokenized[i] & tokenized[j])
            if overlap == 0:
                continue
            scores[i] += overlap / (len(tokenized[i]) + len(tokenized[j]))

    if all(s == 0 for s in scores):
        word_freq = Counter()
        for tokens in tokenized:
            word_freq.update(tokens)
        max_freq = max(word_freq.values()) if word_freq else 1
        for i, sent_tokens in enumerate(tokenized):
            if sent_tokens:
                scores[i] = sum(word_freq.get(w, 0) for w in sent_tokens) / max_freq / len(sent_tokens)
            else:
                scores[i] = 0.1

    return scores


def _select_top_sentences(sentences: list[str], scores: list[float], num: int) -> list[str]:
    if num >= len(sentences):
        return sentences

    top_indices = set(nlargest(num, range(len(sentences)), key=lambda i: scores[i]))
    selected = [s for i, s in enumerate(sentences) if i in top_indices]
    return selected


def _build_summary(text: str, num_sentences: int) -> tuple[str, dict]:
    sentences = _split_sentences(text)
    if len(sentences) <= _MIN_SENTENCES:
        return text, {
            "original_sentences": len(sentences),
            "summary_sentences": len(sentences),
            "compression_ratio": 1.0,
            "note": "原文过短，直接返回",
        }

    if len(sentences) <= num_sentences:
        return " ".join(sentences), {
            "original_sentences": len(sentences),
            "summary_sentences": len(sentences),
            "compression_ratio": 1.0,
            "note": "原文过短，无需摘要",
        }

    scores = _textrank_score(sentences)
    top_sentences = _select_top_sentences(sentences, scores, num_sentences)
    summary = " ".join(top_sentences)

    info = {
        "original_sentences": len(sentences),
        "summary_sentences": num_sentences,
        "compression_ratio": round(num_sentences / len(sentences), 2),
    }

    return summary, info


def _extract_key_points(text: str, num_points: int = 5) -> list[str]:
    sentences = _split_sentences(text)
    if not sentences:
        return []

    scores = _textrank_score(sentences)
    scored = [(scores[i], i, s) for i, s in enumerate(sentences)]
    scored.sort(key=lambda x: x[0], reverse=True)

    key_points = []
    used_indices = set()
    for _, idx, sent in scored:
        if len(key_points) >= num_points:
            break
        if any(abs(idx - ui) <= 1 for ui in used_indices):
            continue
        key_points.append(sent.strip())
        used_indices.add(idx)

    key_points.sort(key=lambda s: sentences.index(s) if s in sentences else 999)
    return key_points


@register
@tool
def summarize_text(
    text: str,
    max_sentences: int = 5,
    ratio: float = 0.0,
    include_key_points: bool = True,
) -> str:
    """自动生成文本摘要，支持中英文。

    使用 TextRank 算法提取原文核心句子，适合长文章、新闻、报告的快速总结。

    Args:
        text: 要摘要的文本内容
        max_sentences: 摘要的最大句子数（默认 5 句，最小 1 句，最大 15 句）
        ratio: 压缩比例（0 表示使用 max_sentences；0.1-0.5 表示压缩为原文的 10%-50%）
        include_key_points: 是否同时列出关键要点（默认 True）
    """
    text = text.strip()
    if not text:
        return "❌ 文本不能为空"

    if len(text) < 100:
        return f"⚠️ 文本过短（{len(text)} 字符），无需摘要：\n\n{text}"

    if len(text) > _MAX_INPUT_LEN:
        return f"❌ 文本过长（{len(text)} > {_MAX_INPUT_LEN} 字符），请分段摘要"

    sentences = _split_sentences(text)
    n = len(sentences)

    if ratio > 0 and 0 < ratio <= 1:
        target = max(_MIN_SENTENCES, min(int(n * ratio), _MAX_SENTENCES))
    else:
        target = max(1, min(max_sentences, _MAX_SENTENCES))

    summary, info = _build_summary(text, target)

    lines = [
        "📝 文本摘要：",
        "",
        summary,
        "",
        f"📊 统计：原文 {info['original_sentences']} 句 → 摘要 {info['summary_sentences']} 句"
        f"（压缩率 {info['compression_ratio']*100:.0f}%）",
    ]

    if include_key_points:
        key_points = _extract_key_points(text, num_points=min(5, target))
        if key_points:
            lines.append("")
            lines.append("🔑 关键要点：")
            for i, kp in enumerate(key_points, 1):
                lines.append(f"  {i}. {kp}")

    return "\n".join(lines)


@register
@tool
def summarize_file(
    filepath: str,
    max_sentences: int = 5,
    include_key_points: bool = True,
) -> str:
    """读取项目目录内的文本文件并生成摘要。

    Args:
        filepath: 文件路径，如 "output/article.txt"、"data/report.md"
        max_sentences: 摘要的最大句子数（默认 5）
        include_key_points: 是否列出关键要点（默认 True）
    """
    from agent.config import settings
    from agent.utils.path_guard import safe_resolve

    try:
        safe = safe_resolve(filepath, settings.FILE_READER_ROOT)
    except ValueError as exc:
        return f"路径错误：{exc}"

    if not os.path.isfile(safe):
        return f"❌ 文件不存在：{filepath}"

    try:
        with open(safe, encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError as exc:
        return f"❌ 无法读取文件：{exc}"

    rel_path = os.path.relpath(safe, os.path.realpath(settings.FILE_READER_ROOT))
    result = summarize_text.invoke({
        "text": content,
        "max_sentences": max_sentences,
        "ratio": 0.0,
        "include_key_points": include_key_points,
    })

    return f"📄 文件摘要：{rel_path}\n\n{result}"


@register
@tool
def summarize_url(
    url: str,
    max_sentences: int = 5,
    include_key_points: bool = True,
) -> str:
    """抓取网页内容并生成摘要。

    先抓取网页正文，再进行自动摘要。

    Args:
        url: 网页 URL，如 "https://example.com/article"
        max_sentences: 摘要的最大句子数（默认 5）
        include_key_points: 是否列出关键要点（默认 True）
    """
    from agent.tools.web_content_fetcher import web_content_fetcher

    fetch_result = web_content_fetcher.invoke({"url": url})

    if fetch_result.startswith("[错误]"):
        return f"❌ 抓取网页失败：{fetch_result}"

    result = summarize_text.invoke({
        "text": fetch_result,
        "max_sentences": max_sentences,
        "ratio": 0.0,
        "include_key_points": include_key_points,
    })

    return f"🌐 网页摘要：{url}\n\n{result}"