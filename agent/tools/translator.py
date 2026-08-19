"""多语言翻译工具。

支持中文、英文、日文、韩文、法文、德文、西班牙文等主流语言互译。
使用免费的 MyMemory Translation API，无需 API Key。
"""

from __future__ import annotations

import json
import re
from urllib.parse import quote

from langchain.tools import tool

from agent.tools.registry import register

_LANG_MAP = {
    "zh": "zh-CN", "cn": "zh-CN", "chinese": "zh-CN", "中文": "zh-CN",
    "en": "en", "english": "en", "英文": "en", "英语": "en",
    "ja": "ja", "jp": "ja", "japanese": "ja", "日文": "ja", "日语": "ja",
    "ko": "ko", "korean": "ko", "韩文": "ko", "韩语": "ko",
    "fr": "fr", "french": "fr", "法文": "fr", "法语": "fr",
    "de": "de", "german": "de", "德文": "de", "德语": "de",
    "es": "es", "spanish": "es", "西班牙文": "es", "西班牙语": "es",
    "ru": "ru", "russian": "ru", "俄文": "ru", "俄语": "ru",
    "it": "it", "italian": "it", "意大利文": "it",
    "pt": "pt", "portuguese": "pt", "葡萄牙文": "it",
    "ar": "ar", "arabic": "ar", "阿拉伯文": "ar",
    "th": "th", "thai": "th", "泰文": "th",
    "vi": "vi", "vietnamese": "vi", "越南文": "vi",
}

_MAX_TEXT_LEN = 5000
_BATCH_SEP = "||"


def _resolve_lang(code: str) -> str | None:
    code = code.strip().lower()
    if not code:
        return None
    if code in _LANG_MAP:
        return _LANG_MAP[code]
    if "-" in code or code.isalpha():
        return code
    return None


def _split_chunks(text: str, max_len: int = 450) -> list[str]:
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    lines = text.split("\n")
    current = ""
    for line in lines:
        if len(current) + len(line) + 1 > max_len:
            if current:
                chunks.append(current.strip())
            current = line
        else:
            current = current + "\n" + line if current else line
    if current:
        chunks.append(current.strip())
    return chunks


def _translate_chunk(text: str, src: str, dst: str, timeout: int = 15) -> tuple[str, str | None]:
    import urllib.request
    import urllib.error

    url = (
        f"https://api.mymemory.translated.net/get"
        f"?q={quote(text)}"
        f"&langpair={quote(src)}|{quote(dst)}"
    )

    req = urllib.request.Request(
        url=url,
        headers={
            "User-Agent": "Mozilla/5.0 (Agent Translator/1.0)",
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            data = json.loads(raw)

        translated = data.get("responseData", {}).get("translatedText", "")
        if not translated:
            match = data.get("matches", [{}])[0] if data.get("matches") else {}
            translated = match.get("translation", "")

        if not translated:
            return text, "API 未返回翻译结果"

        return translated, None

    except urllib.error.URLError as exc:
        return text, f"网络错误: {exc}"
    except urllib.error.HTTPError as exc:
        return text, f"HTTP {exc.code}"
    except json.JSONDecodeError:
        return text, "API 返回格式错误"
    except TimeoutError:
        return text, "请求超时"


@register(tags={"core", "translate"})
@tool
def translate(
    text: str,
    source_lang: str = "auto",
    target_lang: str = "en",
) -> str:
    """翻译文本，支持多语言互译。"""
    text = text.strip()
    if not text:
        return "❌ 文本不能为空"

    if len(text) > _MAX_TEXT_LEN:
        return f"❌ 文本过长（{len(text)} > {_MAX_TEXT_LEN} 字符），请分段翻译"

    src = _resolve_lang(source_lang)
    if not src:
        return f"❌ 无法识别源语言：{source_lang}。支持的语言：zh, en, ja, ko, fr, de, es, ru, it, pt, ar, th, vi"

    dst = _resolve_lang(target_lang)
    if not dst:
        return f"❌ 无法识别目标语言：{target_lang}。支持的语言：zh, en, ja, ko, fr, de, es, ru, it, pt, ar, th, vi"

    if src == dst:
        return f"⚠️ 源语言和目标语言相同（{src}），无需翻译"

    chunks = _split_chunks(text)
    translated_parts: list[str] = []
    errors: list[str] = []

    for chunk in chunks:
        result, err = _translate_chunk(chunk, src, dst)
        translated_parts.append(result)
        if err:
            errors.append(err)

    translated = "\n".join(translated_parts)

    output_lines = [
        f"📝 翻译结果（{src} → {dst}）：",
        "",
        translated,
    ]

    if errors:
        output_lines.extend([
            "",
            f"⚠️ {len(errors)} 个片段翻译可能存在问题：{'; '.join(errors)}",
        ])

    return "\n".join(output_lines)


@register(tags={"translate"})
@tool
def translate_batch(
    texts: str,
    target_lang: str = "en",
    source_lang: str = "auto",
) -> str:
    """批量翻译多条文本（用 || 分隔）。"""
    items = [t.strip() for t in texts.split(_BATCH_SEP) if t.strip()]
    if not items:
        return "❌ 没有可翻译的文本"

    results: list[str] = []
    for i, text in enumerate(items, 1):
        result = translate.invoke({"text": text, "source_lang": source_lang, "target_lang": target_lang})
        results.append(f"{i}. {result}")

    return f"📝 批量翻译结果（共 {len(items)} 条）：\n\n" + "\n\n".join(results)