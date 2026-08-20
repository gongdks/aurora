"""Embedding provider — vector embeddings for semantic memory search.

Supports remote API (OpenAI-compatible) with TF-IDF fallback.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import logging
import math
import threading
import time
import urllib.parse
from collections import OrderedDict
from typing import Any

from agent.config import settings

logger = logging.getLogger(__name__)


_CN_EN_MAP: dict[str, str] = {
    "ai": "人工智能",
    "人工智能": "ai",
    "agent": "智能体 agent",
    "智能体": "agent",
    "助手": "agent assistant",
    "assistant": "助手",
    "robot": "机器人",
    "机器人": "robot",
    "bot": "机器人 bot",
    "chat": "对话 聊天",
    "对话": "chat",
    "聊天": "chat",
    "message": "消息",
    "消息": "message",
    "user": "用户",
    "用户": "user",
    "input": "输入",
    "输入": "input",
    "output": "输出",
    "输出": "output",
    "query": "查询",
    "查询": "query",
    "search": "搜索 查找",
    "搜索": "search",
    "查找": "search",
    "find": "查找",
    "help": "帮助",
    "帮助": "help",
    "name": "名字 姓名",
    "名字": "name",
    "姓名": "name",
    "language": "语言",
    "语言": "language",
    "project": "项目",
    "项目": "project",
    "code": "代码 编程",
    "代码": "code programming",
    "编程": "code programming",
    "programming": "编程 代码",
    "python": "python 编程",
    "java": "java 编程",
    "javascript": "javascript js 编程",
    "software": "软件",
    "软件": "software",
    "app": "应用 程序",
    "应用": "app application",
    "程序": "program app",
    "computer": "电脑 计算机",
    "电脑": "computer",
    "计算机": "computer",
    "data": "数据",
    "数据": "data",
    "file": "文件",
    "文件": "file",
    "folder": "文件夹",
    "文件夹": "folder",
    "document": "文档",
    "文档": "document",
    "report": "报告",
    "报告": "report",
    "email": "邮件",
    "邮件": "email",
    "email_address": "邮箱",
    "网址": "url website",
    "website": "网站",
    "网站": "website",
    "link": "链接",
    "链接": "link",
    "image": "图片",
    "图片": "image",
    "photo": "照片",
    "照片": "photo",
    "video": "视频",
    "视频": "video",
    "music": "音乐",
    "音乐": "music",
    "game": "游戏",
    "游戏": "game",
    "book": "书 书籍",
    "书": "book",
    "书籍": "book",
    "movie": "电影",
    "电影": "movie",
    "food": "食物",
    "食物": "food",
    "favorite": "喜欢 偏好",
    "喜欢": "favorite like",
    "偏好": "favorite",
    "love": "喜欢 爱",
    "color": "颜色",
    "颜色": "color",
    "hello": "你好",
    "你好": "hello",
    "goodbye": "再见",
    "再见": "goodbye",
    "thanks": "谢谢",
    "谢谢": "thanks",
    "weather": "天气",
    "天气": "weather",
    "temperature": "温度",
    "温度": "temperature",
    "time": "时间",
    "时间": "time",
    "date": "日期",
    "日期": "date",
    "today": "今天",
    "今天": "today",
    "tomorrow": "明天",
    "明天": "tomorrow",
    "yesterday": "昨天",
    "昨天": "yesterday",
    "now": "现在",
    "现在": "now",
    "year": "年",
    "年": "year",
    "month": "月",
    "月": "month",
    "week": "周",
    "周": "week",
    "day": "日 天",
    "日": "day",
    "天": "day sky",
    "error": "错误",
    "错误": "error",
    "bug": "bug 错误",
    "issue": "问题",
    "问题": "issue problem",
    "problem": "问题",
    "status": "状态",
    "状态": "status",
    "plan": "计划",
    "计划": "plan",
    "tool": "工具",
    "工具": "tool",
    "action": "动作 操作",
    "操作": "action",
    "step": "步骤",
    "步骤": "step",
    "task": "任务",
    "任务": "task",
    "result": "结果",
    "结果": "result",
    "answer": "答案 回答",
    "回答": "answer reply",
    "question": "问题",
    "prompt": "提示",
    "提示": "prompt hint",
    "system": "系统",
    "系统": "system",
    "model": "模型",
    "模型": "model",
    "memory": "记忆 内存",
    "记忆": "memory",
    "knowledge": "知识",
    "知识": "knowledge",
    "context": "上下文",
    "上下文": "context",
    "conversation": "对话",
    "对话": "conversation",
    "session": "会话",
    "会话": "session",
    "function": "函数 功能",
    "函数": "function",
    "variable": "变量",
    "变量": "variable",
    "class": "类",
    "类": "class",
    "method": "方法",
    "方法": "method",
    "api": "接口",
    "接口": "api",
    "server": "服务器",
    "服务器": "server",
    "client": "客户端",
    "客户端": "client",
    "network": "网络",
    "网络": "network",
    "database": "数据库",
    "数据库": "database",
    "sql": "sql 数据库",
    "git": "git 版本控制",
    "version": "版本",
    "版本": "version",
    "test": "测试",
    "测试": "test",
    "debug": "调试",
    "调试": "debug",
    "install": "安装",
    "安装": "install",
    "config": "配置",
    "配置": "config",
    "setting": "设置",
    "设置": "setting",
    "button": "按钮",
    "按钮": "button",
    "window": "窗口",
    "窗口": "window",
    "screen": "屏幕",
    "屏幕": "screen",
    "text": "文本",
    "文本": "text",
    "number": "数字",
    "数字": "number",
    "list": "列表",
    "列表": "list",
    "table": "表格",
    "表格": "table",
    "form": "表单",
    "表单": "form",
    "chart": "图表",
    "图表": "chart",
    "graph": "图 图形",
    "图": "graph",
    "map": "地图",
    "地图": "map",
    "location": "位置",
    "位置": "location",
    "address": "地址",
    "地址": "address",
    "phone": "电话",
    "电话": "phone",
    "money": "钱 金额",
    "钱": "money",
    "金额": "money",
    "price": "价格",
    "价格": "price",
    "cost": "成本",
    "成本": "cost",
    "payment": "支付",
    "支付": "payment",
    "order": "订单",
    "订单": "order",
    "customer": "客户",
    "客户": "customer",
    "business": "业务 商业",
    "业务": "business",
    "商业": "business",
    "report": "报告",
    "报告": "report",
    "analysis": "分析",
    "分析": "analysis",
    "summary": "摘要 总结",
    "总结": "summary",
    "摘要": "summary",
    "translate": "翻译",
    "翻译": "translate",
    "language_model": "语言模型",
    "语言模型": "language model",
    "llm": "大语言模型 llm",
    "大语言模型": "llm",
    "machine_learning": "机器学习",
    "机器学习": "machine learning",
    "deep_learning": "深度学习",
    "深度学习": "deep learning",
    "neural": "神经",
    "神经": "neural",
    "model": "模型",
    "模型": "model",
    "training": "训练",
    "训练": "training",
    "inference": "推理 推断",
    "推理": "inference",
    "推断": "inference",
    "intelligence": "智能",
    "智能": "intelligence",
    "artificial": "人工",
    "人工": "artificial",
    "artificial_intelligence": "人工智能",
    "人工智能": "artificial intelligence",
}


class EmbeddingProvider:
    """Embedding provider with remote API backend and TF-IDF fallback.

    Designed for robustness:
    - Auto-detects available backend on first use
    - Falls back to TF-IDF on any failure
    - Auto-disables remote after 3 consecutive failures
    - Re-enables after a successful call
    """

    def __init__(self) -> None:
        self._provider_type = settings.EMBEDDING_PROVIDER.strip().lower()
        self._dimension = settings.EMBEDDING_DIMENSION
        self._model = settings.EMBEDDING_MODEL
        self._remote_enabled: bool = False
        self._consecutive_failures: int = 0
        self._cache: OrderedDict[str, tuple[list[float], float]] = OrderedDict()
        self._cache_max: int = 512
        self._cache_ttl: float = 600.0
        self._executor = threading.ThreadPoolExecutor(max_workers=2)

        if self._provider_type == "remote":
            self._init_remote_async()

    def _init_remote(self) -> None:
        if self._quick_remote_check():
            self._remote_enabled = True
            self._dimension = settings.REMOTE_EMBEDDING_DIMENSION
            logger.info("[Embedding] Using remote embedding backend: %s", settings.REMOTE_EMBEDDING_BASE_URL)
        else:
            logger.warning("[Embedding] Remote embedding not available, falling back to TF-IDF")
            self._provider_type = "tfidf"

    def _init_remote_async(self) -> None:
        """Start with TF-IDF, probe remote API in background via shared executor."""
        self._provider_type = "tfidf"
        logger.info("[Embedding] Starting with TF-IDF, probing remote in background...")

        def _probe() -> None:
            if self._quick_remote_check():
                self._remote_enabled = True
                self._dimension = settings.REMOTE_EMBEDDING_DIMENSION
                self._provider_type = "remote"
                logger.info("[Embedding] Remote embedding backend detected: %s", settings.REMOTE_EMBEDDING_BASE_URL)
            else:
                logger.info("[Embedding] Remote embedding not available, staying on TF-IDF")

        self._executor.submit(_probe)

    @property
    def available(self) -> bool:
        return self._remote_enabled

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        """Generate embedding vector for text with LRU+TTL cache."""
        cache_key = hashlib.md5(text.encode("utf-8")).hexdigest()
        now = time.time()

        cached = self._cache.get(cache_key)
        if cached is not None:
            vec, expire_at = cached
            if now < expire_at:
                self._cache.move_to_end(cache_key)
                return vec
            del self._cache[cache_key]

        if self._provider_type == "remote" and self._remote_enabled:
            try:
                vec = self._remote_embed(text)
                if vec:
                    self._consecutive_failures = 0
                    self._cache[cache_key] = (vec, now + self._cache_ttl)
                    self._cache.move_to_end(cache_key)
                    self._evict_cache()
                    return vec
                self._consecutive_failures += 1
            except Exception as exc:
                self._consecutive_failures += 1
                logger.debug("[Embedding] Remote failed (%d/3): %s", self._consecutive_failures, exc)
            if self._consecutive_failures >= 3:
                self._remote_enabled = False
                self._consecutive_failures = 0
                self._provider_type = "tfidf"
                logger.warning("[Embedding] Remote disabled after 3 failures, using TF-IDF")

        vec = self._tfidf_embed(text)
        if vec:
            self._cache[cache_key] = (vec, now + self._cache_ttl)
            self._cache.move_to_end(cache_key)
            self._evict_cache()
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts with batch optimization.

        Checks cache first, then batches uncached texts in a single remote
        API call if available. Falls back to TF-IDF for each text.
        """
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        uncached_positions: list[tuple[int, str]] = []
        now = time.time()

        for i, text in enumerate(texts):
            cache_key = hashlib.md5(text.encode("utf-8")).hexdigest()
            cached = self._cache.get(cache_key)
            if cached is not None:
                vec, expire_at = cached
                if now < expire_at:
                    self._cache.move_to_end(cache_key)
                    results[i] = vec
                    continue
                del self._cache[cache_key]
            uncached_positions.append((i, cache_key))

        if not uncached_positions:
            return results

        if self._provider_type == "remote" and self._remote_enabled:
            try:
                batch_vecs = self._remote_embed_batch(
                    [texts[pos] for pos, _ in uncached_positions]
                )
                if batch_vecs and len(batch_vecs) == len(uncached_positions):
                    for (pos, cache_key), vec in zip(uncached_positions, batch_vecs):
                        if vec:
                            results[pos] = vec
                            self._cache[cache_key] = (vec, now + self._cache_ttl)
                            self._cache.move_to_end(cache_key)
                    self._evict_cache()
                    self._consecutive_failures = 0
                    return [
                        v if v is not None else self._tfidf_embed(texts[i])
                        for i, v in enumerate(results)
                    ]
                self._consecutive_failures += 1
            except Exception as exc:
                self._consecutive_failures += 1
                logger.debug("[Embedding] Batch remote failed: %s", exc)
            if self._consecutive_failures >= 3:
                self._remote_enabled = False
                self._consecutive_failures = 0
                self._provider_type = "tfidf"

        for pos, cache_key in uncached_positions:
            vec = self._tfidf_embed(texts[pos])
            results[pos] = vec
            if vec:
                self._cache[cache_key] = (vec, now + self._cache_ttl)
                self._cache.move_to_end(cache_key)

        self._evict_cache()
        return results

    def _evict_cache(self) -> None:
        """Evict expired and overflow entries from cache."""
        now = time.time()
        expired = [k for k, (_, exp) in self._cache.items() if exp <= now]
        for k in expired:
            del self._cache[k]
        while len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)

    def _quick_remote_check(self) -> bool:
        """Quickly check if remote embedding API is available."""
        def _check() -> bool:
            try:
                base = settings.REMOTE_EMBEDDING_BASE_URL
                if not base:
                    return False
                parsed = urllib.parse.urlparse(base)
                host = parsed.hostname
                port = parsed.port or (443 if parsed.scheme == "https" else 80)
                path = parsed.path or "/"
                conn = http.client.HTTPSConnection(host, port, timeout=5) if parsed.scheme == "https" else http.client.HTTPConnection(host, port, timeout=5)
                try:
                    conn.request("GET", path)
                    resp = conn.getresponse()
                    return 200 <= resp.status < 500
                finally:
                    conn.close()
            except Exception:
                return False

        try:
            return self._executor.submit(_check).result(timeout=6)
        except Exception:
            return False

    def _remote_embed(self, text: str) -> list[float]:
        """Call remote embedding API (OpenAI-compatible) with a 10-second timeout."""
        def _do_request() -> list[float]:
            base = settings.REMOTE_EMBEDDING_BASE_URL.rstrip("/")
            parsed = urllib.parse.urlparse(base)
            host = parsed.hostname
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            conn = http.client.HTTPSConnection(host, port, timeout=8) if parsed.scheme == "https" else http.client.HTTPConnection(host, port, timeout=8)
            try:
                model = settings.REMOTE_EMBEDDING_MODEL
                path = (parsed.path or "/v1") + "/embeddings"
                payload = json.dumps({
                    "model": model,
                    "input": text,
                    "encoding_format": "float",
                })
                headers = {
                    "Content-Type": "application/json",
                }
                api_key = settings.REMOTE_EMBEDDING_API_KEY
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                conn.request("POST", path, payload, headers)
                resp = conn.getresponse()
                data = json.loads(resp.read().decode("utf-8"))
                if "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
                    return data["data"][0].get("embedding", [])
                if "embedding" in data:
                    return data["embedding"]
                logger.warning("[Embedding] Unexpected remote response: %s", str(data)[:200])
                return []
            finally:
                conn.close()

        try:
            return self._executor.submit(_do_request).result(timeout=10)
        except Exception:
            logger.debug("[Embedding] Remote embed failed")
            return []

    def _remote_embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Call remote embedding API for batch texts with a 15-second timeout."""
        def _do_request() -> list[list[float]]:
            base = settings.REMOTE_EMBEDDING_BASE_URL.rstrip("/")
            parsed = urllib.parse.urlparse(base)
            host = parsed.hostname
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            conn = http.client.HTTPSConnection(host, port, timeout=12) if parsed.scheme == "https" else http.client.HTTPConnection(host, port, timeout=12)
            try:
                model = settings.REMOTE_EMBEDDING_MODEL
                path = (parsed.path or "/v1") + "/embeddings"
                payload = json.dumps({
                    "model": model,
                    "input": texts,
                    "encoding_format": "float",
                })
                headers = {
                    "Content-Type": "application/json",
                }
                api_key = settings.REMOTE_EMBEDDING_API_KEY
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                conn.request("POST", path, payload, headers)
                resp = conn.getresponse()
                data = json.loads(resp.read().decode("utf-8"))
                results: list[list[float]] = []
                if "data" in data and isinstance(data["data"], list):
                    for item in data["data"]:
                        emb = item.get("embedding", [])
                        results.append(emb if emb else [])
                elif "embedding" in data:
                    emb = data["embedding"]
                    if emb and isinstance(emb[0], list):
                        results = emb
                    else:
                        results = [emb] * len(texts)
                if len(results) != len(texts):
                    while len(results) < len(texts):
                        results.append([])
                return results
            finally:
                conn.close()

        try:
            return self._executor.submit(_do_request).result(timeout=15)
        except Exception:
            logger.debug("[Embedding] Remote batch embed failed")
            return []

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into character n-grams and word tokens.

        Handles mixed Chinese/English text:
        - Chinese chars: individual chars + bigrams + dictionary lookup
        - English words: lowercased words + word bigrams
        - Cross-language expansion via built-in dictionary
        """
        tokens: list[str] = []
        text_lower = text.lower()
        is_chinese = lambda c: '\u4e00' <= c <= '\u9fff'

        i = 0
        while i < len(text_lower):
            ch = text_lower[i]
            if is_chinese(ch):
                matched = False
                for span_len in (4, 3, 2):
                    if i + span_len <= len(text_lower):
                        span = text_lower[i:i + span_len]
                        if all(is_chinese(c) for c in span):
                            if span in _CN_EN_MAP:
                                tokens.append(span)
                                i += span_len
                                matched = True
                                break
                            found = False
                            for k in _CN_EN_MAP:
                                if k.startswith(span) and len(k) >= span_len:
                                    found = True
                                    break
                            if found:
                                tokens.append(span)
                                i += span_len
                                matched = True
                                break
                if not matched:
                    tokens.append(ch)
                    if i > 0 and is_chinese(text_lower[i - 1]):
                        tokens.append(text_lower[i - 1] + ch)
                    i += 1
            elif ch.isalnum() or ch == '_':
                j = i
                word_chars: list[str] = []
                while j < len(text_lower) and not is_chinese(text_lower[j]) and (text_lower[j].isalnum() or text_lower[j] == '_'):
                    word_chars.append(text_lower[j])
                    j += 1
                word = "".join(word_chars)
                tokens.append(word)
                if word in _CN_EN_MAP:
                    mapped = _CN_EN_MAP[word]
                    for m in mapped.split():
                        tokens.append(m)
                if '_' in word:
                    parts = word.split('_')
                    for p in parts:
                        tokens.append(p)
                        if p in _CN_EN_MAP:
                            for m in _CN_EN_MAP[p].split():
                                tokens.append(m)
                i = j
            else:
                i += 1

        expanded: list[str] = []
        for token in tokens:
            expanded.append(token)
            mapped = _CN_EN_MAP.get(token)
            if mapped:
                for m in mapped.split():
                    expanded.append(m)

        word_tokens = [t for t in expanded if not is_chinese(t[0])] if expanded else []
        for i in range(len(word_tokens) - 1):
            expanded.append(word_tokens[i] + "_" + word_tokens[i + 1])

        return expanded

    def _tfidf_embed(self, text: str) -> list[float]:
        """Improved TF-IDF style embedding with multi-hash projection.

        Uses character n-grams + word tokens with multiple hash functions
        to create a sparse but discriminative vector.
        """
        dimension = self._dimension
        vec = [0.0] * dimension
        tokens = self._tokenize(text)

        if not tokens:
            return vec

        for token in tokens:
            h1 = hash(token)
            h2 = hash(token + "_salt1")
            h3 = hash(token + "_salt2")

            idx1 = abs(h1) % dimension
            idx2 = abs(h2) % dimension
            idx3 = abs(h3) % dimension

            if idx1 != idx2 and idx1 != idx3 and idx2 != idx3:
                vec[idx1] += 1.0
                vec[idx2] += 0.5
                vec[idx3] += 0.25
            else:
                vec[idx1] += 1.0

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]

        return vec


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


_embedding_provider: EmbeddingProvider | None = None


def get_embedding_provider() -> EmbeddingProvider:
    """Get or create the singleton embedding provider."""
    global _embedding_provider
    if _embedding_provider is None:
        _embedding_provider = EmbeddingProvider()
    return _embedding_provider


def reset_embedding_provider() -> None:
    """Force recreation of the embedding provider (e.g. after config change)."""
    global _embedding_provider
    _embedding_provider = None