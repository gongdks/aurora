"""轻量级 LLM 响应缓存 —— 避免重复 API 调用。

使用基于 (model, temperature, prompt_hash) 的键值存储，
支持 TTL 过期和最大条目数限制。
同时提供 @tool_cache 装饰器用于工具结果缓存。
"""

from __future__ import annotations

import functools
import hashlib
import threading
import time
from collections import OrderedDict
from typing import Any, Callable


class LLMCache:
    """LRU + TTL 缓存，用于 LLM 调用结果。

    线程安全（使用 Lock），支持命中缓存时跳过 API 调用。

    Usage:
        cache = LLMCache(max_size=256, ttl=300)
        key = cache.make_key(model, temperature, messages)
        cached = cache.get(key)
        if cached is not None:
            return cached
        result = llm.invoke(messages)
        cache.set(key, result)
    """

    def __init__(self, max_size: int = 256, ttl: float = 300.0) -> None:
        self._max_size = max_size
        self._ttl = ttl
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    def make_key(self, *parts: Any) -> str:
        """从任意组件生成缓存键。"""
        hasher = hashlib.sha256()
        for part in parts:
            hasher.update(str(part).encode("utf-8"))
            hasher.update(b"\x00")
        return hasher.hexdigest()

    def get(self, key: str) -> Any | None:
        """获取缓存值，过期或不存在返回 None。线程安全。"""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            value, expire_at = entry
            if time.time() > expire_at:
                del self._store[key]
                self._misses += 1
                return None
            self._store.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        """设置缓存值，自动淘汰过期条目和超出上限的条目。线程安全。"""
        with self._lock:
            expire_at = time.time() + self._ttl
            self._store[key] = (value, expire_at)
            self._store.move_to_end(key)
            self._evict_locked()

    def _evict_locked(self) -> None:
        """淘汰过期条目和最旧条目。必须在锁内调用。"""
        now = time.time()
        expired = [k for k, (_, exp) in self._store.items() if exp <= now]
        for k in expired:
            del self._store[k]

        while len(self._store) > self._max_size:
            self._store.popitem(last=False)

    @property
    def stats(self) -> dict[str, int]:
        """返回缓存统计。"""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 3),
                "size": len(self._store),
            }

    def clear(self) -> None:
        """清空缓存。"""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0


# Global singleton cache for LLM responses
_global_cache = LLMCache(max_size=512, ttl=600)


def get_cache() -> LLMCache:
    """Get the global LLM response cache instance."""
    return _global_cache


def tool_cache(
    key_func: Callable[..., str] | None = None,
    ttl: float | None = None,
) -> Callable:
    """装饰器：缓存工具函数的调用结果。

    Args:
        key_func: 自定义缓存键生成函数，接收与被装饰函数相同的参数。
                  不指定时使用所有位置参数的哈希作为键。
        ttl: 缓存有效期（秒），None 使用默认值 300s。

    Returns:
        装饰器。装饰后的函数支持 cache=False 参数来跳过缓存，
        并可通过 __wrapped__ 访问原始函数。

    Usage:
        @tool_cache(key_func=lambda query, engine: f"{query}:{engine}", ttl=600)
        def my_tool(query: str, engine: str = "baidu") -> str:
            ...

        # Skip cache for this call:
        my_tool("query", cache=False)
    """
    _ttl = ttl if ttl is not None else 300.0

    def decorator(fn: Callable) -> Callable:
        cache_instance = LLMCache(max_size=128, ttl=_ttl)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            skip_cache = kwargs.pop("cache", True)
            if not skip_cache:
                return fn(*args, **kwargs)

            if key_func:
                cache_key = fn.__name__ + ":" + key_func(*args, **kwargs)
            else:
                key_parts = [fn.__name__]
                key_parts.extend(str(a) for a in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = ":".join(key_parts)

            cached = cache_instance.get(cache_key)
            if cached is not None:
                return cached

            result = fn(*args, **kwargs)
            if result:
                cache_instance.set(cache_key, result)
            return result

        wrapper._cache = cache_instance
        wrapper.__wrapped__ = fn
        return wrapper

    return decorator