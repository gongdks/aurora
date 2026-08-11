"""Tests for agent.utils.cache — LLM response caching."""

import time

import pytest

from agent.utils.cache import LLMCache


class TestLLMCache:
    """Tests for LLMCache."""

    def test_set_and_get(self):
        cache = LLMCache(max_size=10, ttl=300)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_missing_key(self):
        cache = LLMCache(max_size=10, ttl=300)
        assert cache.get("nonexistent") is None

    def test_ttl_expiry(self):
        cache = LLMCache(max_size=10, ttl=0.01)
        cache.set("key1", "value1")
        time.sleep(0.02)
        assert cache.get("key1") is None

    def test_make_key_deterministic(self):
        cache = LLMCache()
        key1 = cache.make_key("model", 0.1, "prompt")
        key2 = cache.make_key("model", 0.1, "prompt")
        assert key1 == key2

    def test_make_key_different_inputs(self):
        cache = LLMCache()
        key1 = cache.make_key("a", 1)
        key2 = cache.make_key("a", 2)
        assert key1 != key2

    def test_lru_eviction(self):
        cache = LLMCache(max_size=3, ttl=300)
        for i in range(5):
            cache.set(f"key{i}", f"value{i}")
        # key0 and key1 should be evicted
        assert cache.get("key0") is None
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"

    def test_lru_reorder_on_get(self):
        cache = LLMCache(max_size=3, ttl=300)
        cache.set("key0", "v0")
        cache.set("key1", "v1")
        cache.set("key2", "v2")
        # Access key0 to move it to front
        cache.get("key0")
        cache.set("key3", "v3")
        # key1 should be evicted (least recently used), key0 still there
        assert cache.get("key0") == "v0"
        assert cache.get("key1") is None
        assert cache.get("key2") == "v2"
        assert cache.get("key3") == "v3"

    def test_stats_tracking(self):
        cache = LLMCache(max_size=10, ttl=300)
        assert cache.stats["hits"] == 0
        assert cache.stats["misses"] == 0

        cache.get("key1")  # miss
        cache.set("key1", "val1")
        cache.get("key1")  # hit
        cache.get("key2")  # miss

        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["hit_rate"] == round(1 / 3, 3)

    def test_clear(self):
        cache = LLMCache(max_size=10, ttl=300)
        cache.set("key1", "val1")
        cache.clear()
        assert cache.get("key1") is None
        assert cache.stats["hits"] == 0


class TestGlobalCacheAccess:
    """Tests for global cache singleton."""

    def test_get_cache_returns_llm_cache(self):
        from agent.utils.cache import get_cache
        cache = get_cache()
        assert isinstance(cache, LLMCache)
