"""Rate limiter — simple token-bucket rate limiting for LLM calls."""

from __future__ import annotations

import time
from collections import deque
from threading import Lock


class RateLimiter:
    """Token-bucket rate limiter.

    Limits the number of calls within a time window.
    Thread-safe for concurrent access.

    Usage:
        limiter = RateLimiter(max_calls=10, period=1.0)
        limiter.acquire()  # blocks if rate exceeded
        # ... make API call ...
    """

    def __init__(self, max_calls: int = 10, period: float = 1.0) -> None:
        self._max_calls = max_calls
        self._period = period
        self._timestamps: deque[float] = deque()
        self._lock = Lock()

    def acquire(self) -> None:
        """Acquire a rate limit slot, blocking if necessary."""
        while True:
            with self._lock:
                now = time.time()
                self._prune(now)

                if len(self._timestamps) < self._max_calls:
                    self._timestamps.append(now)
                    return

                wait_time = self._period - (now - self._timestamps[0])

            if wait_time > 0:
                time.sleep(wait_time)

    def try_acquire(self) -> bool:
        """Try to acquire a slot without blocking. Returns True if successful."""
        with self._lock:
            now = time.time()
            self._prune(now)

            if len(self._timestamps) < self._max_calls:
                self._timestamps.append(now)
                return True

            return False

    def _prune(self, now: float) -> None:
        """Remove timestamps older than the current period."""
        cutoff = now - self._period
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    @property
    def current_count(self) -> int:
        """Return the number of calls in the current window."""
        with self._lock:
            self._prune(time.time())
            return len(self._timestamps)

    def reset(self) -> None:
        """Reset the rate limiter."""
        with self._lock:
            self._timestamps.clear()