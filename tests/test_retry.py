"""Tests for agent.utils.retry — timeout and retry utilities."""

import threading
import time

import pytest

from agent.utils.retry import (
    CancelledError,
    with_timeout_retry,
)


class TestWithTimeoutRetry:
    """Tests for with_timeout_retry()."""

    def test_successful_call(self):
        call_count = 0

        def _fn(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        fn = with_timeout_retry(_fn)
        result = fn(21)
        assert result == 42
        assert call_count == 1

    def test_retry_on_retryable_exception(self):
        call_count = 0

        def _fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("temporary error")
            return "success"

        fn = with_timeout_retry(_fn, max_retries=2, retry_delay=0.05)
        result = fn()
        assert result == "success"
        assert call_count == 3

    def test_max_retries_exhausted(self):
        call_count = 0

        def _fn() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("persistent error")

        fn = with_timeout_retry(_fn, retry_delay=0.05)
        with pytest.raises(ValueError):
            fn()
        assert call_count == 2

    def test_non_retryable_exception_not_retried(self):
        call_count = 0

        def _fn() -> str:
            nonlocal call_count
            call_count += 1
            raise KeyboardInterrupt()

        fn = with_timeout_retry(_fn)
        with pytest.raises(KeyboardInterrupt):
            fn()
        assert call_count == 1

    def test_custom_retryable_exceptions(self):
        call_count = 0

        def _fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise KeyError("missing key")
            return "found"

        fn = with_timeout_retry(_fn, retryable_exceptions=(KeyError,), retry_delay=0.05)
        result = fn()
        assert result == "found"
        assert call_count == 2

    def test_custom_max_retries(self):
        call_count = 0

        def _fn() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("fail")

        fn = with_timeout_retry(_fn, max_retries=3, retry_delay=0.05)
        with pytest.raises(ValueError):
            fn()
        assert call_count == 4

    def test_timeout_func(self):
        def _slow_fn() -> str:
            time.sleep(0.5)
            return "too late"

        fn = with_timeout_retry(_slow_fn, timeout=0.05, max_retries=0)
        with pytest.raises(TimeoutError):
            fn()


class TestCancelledError:
    """Tests for CancelledError."""

    def test_is_exception(self):
        err = CancelledError("test")
        assert isinstance(err, Exception)

    def test_message(self):
        err = CancelledError("用户取消了操作")
        assert str(err) == "用户取消了操作"


class TestCancelPropagation:
    """Tests for cancel_event propagation through retry."""

    def test_cancel_before_retry_stops(self):
        """Cancel event set during retry loop should raise CancelledError."""
        from agent.utils.retry import llm_invoke_with_guard

        cancel = threading.Event()
        cancel.set()  # already cancelled

        class FakeLLM:
            def invoke(self, messages, timeout=None):
                return type("Response", (), {"content": "never called"})()

        llm = FakeLLM()
        with pytest.raises(CancelledError, match="已被用户取消"):
            llm_invoke_with_guard(llm, [{"role": "user", "content": "hi"}], cancel_event=cancel)