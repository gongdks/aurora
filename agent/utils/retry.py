"""LLM 调用超时和重试工具。

为所有 LLM 调用提供统一的超时控制和指数退避重试。
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

from agent.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

_shared_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="llm_timeout")


def shutdown_executor() -> None:
    """Shut down the shared thread pool on application exit."""
    _shared_executor.shutdown(wait=False)


class CancelledError(Exception):
    """Raised when a long-running operation is cancelled by the user."""
    pass


def _retry_call(
    fn: Callable[..., T],
    *args: Any,
    timeout: float,
    max_retries: int,
    retry_delay: float = 1.0,
    cancel_event: threading.Event | None = None,
    retryable_exceptions: tuple[type[BaseException], ...] = (Exception,),
    **kwargs: Any,
) -> T:
    """Core retry+timeout logic shared by all LLM-call helpers.

    Args:
        fn: Callable to invoke
        *args: Positional args for fn
        timeout: Per-attempt timeout in seconds
        max_retries: Maximum number of retries after the first attempt
        retry_delay: Base delay for exponential backoff (delay * 2**attempt)
        cancel_event: If set, raises CancelledError immediately
        retryable_exceptions: Exception types that trigger a retry
        **kwargs: Keyword args for fn

    Returns:
        The return value of fn

    Raises:
        CancelledError: If cancel_event is set
        The last exception encountered if all retries are exhausted
    """
    last_exception: BaseException | None = None

    for attempt in range(max_retries + 1):
        if cancel_event and cancel_event.is_set():
            raise CancelledError("LLM 调用已被用户取消")

        try:
            future = _shared_executor.submit(fn, *args, **kwargs)
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as exc:
            last_exception = exc
            if attempt < max_retries:
                delay = retry_delay * (2 ** attempt)
                logger.warning(
                    "调用超时 (第 %d/%d 次)，%.1fs 后重试...",
                    attempt + 1, max_retries + 1, delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "调用超时 (已达最大重试次数 %d)", max_retries + 1,
                )
        except retryable_exceptions as exc:
            last_exception = exc
            if attempt < max_retries:
                delay = retry_delay * (2 ** attempt)
                logger.warning(
                    "调用失败 (第 %d/%d 次): %s，%.1fs 后重试...",
                    attempt + 1, max_retries + 1, exc, delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "调用失败 (已达最大重试次数 %d): %s",
                    max_retries + 1, exc,
                )
        except CancelledError:
            raise
        except Exception as exc:
            logger.error("调用发生不可重试的异常: %s", exc)
            raise

    raise last_exception  # type: ignore[misc]


def with_timeout_retry(
    fn: Callable[..., T],
    *,
    timeout: float | None = None,
    max_retries: int | None = None,
    retry_delay: float = 1.0,
    retryable_exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[..., T]:
    """包装函数，添加超时和重试机制。

    Args:
        fn: 要包装的函数
        timeout: 超时秒数，None 使用默认值
        max_retries: 最大重试次数，None 使用默认值
        retry_delay: 重试间隔基数（秒），使用指数退避
        retryable_exceptions: 可重试的异常类型

    Returns:
        包装后的函数
    """
    _timeout = timeout if timeout is not None else settings.MAX_EXECUTION_TIME_SEC
    _max_retries = max_retries if max_retries is not None else settings.MAX_RETRIES

    def wrapper(*args: Any, **kwargs: Any) -> T:
        return _retry_call(
            fn, *args,
            timeout=_timeout,
            max_retries=_max_retries,
            retry_delay=retry_delay,
            retryable_exceptions=retryable_exceptions,
            **kwargs,
        )

    return wrapper


def llm_invoke_with_guard(
    llm: Any,
    messages: list,
    *,
    timeout: float | None = None,
    max_retries: int | None = None,
    cancel_event: threading.Event | None = None,
) -> Any:
    """安全调用 LLM.invoke()，带超时、重试和取消支持。

    Args:
        llm: LangChain BaseChatModel 实例
        messages: 消息列表
        timeout: 超时秒数
        max_retries: 最大重试次数
        cancel_event: 取消事件，设置后立即中止重试并抛出 CancelledError

    Returns:
        LLM 响应对象

    Raises:
        CancelledError: cancel_event 被设置
    """
    _timeout = timeout if timeout is not None else settings.MAX_EXECUTION_TIME_SEC
    _max_retries = max_retries if max_retries is not None else settings.MAX_RETRIES

    def _invoke(llm: Any, messages: list) -> Any:
        try:
            return llm.invoke(messages, timeout=_timeout)
        except TypeError:
            return llm.invoke(messages)

    return _retry_call(
        _invoke, llm, messages,
        timeout=_timeout,
        max_retries=_max_retries,
        cancel_event=cancel_event,
    )