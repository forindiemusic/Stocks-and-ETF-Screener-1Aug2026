"""
Retry utility with exponential backoff for API calls.
Used by etf_and_stock_agent.py, backtest.py, and scoring.py to handle
rate-limiting and transient failures from yfinance / requests.
"""

import time
import random
import logging
from functools import wraps
from typing import Callable, TypeVar, Any, Optional

logger = logging.getLogger(__name__)

T = TypeVar('T')

# Default retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 2.0       # seconds
DEFAULT_BACKOFF_FACTOR = 2.0   # multiplicative
DEFAULT_MAX_DELAY = 60.0       # seconds
DEFAULT_JITTER = True          # add random jitter to avoid thundering herd

# Exceptions that are worth retrying (transient failures)
RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,
)


def _is_retryable(exception: Exception) -> bool:
    """Check if an exception is likely transient and worth retrying."""
    # Check by type
    if isinstance(exception, RETRYABLE_EXCEPTIONS):
        return True

    # Check by message content for common rate-limit / transient patterns
    msg = str(exception).lower()
    retryable_keywords = [
        'rate limit', 'too many requests', '429',
        'timeout', 'timed out',
        'connection reset', 'connection refused',
        'temporary failure', 'service unavailable', '503',
        'internal server error', '500',
        'bad gateway', '502', 'gateway timeout', '504',
        'no data found', 'try again',
    ]
    return any(kw in msg for kw in retryable_keywords)


def retry(
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    max_delay: float = DEFAULT_MAX_DELAY,
    jitter: bool = DEFAULT_JITTER,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator that retries a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds before first retry.
        backoff_factor: Multiplier for each subsequent delay.
        max_delay: Maximum delay cap in seconds.
        jitter: If True, add ±25% random jitter to delay.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None
            delay = base_delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if attempt == max_retries or not _is_retryable(e):
                        break

                    # Apply jitter: ±25% randomness
                    actual_delay = delay
                    if jitter:
                        actual_delay = delay * (0.75 + random.random() * 0.5)

                    actual_delay = min(actual_delay, max_delay)

                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} for {func.__name__}: "
                        f"{type(e).__name__}: {e}. Waiting {actual_delay:.1f}s..."
                    )
                    time.sleep(actual_delay)
                    delay *= backoff_factor

            # All retries exhausted
            logger.error(
                f"All {max_retries} retries exhausted for {func.__name__}: "
                f"{type(last_exception).__name__}: {last_exception}"
            )
            assert last_exception is not None
            raise last_exception

        return wrapper
    return decorator


def retry_call(
    func: Callable[..., T],
    *args: Any,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    max_delay: float = DEFAULT_MAX_DELAY,
    jitter: bool = DEFAULT_JITTER,
    **kwargs: Any,
) -> T:
    """
    Call a function with retry logic (non-decorator version).

    Useful for one-off calls where a decorator is impractical,
    e.g., calling third-party library methods directly.

    Args:
        func: The callable to invoke.
        *args: Positional arguments for func.
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds before first retry.
        backoff_factor: Multiplier for each subsequent delay.
        max_delay: Maximum delay cap in seconds.
        jitter: If True, add ±25% random jitter to delay.
        **kwargs: Keyword arguments for func.

    Returns:
        The return value of func(*args, **kwargs).

    Raises:
        The last exception after all retries are exhausted.
    """
    last_exception: Optional[Exception] = None
    delay = base_delay

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e

            if attempt == max_retries or not _is_retryable(e):
                break

            actual_delay = delay
            if jitter:
                actual_delay = delay * (0.75 + random.random() * 0.5)
            actual_delay = min(actual_delay, max_delay)

            logger.warning(
                f"Retry {attempt + 1}/{max_retries} for {getattr(func, '__name__', str(func))}: "
                f"{type(e).__name__}: {e}. Waiting {actual_delay:.1f}s..."
            )
            time.sleep(actual_delay)
            delay *= backoff_factor

    logger.error(
        f"All {max_retries} retries exhausted for {getattr(func, '__name__', str(func))}: "
        f"{type(last_exception).__name__}: {last_exception}"
    )
    assert last_exception is not None
    raise last_exception