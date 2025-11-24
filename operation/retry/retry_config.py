"""
Retry configuration for different operation types.
"""

from dataclasses import dataclass
from typing import Tuple, Type
import os

from .retry import RetryStrategy


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_attempts: int
    initial_delay: float
    max_delay: float
    multiplier: float
    jitter: bool
    retryable_exceptions: Tuple[Type[Exception], ...]
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL


# OpenAI API retry configuration
OPENAI_RETRY_CONFIG = RetryConfig(
    max_attempts=int(os.getenv("OPENAI_MAX_RETRIES", "3")),
    initial_delay=float(os.getenv("OPENAI_INITIAL_DELAY", "1.0")),
    max_delay=float(os.getenv("OPENAI_MAX_DELAY", "60.0")),
    multiplier=2.0,
    jitter=True,
    retryable_exceptions=(
        Exception,  # Will catch rate limits, timeouts, etc.
    ),
    strategy=RetryStrategy.EXPONENTIAL
)

# Yahoo Finance retry configuration
YFINANCE_RETRY_CONFIG = RetryConfig(
    max_attempts=int(os.getenv("YFINANCE_MAX_RETRIES", "2")),
    initial_delay=float(os.getenv("YFINANCE_INITIAL_DELAY", "0.5")),
    max_delay=float(os.getenv("YFINANCE_MAX_DELAY", "10.0")),
    multiplier=2.0,
    jitter=True,
    retryable_exceptions=(
        ConnectionError,
        TimeoutError,
        Exception,  # yfinance may raise various exceptions
    ),
    strategy=RetryStrategy.EXPONENTIAL
)

# File I/O retry configuration
FILE_RETRY_CONFIG = RetryConfig(
    max_attempts=int(os.getenv("FILE_MAX_RETRIES", "3")),
    initial_delay=float(os.getenv("FILE_INITIAL_DELAY", "0.1")),
    max_delay=float(os.getenv("FILE_MAX_DELAY", "5.0")),
    multiplier=1.5,
    jitter=False,
    retryable_exceptions=(
        IOError,
        PermissionError,
    ),
    strategy=RetryStrategy.EXPONENTIAL
)

