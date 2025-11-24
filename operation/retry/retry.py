"""
Retry mechanism with exponential backoff for failed operations.
"""

import time
import random
import functools
from typing import Callable, Type, Tuple, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"


def calculate_backoff(
    attempt: int,
    initial_delay: float,
    max_delay: float,
    multiplier: float,
    jitter: bool = True,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
) -> float:
    """
    Calculate backoff delay for given attempt.
    
    Args:
        attempt: Current attempt number (1-indexed)
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        multiplier: Backoff multiplier
        jitter: Add random jitter to prevent thundering herd
        strategy: Retry strategy (exponential, linear, fixed)
    
    Returns:
        Delay in seconds
    """
    if strategy == RetryStrategy.EXPONENTIAL:
        delay = initial_delay * (multiplier ** (attempt - 1))
    elif strategy == RetryStrategy.LINEAR:
        delay = initial_delay * attempt
    else:  # FIXED
        delay = initial_delay
    
    # Cap at max_delay
    delay = min(delay, max_delay)
    
    # Add jitter (random variation up to 25% of delay)
    if jitter:
        jitter_amount = delay * 0.25 * random.random()
        delay = delay + jitter_amount
    
    return delay


def retry_with_backoff(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    multiplier: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable] = None,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        multiplier: Backoff multiplier
        jitter: Add random jitter to prevent thundering herd
        retryable_exceptions: Tuple of exception types to retry
        on_retry: Optional callback function called on each retry
        strategy: Retry strategy (exponential, linear, fixed)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts:
                        delay = calculate_backoff(
                            attempt, initial_delay, max_delay, multiplier, jitter, strategy
                        )
                        
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt}/{max_attempts}): {str(e)}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        
                        if on_retry:
                            on_retry(attempt, delay, e)
                        
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {str(e)}"
                        )
            
            # All attempts failed
            raise last_exception
        
        return wrapper
    return decorator

