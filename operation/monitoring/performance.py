"""
Performance monitoring utilities.
"""

import time
import functools
from typing import Callable, Any
from contextlib import contextmanager

from operation.monitoring.metrics import get_metrics_registry


def track_performance(func: Callable) -> Callable:
    """
    Decorator to track function execution time.
    
    Args:
        func: Function to track
    
    Returns:
        Wrapped function with performance tracking
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        registry = get_metrics_registry()
        timer = registry.timer(f"{func.__module__}.{func.__name__}")
        
        start = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start
            timer.record(duration)
    
    return wrapper


@contextmanager
def performance_timer(name: str):
    """
    Context manager for manual performance timing.
    
    Usage:
        with performance_timer("my_operation"):
            # do something
    """
    registry = get_metrics_registry()
    timer = registry.timer(name)
    
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        timer.record(duration)

