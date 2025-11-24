"""
Logging configuration for the robo-advisor system.
Provides structured logging with correlation ID support.
"""

import logging
import sys
import os
from typing import Optional
from datetime import datetime
import uuid
from contextvars import ContextVar

# Correlation ID for request tracking
correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)

# Global logger configuration flag
_logging_configured = False


class CorrelationFilter(logging.Filter):
    """Add correlation ID to log records"""
    def filter(self, record):
        record.correlation_id = correlation_id.get() or 'N/A'
        return True


class StructuredFormatter(logging.Formatter):
    """Structured log formatter with correlation ID"""
    def format(self, record):
        # Format: [TIMESTAMP] [LEVEL] [CORRELATION_ID] [MODULE] MESSAGE
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        return f"[{timestamp}] [{record.levelname}] [{record.correlation_id}] [{record.name}] {record.getMessage()}"


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    enable_json: bool = False
) -> None:
    """
    Configure application-wide logging.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        enable_json: Enable JSON formatting for production (not implemented yet)
    """
    global _logging_configured
    
    if _logging_configured:
        return
    
    # Get log level
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = StructuredFormatter()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(CorrelationFilter())
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(CorrelationFilter())
        root_logger.addHandler(file_handler)
    
    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with correlation ID support.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def set_correlation_id(cid: Optional[str] = None) -> str:
    """
    Set correlation ID for current context.
    Generates new ID if None provided.
    
    Args:
        cid: Optional correlation ID (generates new if None)
    
    Returns:
        Correlation ID string
    """
    if cid is None:
        cid = str(uuid.uuid4())
    correlation_id.set(cid)
    return cid


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID"""
    return correlation_id.get()


def log_function_call(func):
    """Decorator to log function entry/exit"""
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Entering {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Exiting {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
            raise
    return wrapper


def log_performance(func):
    """Decorator to log function execution time"""
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"{func.__name__} executed in {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{func.__name__} failed after {elapsed:.3f}s: {e}", exc_info=True)
            raise
    return wrapper

