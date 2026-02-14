"""
Helper Utilities
Common utility functions for the data engineering pipeline.
"""

import os
import time
import psutil
import functools
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def timer(func: Callable) -> Callable:
    """
    Decorator to time function execution.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logger.debug(f"{func.__name__} executed in {duration:.4f} seconds")
        return result
    return wrapper


def memory_usage() -> Dict[str, str]:
    """
    Get current memory usage.
    
    Returns:
        Dictionary with memory metrics
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    return {
        "rss": format_bytes(mem_info.rss),
        "vms": format_bytes(mem_info.vms),
        "rss_bytes": mem_info.rss,
        "vms_bytes": mem_info.vms,
        "percent": f"{process.memory_percent():.1f}%"
    }


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes to human-readable string.
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(bytes_value) < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def ensure_dir(path: str) -> str:
    """
    Ensure directory exists, create if not.
    
    Args:
        path: Directory path
        
    Returns:
        Path string
    """
    os.makedirs(path, exist_ok=True)
    return path


def get_file_size(path: str) -> Optional[int]:
    """
    Get file size in bytes.
    
    Args:
        path: File path
        
    Returns:
        Size in bytes or None if file doesn't exist
    """
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def get_timestamp() -> str:
    """
    Get current timestamp string.
    
    Returns:
        Timestamp in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')


class ProgressTracker:
    """
    Track progress of long-running operations.
    """
    
    def __init__(
        self,
        total: int,
        description: str = "Processing",
        log_every: int = 10
    ):
        self.total = total
        self.description = description
        self.log_every = log_every
        self.current = 0
        self.start_time = time.time()
    
    def update(self, n: int = 1) -> None:
        """Update progress by n items."""
        self.current += n
        
        percent = (self.current / self.total) * 100
        
        if self.current % self.log_every == 0 or self.current == self.total:
            elapsed = time.time() - self.start_time
            rate = self.current / elapsed if elapsed > 0 else 0
            eta = (self.total - self.current) / rate if rate > 0 else 0
            
            logger.info(
                f"{self.description}: {self.current:,}/{self.total:,} "
                f"({percent:.1f}%) - {rate:.1f}/s - ETA: {eta:.0f}s"
            )
    
    def finish(self) -> Dict[str, Any]:
        """Mark progress as complete."""
        elapsed = time.time() - self.start_time
        return {
            "total": self.total,
            "processed": self.current,
            "duration_seconds": elapsed,
            "rate": self.current / elapsed if elapsed > 0 else 0
        }


class RetryHandler:
    """
    Handle retries with exponential backoff.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with retries.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries - 1:
                    delay = min(
                        self.base_delay * (self.exponential_base ** attempt),
                        self.max_delay
                    )
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
        
        raise last_exception


def chunk_list(lst: list, chunk_size: int):
    """
    Split list into chunks.
    
    Args:
        lst: List to split
        chunk_size: Size of each chunk
        
    Yields:
        List chunks
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
    """
    Flatten nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator between keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if denominator is zero
        
    Returns:
        Division result or default
    """
    if denominator == 0:
        return default
    return numerator / denominator
