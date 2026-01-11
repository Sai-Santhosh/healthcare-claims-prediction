"""
Helper Utilities Module
Provides various utility functions for the project.
"""

import os
import time
import functools
import psutil
from datetime import datetime
from typing import Callable, Any, Optional
from pathlib import Path

from .logger import get_logger

logger = get_logger(__name__)


def timer(func: Callable) -> Callable:
    """
    Decorator to measure and log function execution time.
    
    Args:
        func: Function to wrap
    
    Returns:
        Wrapped function with timing
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_time = time.perf_counter() - start_time
        
        # Format time appropriately
        if elapsed_time < 60:
            time_str = f"{elapsed_time:.2f} seconds"
        elif elapsed_time < 3600:
            minutes, seconds = divmod(elapsed_time, 60)
            time_str = f"{int(minutes)}m {seconds:.2f}s"
        else:
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            time_str = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
        
        logger.info(f"⏱️  {func.__name__} completed in {time_str}")
        return result
    
    return wrapper


def memory_usage() -> dict:
    """
    Get current memory usage statistics.
    
    Returns:
        Dictionary with memory statistics
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    return {
        'rss': format_bytes(mem_info.rss),
        'vms': format_bytes(mem_info.vms),
        'percent': f"{process.memory_percent():.2f}%",
        'rss_bytes': mem_info.rss,
    }


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes to human-readable string.
    
    Args:
        bytes_value: Size in bytes
    
    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(bytes_value) < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def ensure_dir(path: str) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
    
    Returns:
        Path object
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_timestamp() -> str:
    """Get current timestamp string for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: The numerator
        denominator: The denominator
        default: Value to return if denominator is zero
    
    Returns:
        Result of division or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator


class ProgressTracker:
    """
    Simple progress tracker for long-running operations.
    """
    
    def __init__(self, total: int, description: str = "Processing", log_interval: int = 10):
        self.total = total
        self.description = description
        self.log_interval = log_interval
        self.current = 0
        self.start_time = time.perf_counter()
        self.last_log_percent = 0
    
    def update(self, n: int = 1) -> None:
        """Update progress by n items."""
        self.current += n
        percent = (self.current / self.total) * 100
        
        # Log at intervals
        if percent - self.last_log_percent >= self.log_interval or self.current == self.total:
            elapsed = time.perf_counter() - self.start_time
            items_per_sec = self.current / elapsed if elapsed > 0 else 0
            remaining = (self.total - self.current) / items_per_sec if items_per_sec > 0 else 0
            
            logger.info(
                f"{self.description}: {percent:.1f}% "
                f"({self.current:,}/{self.total:,}) "
                f"- {items_per_sec:.0f} items/s "
                f"- ETA: {remaining:.0f}s"
            )
            self.last_log_percent = percent
    
    def finish(self) -> None:
        """Mark progress as complete."""
        elapsed = time.perf_counter() - self.start_time
        logger.info(
            f"{self.description}: Complete! "
            f"Processed {self.total:,} items in {elapsed:.2f}s"
        )


def validate_dataframe(df, required_columns: list, df_name: str = "DataFrame") -> bool:
    """
    Validate that a DataFrame has required columns.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: List of required column names
        df_name: Name for error messages
    
    Returns:
        True if valid
    
    Raises:
        ValueError: If required columns are missing
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"{df_name} is missing required columns: {missing}"
        )
    return True
