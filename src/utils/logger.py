"""
Logging Module
Centralized logging configuration with colored output and CloudWatch integration.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional
from contextlib import contextmanager
import time


# Color codes for terminal output
class Colors:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"


class ColoredFormatter(logging.Formatter):
    """Formatter with colored output for different log levels."""
    
    COLORS = {
        logging.DEBUG: Colors.CYAN,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.MAGENTA,
    }
    
    def format(self, record):
        color = self.COLORS.get(record.levelno, Colors.WHITE)
        record.levelname = f"{color}{record.levelname}{Colors.RESET}"
        record.name = f"{Colors.BLUE}{record.name}{Colors.RESET}"
        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        log_format: Optional custom log format
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    format_str = log_format or "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ColoredFormatter(format_str))
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(format_str))
        root_logger.addHandler(file_handler)
    
    # Reduce noise from third-party libraries
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class PipelineLogger:
    """
    Context manager for logging pipeline stages with timing.
    """
    
    def __init__(self, stage_name: str, logger: logging.Logger = None):
        self.stage_name = stage_name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"{'─' * 40}")
        self.logger.info(f"Starting: {self.stage_name}")
        self.logger.info(f"{'─' * 40}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        if exc_type is None:
            self.logger.info(f"✓ Completed: {self.stage_name} ({duration:.2f}s)")
        else:
            self.logger.error(f"✗ Failed: {self.stage_name} ({duration:.2f}s)")
            self.logger.error(f"  Error: {exc_val}")
        
        return False  # Don't suppress exceptions


@contextmanager
def log_stage(name: str, logger: logging.Logger = None):
    """
    Context manager for logging a stage with timing.
    
    Args:
        name: Stage name
        logger: Logger instance
        
    Yields:
        None
    """
    _logger = logger or logging.getLogger(__name__)
    start = time.time()
    
    _logger.info(f"Starting: {name}")
    
    try:
        yield
    except Exception as e:
        duration = time.time() - start
        _logger.error(f"Failed: {name} ({duration:.2f}s) - {e}")
        raise
    else:
        duration = time.time() - start
        _logger.info(f"Completed: {name} ({duration:.2f}s)")
