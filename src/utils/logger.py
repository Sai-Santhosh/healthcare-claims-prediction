"""
Logging Utilities Module
Provides centralized logging configuration and utilities.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for console."""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> None:
    """
    Set up logging configuration for the project.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        log_format: Custom log format string (optional)
    """
    if log_format is None:
        log_format = "%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s"
    
    # Create logs directory if needed
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = ColoredFormatter(log_format)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (no colors)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Name of the logger (usually __name__)
    
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class PipelineLogger:
    """
    Context manager for logging pipeline stages.
    Provides automatic timing and status logging.
    """
    
    def __init__(self, stage_name: str, logger: Optional[logging.Logger] = None):
        self.stage_name = stage_name
        self.logger = logger or get_logger(__name__)
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Starting: {self.stage_name}")
        self.logger.info(f"{'='*60}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = datetime.now() - self.start_time
        
        if exc_type is None:
            self.logger.info(f"Completed: {self.stage_name}")
            self.logger.info(f"Duration: {elapsed}")
            self.logger.info(f"{'='*60}\n")
        else:
            self.logger.error(f"Failed: {self.stage_name}")
            self.logger.error(f"Error: {exc_val}")
            self.logger.error(f"Duration: {elapsed}")
            self.logger.info(f"{'='*60}\n")
        
        return False  # Don't suppress exceptions


def log_dataframe_info(df, logger: logging.Logger, name: str = "DataFrame") -> None:
    """
    Log information about a pandas DataFrame.
    
    Args:
        df: pandas DataFrame to log info about
        logger: Logger instance to use
        name: Name to identify the DataFrame in logs
    """
    logger.info(f"{name} Info:")
    logger.info(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    logger.info(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    logger.info(f"  Columns: {list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}")
    
    # Missing values summary
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        logger.info(f"  Columns with missing values: {len(missing_cols)}")
