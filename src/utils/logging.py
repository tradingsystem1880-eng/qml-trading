"""
Logging Configuration for QML Trading System
=============================================
Centralized logging setup using loguru with file rotation,
structured output, and level-based filtering.
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from config.settings import settings


def setup_logging(
    log_level: Optional[str] = None,
    log_dir: str = "./logs",
    enable_file: bool = True,
    enable_json: bool = False
) -> None:
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files
        enable_file: Enable file logging
        enable_json: Enable JSON formatted logs
    """
    # Remove default handler
    logger.remove()
    
    # Get log level from settings or parameter
    level = log_level or settings.log_level
    
    # Console handler with colored output
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=True,
        diagnose=settings.debug
    )
    
    if enable_file:
        # Create log directory
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Main application log (rotated daily)
        logger.add(
            log_path / "qml_system_{time:YYYY-MM-DD}.log",
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="00:00",  # Rotate at midnight
            retention="30 days",
            compression="gz",
            backtrace=True,
            diagnose=False
        )
        
        # Error log (separate file for errors)
        logger.add(
            log_path / "errors_{time:YYYY-MM-DD}.log",
            level="ERROR",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}\n{exception}",
            rotation="00:00",
            retention="90 days",
            compression="gz",
            backtrace=True,
            diagnose=True
        )
        
        # Trading/Pattern log (for audit)
        logger.add(
            log_path / "patterns_{time:YYYY-MM-DD}.log",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
            filter=lambda record: "pattern" in record["extra"],
            rotation="00:00",
            retention="365 days"
        )
    
    if enable_json:
        # JSON formatted logs for log aggregation systems
        logger.add(
            log_path / "qml_system_{time:YYYY-MM-DD}.json",
            level=level,
            format="{message}",
            serialize=True,
            rotation="00:00",
            retention="30 days",
            compression="gz"
        )
    
    logger.info(f"Logging initialized at {level} level")


def get_pattern_logger():
    """Get a logger bound with pattern context for trading logs."""
    return logger.bind(pattern=True)

