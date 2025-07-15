"""
Logging configuration for DuoPet AI Service

This module provides structured logging with support for both
console and file output, with automatic log rotation.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional
import json

from common.config import get_settings

# Get settings
settings = get_settings()


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ["name", "msg", "args", "created", "filename", "funcName",
                          "levelname", "levelno", "lineno", "module", "exc_info",
                          "exc_text", "stack_info", "pathname", "processName",
                          "process", "threadName", "thread", "getMessage"]:
                log_data[key] = value
        
        return json.dumps(log_data, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors"""
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        
        # Format the message
        message = super().format(record)
        
        # Reset levelname for other handlers
        record.levelname = levelname
        
        return message


def setup_logger(
    name: str,
    level: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with console and file handlers
    
    Args:
        name: Logger name
        level: Log level (defaults to settings)
        log_file: Log file path (defaults to settings)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Set log level
    log_level = getattr(logging, level or settings.LOG_LEVEL)
    logger.setLevel(log_level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Use colored formatter for console in development
    if settings.is_development:
        console_format = ColoredFormatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        console_format = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (only in production or if explicitly specified)
    if settings.is_production or log_file:
        file_path = log_file or settings.LOG_FILE_PATH
        
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Parse max size (e.g., "100MB" -> 100 * 1024 * 1024)
        max_size = settings.LOG_MAX_SIZE
        if max_size.endswith("MB"):
            max_bytes = int(max_size[:-2]) * 1024 * 1024
        elif max_size.endswith("GB"):
            max_bytes = int(max_size[:-2]) * 1024 * 1024 * 1024
        else:
            max_bytes = int(max_size)
        
        # Create rotating file handler
        file_handler = RotatingFileHandler(
            filename=file_path,
            maxBytes=max_bytes,
            backupCount=settings.LOG_BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        
        # Use JSON formatter for file output
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)
    
    return logger


# Create default loggers
def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Logger instance
    """
    return setup_logger(name)


# Convenience functions for logging
def log_api_request(logger: logging.Logger, method: str, path: str, **kwargs):
    """Log API request details"""
    logger.info(
        f"API Request: {method} {path}",
        extra={
            "api_method": method,
            "api_path": path,
            **kwargs
        }
    )


def log_api_response(logger: logging.Logger, status_code: int, duration_ms: float, **kwargs):
    """Log API response details"""
    logger.info(
        f"API Response: {status_code} ({duration_ms:.2f}ms)",
        extra={
            "api_status_code": status_code,
            "api_duration_ms": duration_ms,
            **kwargs
        }
    )


def log_model_inference(logger: logging.Logger, model_name: str, duration_ms: float, **kwargs):
    """Log model inference details"""
    logger.info(
        f"Model Inference: {model_name} ({duration_ms:.2f}ms)",
        extra={
            "model_name": model_name,
            "inference_duration_ms": duration_ms,
            **kwargs
        }
    )


def log_error_with_context(logger: logging.Logger, error: Exception, context: dict):
    """Log error with additional context"""
    logger.error(
        f"Error: {type(error).__name__}: {str(error)}",
        exc_info=True,
        extra={"error_context": context}
    )


# Configure root logger
def configure_root_logger():
    """Configure the root logger for the application"""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL))
    
    # Remove default handlers
    root_logger.handlers = []
    
    # Add our configured handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    root_logger.addHandler(handler)


# Configure root logger on module import
configure_root_logger()