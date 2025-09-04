"""Enhanced LoggerManager for ABACUS-STRU-Analyser v2.0"""

import io
import logging
import logging.handlers
import multiprocessing as mp
import os
import sys
import traceback
from pathlib import Path
from typing import Optional, Tuple


from .common import Constants


class LoggerManager:
    """Centralized logger management with consistent formatting and configuration"""

    DEFAULT_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    # Use the same date format as defined in Constants
    DEFAULT_DATE_FORMAT = Constants.DEFAULT_DATE_FORMAT
    """Centralized logger management with consistent formatting and configuration"""

    DEFAULT_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    DEFAULT_DATE_FORMAT = "%H:%M:%S"

    @staticmethod
    def create_logger(
        name: str,
        level: int = logging.INFO,
        add_console: bool = True,
        log_file: Optional[str] = None,
        log_format: str = DEFAULT_FORMAT,
        date_format: str = DEFAULT_DATE_FORMAT,
    ) -> logging.Logger:
        """Create a standardized logger instance

        Args:
            name: Logger name (typically __name__ from calling module)
            level: Logging level (default: INFO)
            add_console: Whether to add console handler
            log_file: Optional log file path
            log_format: Custom format string
            date_format: Custom date format string

        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(log_format, datefmt=date_format)

        # Add console handler
        if add_console:
            console_stream = LoggerManager._get_console_stream()
            console_handler = logging.StreamHandler(console_stream)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # Add file handler if specified
        if log_file:
            LoggerManager._add_file_handler_internal(logger, log_file, formatter)

        return logger

    @staticmethod
    def create_logger_with_error_log(
        name: str,
        level: int = logging.INFO,
        add_console: bool = True,
        log_file: Optional[str] = None,
        error_log_file: Optional[str] = None,
        log_format: str = DEFAULT_FORMAT,
        date_format: str = DEFAULT_DATE_FORMAT,
    ) -> logging.Logger:
        """Create a logger with optional error log file

        Args:
            name: Logger name (typically __name__ from calling module)
            level: Logging level (default: INFO)
            add_console: Whether to add console handler
            log_file: Optional main log file path
            error_log_file: Optional error log file path
            log_format: Custom format string
            date_format: Custom date format string

        Returns:
            Configured logger instance
        """
        logger = LoggerManager.create_logger(
            name=name,
            level=level,
            add_console=add_console,
            log_file=log_file,
            log_format=log_format,
            date_format=date_format,
        )

        # Add error log handler if specified
        if error_log_file:
            LoggerManager.add_error_log_handler(
                logger=logger,
                error_log_file=error_log_file,
                log_format=log_format,
                date_format=date_format,
            )

        return logger

    @staticmethod
    def _add_file_handler_internal(
        logger: logging.Logger, log_file: str, formatter: logging.Formatter
    ) -> logging.FileHandler:
        """Internal method to add file handler"""
        # Ensure directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return file_handler

    @staticmethod
    def add_error_log_handler(
        logger: logging.Logger,
        error_log_file: str,
        log_format: str = DEFAULT_FORMAT,
        date_format: str = DEFAULT_DATE_FORMAT,
    ) -> logging.FileHandler:
        """Add dedicated error log handler to existing logger

        Args:
            logger: Logger instance to modify
            error_log_file: Path to error log file
            log_format: Format string for log messages
            date_format: Format string for timestamps

        Returns:
            Created error log file handler
        """
        formatter = logging.Formatter(log_format, datefmt=date_format)

        # Ensure directory exists
        error_log_path = Path(error_log_file)
        error_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Create error file handler with append mode
        error_handler = logging.FileHandler(error_log_file, mode="a", encoding="utf-8")
        error_handler.setLevel(logging.ERROR)  # Only log ERROR and above
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)

        return error_handler

    @staticmethod
    def create_multiprocess_logging_setup(
        output_dir: str,
        log_filename: str = "main.log",
        when: str = "D",
        backup_count: int = 14,
        log_format: str = DEFAULT_FORMAT,
        date_format: str = DEFAULT_DATE_FORMAT,
    ) -> Tuple[mp.Queue, logging.handlers.QueueListener]:
        """Create multiprocess-safe logging setup with queue and rotating file handler

        Args:
            output_dir: Directory for log files
            log_filename: Base name of log file (without extension)
            when: Rotation interval ('D' for daily, 'H' for hourly, etc.)
            backup_count: Number of backup files to keep
            log_format: Format string for log messages
            date_format: Format string for timestamps

        Returns:
            Tuple of (queue, listener) for multiprocess logging
        """
        # Ensure output directory exists
        log_dir = Path(output_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create log queue for multiprocess communication
        log_queue = mp.Queue(-1)

        # Create formatter
        formatter = logging.Formatter(log_format, datefmt=date_format)

        # Create rotating file handler
        log_path = log_dir / log_filename
        file_handler = logging.handlers.TimedRotatingFileHandler(
            str(log_path),
            when=when,
            backupCount=backup_count,
            encoding="utf-8",
            delay=True,  # Delay opening the file until first log message (allows append behavior)
        )
        file_handler.setFormatter(formatter)

        # Create queue listener to handle log records from worker processes
        listener = logging.handlers.QueueListener(
            log_queue, file_handler, respect_handler_level=True
        )

        return log_queue, listener

    @staticmethod
    def get_queue_handler(queue: mp.Queue) -> logging.handlers.QueueHandler:
        """Get a queue handler for worker processes

        Args:
            queue: Multiprocessing queue for log records

        Returns:
            QueueHandler instance for worker process logging
        """
        return logging.handlers.QueueHandler(queue)

    @staticmethod
    def _get_console_stream():
        """Return a text stream for console output that uses UTF-8 and replaces
        characters that cannot be encoded by the terminal. This avoids
        UnicodeEncodeError when logging emoji on terminals with limited encodings.
        """
        try:
            # Prefer wrapping the underlying buffer to ensure UTF-8 encoding
            buf = sys.stdout.buffer
            return io.TextIOWrapper(
                buf, encoding="utf-8", errors="replace", line_buffering=True
            )
        except Exception:
            # Fallback to sys.stdout (may still raise on some platforms)
            return sys.stdout

    @staticmethod
    def setup_worker_logger(
        name: str,
        queue: mp.Queue,
        level: int = logging.INFO,
        add_console: bool = False,
    ) -> logging.Logger:
        """Setup logger for worker processes using queue handler

        Args:
            name: Logger name
            queue: Multiprocessing queue for log records
            level: Logging level
            add_console: Whether to add console handler (usually False for workers)

        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Clear existing handlers
        logger.handlers.clear()

        # Add queue handler for multiprocess logging
        queue_handler = LoggerManager.get_queue_handler(queue)
        logger.addHandler(queue_handler)

        # Optionally add console handler
        if add_console:
            console_stream = LoggerManager._get_console_stream()
            console_handler = logging.StreamHandler(console_stream)
            formatter = logging.Formatter(
                LoggerManager.DEFAULT_FORMAT, datefmt=LoggerManager.DEFAULT_DATE_FORMAT
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        logger.propagate = False

        return logger

    @staticmethod
    def stop_listener(listener: logging.handlers.QueueListener) -> None:
        """Stop and cleanup queue listener

        Args:
            listener: QueueListener instance to stop
        """
        listener.stop()
