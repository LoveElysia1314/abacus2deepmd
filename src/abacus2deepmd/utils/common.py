#!/usr/bin/env python
"""Common utilities and helpers for the project."""

import os
import logging
import traceback
import csv
import glob
import re
from typing import Any, List, Optional, Union, Callable, Tuple

import numpy as np


# Constants
DEFAULT_TEMPERATURE = "300"  # Default temperature for system parsing
DEFAULT_CONF = "0"  # Default configuration for system parsing


class CommonUtils:
    """Common utility functions used across the project."""

    @staticmethod
    def ensure_directory(path: str) -> None:
        """Ensure directory exists, create if necessary."""
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def get_project_root() -> str:
        """Get the project root directory."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigate up to project root (assuming this is in src/utils/)
        project_root = os.path.dirname(os.path.dirname(current_dir))
        return project_root

    @staticmethod
    def safe_divide(
        numerator: float, denominator: float, default: float = 0.0
    ) -> float:
        """Safe division that returns default value if denominator is zero."""
        if abs(denominator) < Constants.EPSILON:
            return default
        return numerator / denominator

    @staticmethod
    def nan_safe_mean(values: Union[List[float], np.ndarray]) -> float:
        """Calculate mean while safely handling NaN values."""
        if not values:
            return np.nan
        values_array = np.array(values)
        if np.all(np.isnan(values_array)):
            return np.nan
        return float(np.nanmean(values_array))

    @staticmethod
    def nan_safe_sem(values: Union[List[float], np.ndarray], ddof: int = 1) -> float:
        """Calculate standard error of mean while safely handling NaN values."""
        if not values or len(values) < 2:
            return np.nan
        values_array = np.array(values)
        if np.all(np.isnan(values_array)):
            return np.nan
        return float(np.nanstd(values_array, ddof=ddof) / np.sqrt(len(values_array)))

    @staticmethod
    def parse_system_name(system_name: str) -> Tuple[str, str, str]:
        """Parse system name to extract mol_id, conf, temperature.

        Expected format: struct_mol_{mol_id}_conf_{conf}_T{temperature}K
        Returns: (mol_id, conf, temperature) or defaults if parsing fails
        """
        match = re.match(r"struct_mol_(\d+)_conf_(\d+)_T(\d+)K", system_name)
        if match:
            return match.group(1), match.group(2), match.group(3)
        else:
            return system_name, DEFAULT_CONF, DEFAULT_TEMPERATURE

    @staticmethod
    def sort_system_names(system_names: List[str]) -> List[str]:
        """Sort system names by parsing and comparing numerical components.

        Sorting priority (ascending):
        1. mol_id (numerical)
        2. conf (numerical)
        3. temperature (numerical)

        Args:
            system_names: List of system names to sort

        Returns:
            Sorted list of system names
        """

        def get_sort_key(system_name: str):
            try:
                mol_id, conf, temp = CommonUtils.parse_system_name(system_name)
                # Convert to integers for proper numerical sorting
                return (int(mol_id), int(conf), int(temp))
            except (ValueError, TypeError):
                # If parsing fails, sort by string as fallback
                return (float("inf"), float("inf"), float("inf"), system_name)

        return sorted(system_names, key=get_sort_key)


# Re-exports for backward compatibility
ensure_directory = CommonUtils.ensure_directory
get_project_root = CommonUtils.get_project_root
safe_divide = CommonUtils.safe_divide
nan_safe_mean = CommonUtils.nan_safe_mean
nan_safe_sem = CommonUtils.nan_safe_sem


# ---- Data Processing Utilities (merged from data_utils.py) ----


class DataUtils:
    """Data processing and validation utilities"""

    @staticmethod
    def to_python_types(obj):
        """Convert numpy types to Python native types"""
        try:
            if isinstance(obj, (list, tuple)):
                return [DataUtils.to_python_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        except ImportError:
            return obj


class Constants:
    """Application constants"""

    # Default values
    DEFAULT_MAX_WORKERS = -1

    # File patterns

    # Analysis parameters

    # Numerical tolerances
    EPSILON = 1e-15

    # Logging formats
    DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ---- Parallel Execution Utilities (merged from parallel_utils.py) ----

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Callable, List, Any, Optional, Tuple


def setup_multiprocess_logging(
    output_dir: str,
    log_filename: str = "main.log",
    when: str = "D",
    backup_count: int = 14,
):
    """Create multiprocess logging queue and listener, return (log_queue, log_listener)"""
    from .logmanager import LoggerManager

    log_queue, log_listener = LoggerManager.create_multiprocess_logging_setup(
        output_dir=output_dir,
        log_filename=log_filename,
        when=when,
        backup_count=backup_count,
    )
    log_listener.start()
    return log_queue, log_listener


def stop_multiprocess_logging(log_listener):
    """Stop multiprocess logging listener"""
    from .logmanager import LoggerManager

    LoggerManager.stop_listener(log_listener)


def run_parallel_tasks(
    tasks: List[Any],
    worker_fn: Callable,
    workers: int = Constants.DEFAULT_MAX_WORKERS,
    initializer: Optional[Callable] = None,
    initargs: Optional[Tuple] = None,
    log_queue: Any = None,
    logger: Optional[logging.Logger] = None,
    error_logger: Optional[logging.Logger] = None,
    desc: str = "tasks",
    json_save_callback: Optional[Callable] = None,
    single_analysis_save_callback: Optional[Callable] = None,
) -> List[Any]:
    """
    Generic parallel task dispatching and collection
    Automatically handle exceptions and progress logging
    """
    from abacus2deepmd.utils.progress import ProgressReporter

    results: List[Any] = []
    total = len(tasks)
    # Note: do not pass single_analysis_save_callback to ProgressReporter
    # Instead, call the callback directly when each task completes and pass the result
    reporter = ProgressReporter(
        total=total,
        logger=logger if logger else logging.getLogger(__name__),
        desc=desc,
        error_logger=error_logger,
        json_save_callback=json_save_callback,  # Pass JSON save callback
    )

    pool_cls = ProcessPoolExecutor
    # Auto parallel: use CPU core count when workers=-1
    if workers == -1:
        try:
            workers = mp.cpu_count()
        except NotImplementedError:
            workers = 1
    if workers < 1:
        workers = 1
    pool_kwargs = {"max_workers": workers}
    if initializer:
        pool_kwargs["initializer"] = initializer
        if initargs:
            pool_kwargs["initargs"] = initargs

    if logger:
        logger.info(f"Starting parallel processing of {total} {desc}...")

    with pool_cls(**pool_kwargs) as pool:
        future_to_task = {}
        for task in tasks:
            future = pool.submit(worker_fn, task)
            future_to_task[future] = task
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
                # Success judgment: if result is tuple and first element is None, consider it failure
                is_success = True
                error_msg = None

                if result is None:
                    is_success = False
                    error_msg = "Worker returned None"
                elif (
                    isinstance(result, tuple)
                    and len(result) >= 1
                    and (result[0] is None)
                ):
                    is_success = False
                    error_msg = result[1] if len(result) > 1 else "Unknown error"

                # Stream: if single_analysis callback is provided and task succeeds, call it immediately with result
                if is_success and single_analysis_save_callback:
                    try:
                        single_analysis_save_callback(result)
                    except Exception as e:
                        if logger:
                            logger.warning(f"single_analysis callback failed: {e}")

                reporter.item_done(success=is_success, error=error_msg)
            except Exception as e:
                reporter.item_done(success=False, error=e)
                results.append((None, f"Exception: {e}"))

    reporter.finish()
    return results


def run_parallel_tasks_streaming(
    tasks: List[Any],
    worker_fn: Callable,
    workers: int = 1,
    initializer: Optional[Callable] = None,
    initargs: Optional[Tuple] = None,
    log_queue: Any = None,
    logger: Optional[logging.Logger] = None,
    error_logger: Optional[logging.Logger] = None,
    desc: str = "tasks",
    json_save_callback: Optional[callable] = None,  # New: JSON save callback
    single_analysis_save_callback: Optional[
        callable
    ] = None,  # New: single_analysis save callback
) -> List[Any]:
    """
    Streaming parallel task dispatching and collection - record detailed results immediately when each task completes
    """
    from abacus2deepmd.utils.progress import ProgressReporter

    results: List[Any] = []
    total = len(tasks)
    # Note: do not pass single_analysis_save_callback to ProgressReporter
    # Instead, call the callback directly when each task completes and pass the result
    reporter = ProgressReporter(
        total=total,
        logger=logger if logger else logging.getLogger(__name__),
        desc=desc,
        error_logger=error_logger,
        json_save_callback=json_save_callback,  # Pass JSON save callback
    )

    pool_cls = ProcessPoolExecutor
    # Auto parallel: use CPU core count when workers=-1
    if workers == -1:
        try:
            workers = mp.cpu_count()
        except NotImplementedError:
            workers = 1
    if workers < 1:
        workers = 1
    pool_kwargs = {"max_workers": workers}
    if initializer:
        pool_kwargs["initializer"] = initializer
        if initargs:
            pool_kwargs["initargs"] = initargs

    if logger:
        logger.info(f"Starting streaming parallel processing of {total} {desc}...")

    with pool_cls(**pool_kwargs) as pool:
        future_to_task = {}
        for task in tasks:
            future = pool.submit(worker_fn, task)
            future_to_task[future] = task
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
                # Success judgment: if result is tuple and first element is None, consider it failure
                is_success = True
                error_msg = None

                if result is None:
                    is_success = False
                    error_msg = "Worker returned None"
                elif (
                    isinstance(result, tuple)
                    and len(result) >= 1
                    and (result[0] is None)
                ):
                    is_success = False
                    error_msg = result[1] if len(result) > 1 else "Unknown error"

                # Stream: if single_analysis callback is provided and task succeeds, call it immediately with result
                if is_success and single_analysis_save_callback:
                    try:
                        single_analysis_save_callback(result)
                    except Exception as e:
                        if logger:
                            logger.warning(f"single_analysis callback failed: {e}")

                reporter.item_done(success=is_success, error=error_msg)
            except Exception as e:
                reporter.item_done(success=False, error=e)
                results.append((None, f"Exception: {e}"))

    reporter.finish()
    return results
