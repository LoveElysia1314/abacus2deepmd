"""Unified progress and error reporting tool

ProgressReporter is responsible for:
1. Timed progress output (based on time intervals or completion)
2. Statistics on success/failure/throughput
3. Optional structured error output
4. Reducing duplicate code across steps (sequential analysis/parallel tasks/DeepMD export)

Usage:
    reporter = ProgressReporter(total, logger, desc="System analysis", error_logger=err_logger)
    for item in items:
        try:
            ...
            reporter.item_done(success=True)
        except Exception as e:
            reporter.item_done(success=False, error=e, context={"item": str(item)})
    reporter.finish()

Environment variables:
    LOG_PROGRESS_INTERVAL overrides default progress output interval (seconds), minimum 5 seconds
"""

from __future__ import annotations

import os
import time
import logging
from typing import Optional, Dict, Any, Union
from ..io.result_saver import ResultSaver


class ProgressReporter:
    def __init__(
        self,
        total: int,
        logger: logging.Logger,
        desc: str = "task",
        interval: float = 60.0,
        error_logger: Optional[logging.Logger] = None,
        show_start: bool = True,
        # New: Support timed JSON save callback
        json_save_callback: Optional[callable] = None,
        # New: Support timed single_analysis file save callback
        single_analysis_save_callback: Optional[callable] = None,
    ) -> None:
        self.total = max(total, 0)
        self.logger = logger
        self.error_logger = error_logger
        self.desc = desc
        # Allow environment variable override
        env_interval = os.environ.get("LOG_PROGRESS_INTERVAL")
        if env_interval:
            try:
                interval = float(env_interval)
            except ValueError:
                pass
        self.interval = max(interval, 5.0)  # Minimum 5s to avoid spam

        self.start_time = time.time()
        self.last_report_time = self.start_time
        self.done = 0
        self.success = 0
        self.failed = 0
        self.show_start = show_start
        # New: JSON save callback
        self.json_save_callback = json_save_callback
        # New: single_analysis save callback
        self.single_analysis_save_callback = single_analysis_save_callback
        # New: Error set for collecting duplicate errors
        self.error_set = set()

        if self.show_start and self.logger:
            self.logger.info(f"Starting {self.desc} processing: Total {self.total}")

    # ---- Public methods ----
    def item_done(
        self,
        success: bool,
        error: Optional[Union[Exception, str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record completion of individual task

        success: Whether successful
        error: Exception object or error message string
        context: Additional context dictionary (e.g., system name/path, etc.)
        """
        self.done += 1
        if success:
            self.success += 1
        else:
            self.failed += 1
            self._log_error(error, context)
        self._maybe_report()

    def finish(self) -> None:
        elapsed = time.time() - self.start_time
        throughput = (self.done / elapsed) if elapsed > 0 else 0.0
        if self.logger:
            self.logger.info(
                f"{self.desc} completed: Total {self.done}/{self.total} | Success {self.success} | Failed {self.failed} | "
                f"Elapsed {elapsed:.1f}s | Throughput {throughput:.2f}/s"
            )

        # Output remaining error messages
        if self.error_set:
            for error_msg in self.error_set:
                msg = f"{self.desc} failed | Message {error_msg}"
                try:
                    if self.error_logger:
                        self.error_logger.error(msg)
                    elif self.logger:
                        self.logger.error(msg)
                    else:
                        print(msg)
                except Exception as e:
                    if self.logger:
                        self.logger.warning(
                            f"Error logging failed: {e}, original error: {msg}"
                        )
            # Clear error set
            self.error_set.clear()

    # ---- Internal methods ----
    def _maybe_report(self, force: bool = False) -> None:
        if not self.logger:
            return
        now = time.time()
        should_report = (
            force
            or (now - self.last_report_time >= self.interval)
            or (self.done >= self.total and self.total > 0)
        )

        if should_report:
            pct = (self.done / self.total * 100) if self.total else 100.0
            self.logger.info(
                f"{self.desc} progress: {self.done}/{self.total} ({pct:.1f}%) - Success {self.success} Failed {self.failed}"
            )
            self.last_report_time = now

            # Output collected error messages
            if self.error_set:
                for error_msg in self.error_set:
                    msg = f"{self.desc} failed | Message {error_msg}"
                    try:
                        if self.error_logger:
                            self.error_logger.error(msg)
                        elif self.logger:
                            self.logger.error(msg)
                        else:
                            print(msg)
                    except Exception as e:
                        if self.logger:
                            self.logger.warning(
                                f"Error logging failed: {e}, original error: {msg}"
                            )
            # Clear error set
            self.error_set.clear()

            # Added: Trigger JSON save during progress reporting
            if (
                self.json_save_callback and not force
            ):  # force=True is called by finish(), avoid duplicate saves
                try:
                    self.json_save_callback()
                    self.logger.debug(
                        f"{self.desc} progress saved: analysis_targets.json updated"
                    )
                except Exception as e:
                    self.logger.warning(f"{self.desc} progress save failed: {e}")

            # Added: Trigger single_analysis save during progress reporting
            # Note: single_analysis save now handled through other methods, no longer called by ProgressReporter
            # if self.single_analysis_save_callback and not force:
            #     try:
            #         self.single_analysis_save_callback()
            #         self.logger.debug(f"{self.desc} progress saved: single_analysis files updated")
            #     except Exception as e:
            #         self.logger.warning(f"{self.desc} single_analysis save failed: {e}")

    def _log_error(
        self, error: Optional[Union[Exception, str]], context: Optional[Dict[str, Any]]
    ) -> None:
        if not error:
            return
        msg = None
        if isinstance(error, Exception):
            msg = f"{type(error).__name__}: {str(error)}"
        else:
            msg = str(error)
        # Add to error set to avoid duplicates
        self.error_set.add(msg)


__all__ = ["ProgressReporter"]


def stream_save(
    result: tuple, path_manager, logger: logging.Logger, sampling_only: bool = False
) -> None:
    """Universal streaming save function for immediate saving of analysis results

    Args:
        result: (metrics, sampled_frames) or (metrics, sampled_frames, swap_count)
        path_manager: PathManager instance
        logger: Logger instance
        sampling_only: Whether sampling only mode
    """
    try:

        ResultSaver.save_single_system(
            output_dir=path_manager.output_dir,
            result=result,
            sampling_only=sampling_only,
        )
        # Incremental JSON save: Update analysis_targets.json in real-time
        if hasattr(result[0], "system_name"):
            # Find corresponding target and update sampled_frames
            for target in path_manager.targets:
                if target.system_name == result[0].system_name:
                    target.sampled_frames = getattr(result[0], "sampled_frames", [])
                    break
    except Exception as save_e:
        logger.warning(
            f"Streaming save failed {result[0].system_name if hasattr(result[0], 'system_name') else 'unknown'}: {save_e}"
        )
