#!/usr/bin/env python
"""Unified scheduler module for ABACUS-STRU-Analyser.

This module combines both thread-based and process-based scheduling
for system analysis tasks.
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from typing import List, Optional, Callable, Any, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp

logger = logging.getLogger(__name__)


# ---- Thread-based Task Scheduler (from task_scheduler.py) ----


@dataclass
class AnalysisTask:
    system_path: str
    system_name: str
    pre_sampled_frames: Optional[List[int]] = None
    pre_stru_files: Optional[List[str]] = None
    start_time: float = 0.0
    error: Optional[str] = None

    def mark_start(self):
        self.start_time = time.time()

    def mark_done(self):
        pass

    def mark_failed(self, err: str):
        self.error = err


class TaskScheduler:
    """Simple threaded task scheduler for deferred system analysis."""

    def __init__(self, max_workers: int = 8):
        self.max_workers = max_workers
        self.tasks: List[AnalysisTask] = []
        self.results: List[Any] = []
        self.failed: List[AnalysisTask] = []

    def run(self, analyse_fn: Callable[[AnalysisTask], Any]) -> List[Any]:
        if not self.tasks:
            return []
        logger.info(
            f"TaskScheduler: 启动 {len(self.tasks)} 个任务，线程数={self.max_workers}"
        )
        start_overall = time.time()
        futures = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for task in self.tasks:

                def _wrap(t: AnalysisTask):
                    def _call():
                        t.mark_start()
                        try:
                            result = analyse_fn(t)
                            t.mark_done()
                            return (t, result)
                        except Exception as e:  # noqa
                            t.mark_failed(str(e))
                            logger.warning(f"任务失败 {t.system_name}: {e}")
                            return (t, None)

                    return _call

                futures.append(executor.submit(_wrap(task)))

            completed = 0
            last_log = time.time()
            for fut in as_completed(futures):
                task, result = fut.result()
                completed += 1
                if result is not None:
                    self.results.append(result)
                else:
                    self.failed.append(task)
                now = time.time()
                if now - last_log >= 10 or completed == len(futures):
                    logger.info(
                        f"进度: {completed}/{len(futures)} 完成, 耗时 {now-start_overall:.1f}s, 失败 {len(self.failed)}"
                    )
                    last_log = now
        total = time.time() - start_overall
        logger.info(
            f"TaskScheduler: 全部完成 成功 {len(self.results)} 失败 {len(self.failed)} 总耗时 {total:.1f}s"
        )
        return self.results


# ---- Process-based Scheduler (from process_scheduler.py) ----


# 全局上下文用于进程间通信
class ProcessSchedulerContext:
    """ProcessScheduler的全局上下文"""

    _log_queue = None

    @classmethod
    def set_log_queue(cls, log_queue: Optional[mp.Queue]):
        """设置日志队列"""
        cls._log_queue = log_queue

    @classmethod
    def get_log_queue(cls) -> Optional[mp.Queue]:
        """获取日志队列"""
        return cls._log_queue


@dataclass
class ProcessAnalysisTask:
    system_path: str
    system_name: str
    pre_sampled_frames: Optional[List[int]]
    pre_stru_files: Optional[List[str]]
    sampling_only: bool = False


def _set_single_thread_env():
    # 通用环境变量限制线程数
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("BLIS_NUM_THREADS", "1")


def _worker(
    task: ProcessAnalysisTask, analyser_params: Dict[str, Any]
) -> Tuple[str, Any, float]:
    start = time.time()
    _set_single_thread_env()

    # Setup multiprocess logging for worker
    log_queue = ProcessSchedulerContext.get_log_queue()
    if log_queue is not None:
        from ..utils.logmanager import LoggerManager  # type: ignore

        LoggerManager.setup_worker_logger(
            name="WorkerProcess", queue=log_queue, level=logging.INFO, add_console=False
        )

    try:
        # 延迟导入（确保环境变量已生效）
        from .system_analyser import SystemAnalyser, ErrorHandler  # type: ignore

        analyser = ErrorHandler.create_analyser(
            sample_ratio=analyser_params.get("sample_ratio", 0.1),
            power_p=analyser_params.get("power_p", 0.5),
            pca_variance_ratio=analyser_params.get("pca_variance_ratio", 0.90),
        )

        if task.sampling_only:
            result = analyser.analyse_system_sampling_only(
                task.system_path, pre_sampled_frames=task.pre_sampled_frames
            )
        else:
            result = analyser.analyse_system(
                task.system_path,
                pre_sampled_frames=task.pre_sampled_frames,
                pre_stru_files=task.pre_stru_files,
            )
        return task.system_name, result, time.time() - start
    except Exception as e:  # noqa
        return task.system_name, (None, str(e)), time.time() - start


class ProcessScheduler:
    """Process based scheduler for per-system parallel analysis.

    设计目标:
     - 体系为粒度, 动态工作窃取 (进程池自动调度)
     - 单体系内部 numpy / BLAS / OpenMP 限制为 1 线程, 避免超订阅
     - 复用轻量发现的 selected_files, 避免重复 I/O

    Windows 下使用 spawn, 需在调用端放在 `if __name__ == "__main__":` 后。
    """

    def __init__(
        self,
        max_workers: int,
        analyser_params: Dict[str, Any],
        log_queue: Optional[mp.Queue] = None,
    ):
        self.max_workers = max_workers
        self.analyser_params = analyser_params
        # 设置全局日志队列
        ProcessSchedulerContext.set_log_queue(log_queue)
        self.tasks: List[ProcessAnalysisTask] = []

        # 设置调度器的logger，使用传入的log_queue
        if log_queue is not None:
            from ..utils.logmanager import LoggerManager

            self.logger = LoggerManager.setup_worker_logger(
                name="ProcessScheduler",
                queue=log_queue,
                level=logging.INFO,
                add_console=False,
            )
        else:
            self.logger = logging.getLogger("ProcessScheduler")

    def run(self) -> List[Any]:
        if not self.tasks:
            return []
        t0 = time.time()
        self.logger.info(
            f"ProcessScheduler: 提交 {len(self.tasks)} 个体系, 进程数={self.max_workers}"
        )
        results: List[Any] = []
        failures = 0
        with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
            future_map = {
                pool.submit(_worker, task, self.analyser_params): task
                for task in self.tasks
            }
            completed = 0
            last_log = t0
            for fut in as_completed(future_map):
                completed += 1
                sys_name, result, dur = fut.result()
                if result is None or (isinstance(result, tuple) and result[0] is None):
                    failures += 1
                    self.logger.warning(
                        f"({completed}/{len(future_map)}) {sys_name} 体系分析失败 (用时 {dur:.2f}s)"
                    )
                else:
                    results.append(result)
                    # 完成体系分析任务（不输出单独体系完成信息）
                    # self.logger.info(f"({completed}/{len(future_map)}) {sys_name} 体系分析完成 (用时 {dur:.2f}s)")
                now = time.time()
                if now - last_log >= 30 or completed == len(
                    future_map
                ):  # 增加到30秒输出一次总体进度
                    self.logger.info(
                        f"总体进度: {completed}/{len(future_map)} 成功 {len(results)} 失败 {failures} 累计耗时 {now-t0:.1f}s"
                    )
                    last_log = now
        self.logger.info(
            f"ProcessScheduler 完成: 成功 {len(results)} 失败 {failures} 总耗时 {time.time()-t0:.1f}s"
        )
        return results
