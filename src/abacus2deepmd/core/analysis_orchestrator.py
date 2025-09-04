#!/usr/bin/env python
"""Analysis Orchestrator for coordinating analysis components."""

import os
import logging
import glob
import multiprocessing as mp
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

from ..core.system_analyser import SystemAnalyser, ErrorHandler
from ..io.result_saver import ResultSaver
from ..io.file_utils import FileUtils
from ..io.path_manager import PathManager
from ..io.file_utils import load_sampling_reuse_map
from ..utils.logmanager import LoggerManager
from ..utils.common import run_parallel_tasks
from ..utils.progress import stream_save, ProgressReporter


@dataclass
class AnalysisConfig:
    """分析配置类"""

    # 核心参数
    sample_ratio: float = 0.1
    power_p: float = -0.5
    pca_variance_ratio: float = 0.90

    # 运行配置
    workers: int = -1
    output_dir: str = "analysis_results"
    search_paths: List[str] = None
    include_project: bool = False
    force_recompute: bool = False

    # Power参数测试相关参数
    max_systems: Optional[int] = None  # 测试时最多使用的体系数量

    # 流程控制
    steps: List[int] = None  # 要执行的步骤列表，如[1,2,3,4]

    def __post_init__(self):
        if self.search_paths is None:
            self.search_paths = []
        if self.steps is None:
            self.steps = [1, 2, 3, 4]  # 默认执行所有步骤

    def as_param_dict(self) -> Dict[str, Any]:
        """将配置转换为参数字典，用于保存和比较"""
        return {
            "sample_ratio": self.sample_ratio,
            "power_p": self.power_p,
            "pca_variance_ratio": self.pca_variance_ratio,
        }


# 多进程工作上下文
class WorkerContext:
    """多进程工作上下文，避免使用全局变量"""

    _analyser = None
    _reuse_map: Dict[str, List[int]] = {}

    @classmethod
    def initialize(cls, analyser, reuse_map: Dict[str, List[int]]):
        """初始化工作上下文"""
        cls._analyser = analyser
        cls._reuse_map = reuse_map or {}

    @classmethod
    def get_analyser(cls):
        """获取分析器实例"""
        return cls._analyser

    @classmethod
    def get_reuse_map(cls):
        """获取复用映射"""
        return cls._reuse_map

    @classmethod
    def set_sampling_only(cls, sampling_only: bool):
        """设置采样模式"""
        if cls._analyser:
            pass


class AnalysisOrchestrator:
    """分析流程编排器 - 核心逻辑提取"""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.logger: Optional[logging.Logger] = None
        self.log_queue = None
        self.log_listener = None
        self.current_step: int = 0  # 当前执行的步骤编号

    def setup_logging(self) -> None:
        """设置多进程安全日志系统"""
        analysis_results_dir = os.path.join(
            FileUtils.get_project_root(), "analysis_results"
        )
        os.makedirs(analysis_results_dir, exist_ok=True)

        self.log_queue, self.log_listener = (
            LoggerManager.create_multiprocess_logging_setup(
                output_dir=analysis_results_dir,
                log_filename="main.log",
                when="D",
                backup_count=14,
            )
        )

        self.log_listener.start()

        # 创建主日志记录器
        self.logger = LoggerManager.setup_worker_logger(
            name=__name__, queue=self.log_queue, level=logging.INFO, add_console=True
        )

        # 创建错误日志记录器（专门记录错误信息）
        error_log_file = os.path.join(analysis_results_dir, "analysis_errors.log")
        self.error_logger = LoggerManager.create_logger_with_error_log(
            name=f"{__name__}.errors",
            level=logging.ERROR,
            add_console=False,  # 错误日志不输出到控制台
            error_log_file=error_log_file,
            log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            date_format="%Y-%m-%d %H:%M:%S",
        )

    def cleanup_logging(self) -> None:
        """清理日志系统"""
        if hasattr(self, "log_listener") and self.log_listener:
            try:
                LoggerManager.stop_listener(self.log_listener)
                if self.logger:
                    self.logger.info("日志监听器已停止")
            except (OSError, ValueError) as e:
                if self.logger:
                    self.logger.warning("停止日志监听器时出错: %s", e)

    def _persist_progress(
        self, path_manager: PathManager, save_mode: str = "full", target=None
    ) -> None:
        """统一进度持久化方法

        Args:
            path_manager: PathManager实例
            save_mode: 'full' 或 'incremental'
            target: 仅在incremental模式下使用
        """
        try:
            current_analysis_params = self.config.as_param_dict()
            if save_mode == "incremental" and target:
                path_manager.save_analysis_targets_incremental(
                    target, current_analysis_params
                )
            else:
                path_manager.save_analysis_targets(current_analysis_params)
        except Exception as e:
            self.logger.debug(f"进度持久化失败: {e}")

    def resolve_search_paths(self) -> List[str]:
        """解析搜索路径，支持通配符展开"""
        if not self.config.search_paths:
            return [os.path.abspath(os.path.join(os.getcwd(), ".."))]

        resolved_paths = []
        for path_pattern in self.config.search_paths:
            expanded = glob.glob(path_pattern, recursive=True)
            if expanded:
                expanded_dirs = [p for p in expanded if os.path.isdir(p)]
                resolved_paths.extend(expanded_dirs)
                if expanded_dirs:
                    self.logger.info(
                        "通配符 '%s' 展开为 %d 个目录", path_pattern, len(expanded_dirs)
                    )
            else:
                if os.path.isdir(path_pattern):
                    resolved_paths.append(path_pattern)
                else:
                    self.logger.warning("路径不存在或不是目录: %s", path_pattern)

        unique_paths = list(set(os.path.abspath(p) for p in resolved_paths))
        return unique_paths

    def setup_output_directory(self) -> Tuple[PathManager, str]:
        """设置输出目录和路径管理器"""
        current_analysis_params = self.config.as_param_dict()

        path_manager = PathManager(self.config.output_dir)
        actual_output_dir = path_manager.set_output_dir_for_params(
            current_analysis_params
        )
        path_manager.output_dir = actual_output_dir  # 设置output_dir属性

        self.logger.info(f"使用参数专用目录: {actual_output_dir}")
        return path_manager, actual_output_dir

    def setup_analysis_targets(
        self, path_manager: PathManager, search_paths: List[str]
    ) -> bool:
        """设置分析目标，返回是否可以使用现有结果"""

        # 检查是否已有完整结果（快速路径）
        if (
            not self.config.force_recompute
            and path_manager.check_existing_complete_results()
        ):
            path_manager.load_sampled_frames_from_csv()
            return True

        # 加载现有目标状态
        loaded_existing = path_manager.load_analysis_targets()
        if loaded_existing:
            self.logger.info("成功加载已有的分析目标状态")
        else:
            self.logger.info("未找到已有的分析目标文件，将创建新的")

        # 加载发现结果并去重
        path_manager.load_from_discovery(search_paths)
        path_manager.deduplicate_targets()

        return False

    def determine_workers(self) -> int:
        """确定工作进程数"""
        if self.config.workers == -1:
            try:
                workers = int(
                    os.environ.get(
                        "SLURM_CPUS_PER_TASK",
                        os.environ.get("SLURM_JOB_CPUS_PER_NODE", mp.cpu_count()),
                    )
                )
            except (ValueError, TypeError) as e:
                self.logger.warning(
                    "Failed to determine optimal worker count from environment: %s", e
                )
                workers = max(1, mp.cpu_count())
        else:
            workers = max(1, self.config.workers)
        return workers

    def execute_analysis(self, path_manager: PathManager) -> List[tuple]:
        """执行分析（支持仅采样模式）"""
        # 保存当前path_manager引用，用于紧急保存
        self.current_path_manager = path_manager

        # 创建分析器
        analyser = ErrorHandler.create_analyser(
            sample_ratio=self.config.sample_ratio,
            power_p=self.config.power_p,
            pca_variance_ratio=self.config.pca_variance_ratio,
        )

        # 获取全部分析目标
        all_targets = path_manager.get_all_targets()

        # 过滤掉已有分析结果的系统（增量计算）
        pending_targets = []
        skipped_count = 0
        for target in all_targets:
            if not self.config.force_recompute and ResultSaver.should_skip_analysis(
                path_manager.output_dir, target.system_name
            ):
                self.logger.debug(f"{target.system_name} 体系分析文件存在，跳过分析")
                skipped_count += 1
                continue
            pending_targets.append(target)

        if skipped_count > 0:
            self.logger.info(f"增量计算：跳过 {skipped_count} 个已有分析结果的体系")

        if not pending_targets:
            return []

        # 采样复用判定
        reuse_map = {}
        if not self.config.force_recompute and hasattr(path_manager, "output_dir"):
            targets_file = os.path.join(
                path_manager.output_dir, "analysis_targets.json"
            )
            if os.path.exists(targets_file):
                reuse_map = load_sampling_reuse_map(targets_file)
        self.logger.info(
            f"采样复用：待处理 {len(pending_targets)} 个体系，可复用 {len(reuse_map)} 个采样帧"
        )

        analysis_targets = pending_targets

        # 按照与JSON相同的排序规则排序分析目标，确保并行执行顺序一致
        from ..utils.common import CommonUtils

        analysis_targets.sort(
            key=lambda t: CommonUtils.sort_system_names([t.system_name])[0]
        )

        system_paths = [t.system_path for t in analysis_targets]

        if not system_paths:
            self.logger.info("没有需要处理的系统")
            return []

        workers = self.determine_workers()

        self.logger.info(f"准备分析 {len(system_paths)} 个系统...")

        # 执行分析
        if workers > 1:
            return self._parallel_analysis(
                analyser, system_paths, path_manager, workers, reuse_map
            )
        else:
            return self._sequential_analysis(
                analyser, system_paths, path_manager, reuse_map
            )

    def _parallel_analysis(
        self,
        analyser: SystemAnalyser,
        system_paths: List[str],
        path_manager: PathManager,
        workers: int,
        reuse_map: Dict[str, List[int]],
    ) -> List[tuple]:
        """并行分析系统"""
        # 在强制重算模式下，使用空的reuse_map
        effective_reuse_map = {} if self.config.force_recompute else reuse_map
        initializer_args = (
            analyser.sample_ratio,
            analyser.power_p,
            analyser.pca_variance_ratio,
            effective_reuse_map,
            False,
            self.log_queue,
        )

        # 创建JSON保存回调
        def save_json_callback():
            self._persist_progress(path_manager, "full")

        # 使用列表来存储结果，方便回调函数访问
        analysis_results = []

        # 创建single_analysis保存回调（接收单个 result 并立即保存）
        def save_single_analysis_callback(result):
            try:
                if result:
                    stream_save(result, path_manager, self.logger, sampling_only=False)
            except Exception as e:
                # 不阻塞主流程，仅记录警告
                if self.logger:
                    self.logger.warning(f"保存 single_analysis 失败: {e}")

        # 调用通用并行工具
        results = run_parallel_tasks(
            tasks=system_paths,
            worker_fn=_child_worker,
            workers=workers,
            initializer=_child_init,
            initargs=initializer_args,
            log_queue=self.log_queue,
            logger=self.logger,
            desc="体系分析",
            json_save_callback=save_json_callback,  # 传入JSON保存回调
            single_analysis_save_callback=save_single_analysis_callback,  # 传入single_analysis保存回调（接收单个result）
        )

        # 过滤有效结果并流式保存
        for result in results:
            if result:
                analysis_results.append(result)
                # 流式保存：每个体系分析完成后立即保存其结果
                stream_save(result, path_manager, self.logger, sampling_only=False)

        return analysis_results

    def _sequential_analysis(
        self,
        analyser: SystemAnalyser,
        system_paths: List[str],
        path_manager: PathManager,
        reuse_map: dict = None,
    ) -> List[tuple]:
        """顺序分析系统"""
        reuse_map = reuse_map or {}
        path_to_presampled = {}
        for t in path_manager.targets:
            if t.system_path and t.system_name in reuse_map:
                path_to_presampled[t.system_path] = reuse_map[t.system_name]

        # 创建JSON保存回调
        def save_json_callback():
            self._persist_progress(path_manager, "full")

        # 初始化结果列表
        analysis_results = []

        # 创建single_analysis保存回调（接收单个 result 并立即保存）
        def save_single_analysis_callback(result):
            try:
                if result:
                    stream_save(result, path_manager, self.logger, sampling_only=False)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"保存 single_analysis 失败: {e}")

        total_systems = len(system_paths)
        reporter = ProgressReporter(
            total=total_systems,
            logger=self.logger,
            desc="体系分析",
            error_logger=self.error_logger,
            json_save_callback=save_json_callback,  # 传入JSON保存回调
        )
        for idx, system_path in enumerate(system_paths):
            pre_frames_data = path_to_presampled.get(system_path)
            pre_frames = None
            # 在强制重算模式下，忽略预采样帧
            if (
                not self.config.force_recompute
                and pre_frames_data
                and isinstance(pre_frames_data, dict)
            ):
                pre_frames = pre_frames_data.get("sampled_frames")
                # 只有当预采样帧不为空时才使用
                if not pre_frames:
                    pre_frames = None
            try:
                result = analyser.analyse_system(
                    system_path, pre_sampled_frames=pre_frames
                )
                if result:
                    analysis_results.append(result)
                    # 流式保存：每个体系分析完成后立即保存其结果
                    stream_save(result, path_manager, self.logger, sampling_only=False)
                    reporter.item_done(success=True)
                else:
                    reporter.item_done(success=False, error="返回结果为空")
            except Exception as e:
                reporter.item_done(success=False, error=e)

        reporter.finish()
        return analysis_results

    def save_results(
        self, analysis_results: List[tuple], path_manager: PathManager
    ) -> None:
        """保存分析结果（单帧指标已在流式处理中保存，这里只保存汇总信息）"""
        if not analysis_results:
            self.logger.warning("没有分析结果需要保存")
            return

        # 注意：单帧指标文件已在每个体系分析完成后流式保存
        # 这里只同步采样帧到PathManager.targets并保存analysis_targets.json

        # 同步采样帧到PathManager.targets
        path_manager.update_sampled_frames_from_results(analysis_results)

        # 保存analysis_targets.json
        try:
            current_analysis_params = self.config.as_param_dict()
            path_manager.save_analysis_targets(current_analysis_params)
        except Exception as e:
            self.logger.warning(f"保存analysis_targets.json时出错: {e}")


def _worker_analyse_system(
    system_path: str,
    sample_ratio: float,
    power_p: float,
    pca_variance_ratio: float,
    pre_sampled_frames: Optional[List[int]] = None,
    sampling_only: bool = False,
):
    """备用工作函数：独立创建分析器并执行"""
    analyser = ErrorHandler.create_analyser(
        sample_ratio=sample_ratio,
        power_p=power_p,
        pca_variance_ratio=pca_variance_ratio,
    )

    if sampling_only:
        return analyser.analyse_system_sampling_only(
            system_path, pre_sampled_frames=pre_sampled_frames
        )
    else:
        return analyser.analyse_system(
            system_path, pre_sampled_frames=pre_sampled_frames
        )


def _child_init(
    sample_ratio: float,
    power_p: float,
    pca_variance_ratio: float,
    reuse_map: Dict[str, List[int]],
    sampling_only: bool = False,
    log_queue: mp.Queue = None,
):
    """工作进程初始化"""
    analyser = ErrorHandler.create_analyser(
        sample_ratio=sample_ratio,
        power_p=power_p,
        pca_variance_ratio=pca_variance_ratio,
    )

    WorkerContext.initialize(analyser, reuse_map)
    WorkerContext.set_sampling_only(sampling_only)

    # Setup multiprocess logging for worker
    if log_queue is not None:
        LoggerManager.setup_worker_logger(
            name="WorkerProcess", queue=log_queue, level=logging.INFO, add_console=False
        )


def _child_worker(system_path: str):
    """工作进程执行函数（支持采样帧复用）"""
    analyser = WorkerContext.get_analyser()

    if analyser is None:
        # 备用情况：如果上下文未初始化，使用默认参数
        return _worker_analyse_system(
            system_path, 0.05, 0.5, 0.90, pre_sampled_frames=None, sampling_only=True
        )

    sys_name = os.path.basename(system_path.rstrip("/\\"))
    reuse_map = WorkerContext.get_reuse_map()
    pre_frames_data = reuse_map.get(sys_name)

    # 从字典中提取采样帧列表
    pre_frames = None
    if pre_frames_data and isinstance(pre_frames_data, dict):
        pre_frames = pre_frames_data.get("sampled_frames")
        # 只有当预采样帧不为空时才使用
        if not pre_frames:
            pre_frames = None

    # 步骤1只执行采样，不包含DeepMD导出
    return analyser.analyse_system(system_path, pre_sampled_frames=pre_frames)
