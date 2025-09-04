#!/usr/bin/env python

import logging
import os
import re
from typing import List, Optional, Tuple

import numpy as np

from ..io.stru_parser import StrUParser
from .metrics import MetricCalculator, TrajectoryMetrics
from .sampler import PowerMeanSampler


class ErrorHandler:
    """Utilities for enhanced error handling and logging"""

    @staticmethod
    def create_analyser(
        sample_ratio: float, power_p: float, pca_variance_ratio: float
    ) -> "SystemAnalyser":
        """Factory method to create SystemAnalyser with standard parameters"""
        return SystemAnalyser(
            include_hydrogen=False,
            sample_ratio=sample_ratio,
            power_p=power_p,
            pca_variance_ratio=pca_variance_ratio,
        )

    @staticmethod
    def log_detailed_error(
        logger: logging.Logger,
        error: Exception,
        context: str = "",
        additional_info: dict = None,
    ) -> None:
        """Log detailed error information including stack trace and context

        Args:
            logger: Logger instance to use for logging
            error: The exception that occurred
            context: Additional context information about where the error occurred
            additional_info: Dictionary of additional information to log
        """
        import traceback

        error_details = traceback.format_exc()

        if context:
            logger.error(f"{context}: {str(error)}")
        else:
            logger.error(f"错误: {str(error)}")

        logger.error(f"详细错误信息:\n{error_details}")

        # Log additional context information
        if additional_info:
            for key, value in additional_info.items():
                logger.error(f"{key}: {value}")

        # Log chained exceptions if present
        if hasattr(error, "__cause__") and error.__cause__:
            logger.error(f"根本原因: {error.__cause__}")
        if hasattr(error, "__context__") and error.__context__:
            logger.error(f"上下文: {error.__context__}")

    @staticmethod
    def log_simple_error(
        logger: logging.Logger, error: Exception, context: str = ""
    ) -> None:
        """Log simple error information without stack trace

        Args:
            logger: Logger instance to use for logging
            error: The exception that occurred
            context: Additional context information about where the error occurred
        """
        if context:
            logger.error(f"{context}: {str(error)}")
        else:
            logger.error(f"错误: {str(error)}")


class ValidationUtils:
    """Utilities for data validation and checking"""

    @staticmethod
    def is_empty(obj) -> bool:
        """Unified empty check for various data types

        Args:
            obj: Object to check (None, list, numpy array, etc.)

        Returns:
            True if object is empty or None, False otherwise
        """
        if obj is None:
            return True

        # For numpy arrays, check size attribute first
        if hasattr(obj, "size"):
            return obj.size == 0

        # For containers with len() method
        if hasattr(obj, "__len__"):
            return len(obj) == 0

        # For other types, consider non-None as non-empty
        return False


class RMSDCalculator:
    """RMSD计算和结构对齐工具类"""

    @staticmethod
    def kabsch_align(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """Align coordinates P onto Q using Kabsch algorithm.

        Args:
            P: (n_atoms, 3)
            Q: (n_atoms, 3)
        Returns:
            Aligned copy of P.
        """
        if P.size == 0 or Q.size == 0:
            return P.copy()
        Pc = P - P.mean(axis=0)
        Qc = Q - Q.mean(axis=0)
        C = Pc.T @ Qc
        V, _, Wt = np.linalg.svd(C)
        d = np.sign(np.linalg.det(V @ Wt))
        U = V @ np.diag([1, 1, d]) @ Wt
        return Pc @ U

    @staticmethod
    def iterative_mean_structure(
        positions_list: List[np.ndarray], max_iter: int = 20, tol: float = 1e-6
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Iteratively align frames to a reference and recompute mean structure.

        Returns mean structure and list of aligned frames.
        """
        if not positions_list:
            return np.array([]), []
        ref = positions_list[0].copy()
        aligned_positions = list(positions_list)
        for _ in range(max_iter):
            aligned_positions = [
                RMSDCalculator.kabsch_align(pos, ref) for pos in positions_list
            ]
            mean_structure = np.mean(aligned_positions, axis=0)
            if np.linalg.norm(mean_structure - ref) < tol:
                break
            ref = mean_structure
        return mean_structure, aligned_positions

    @staticmethod
    def compute_rmsf(positions_list: List[np.ndarray]) -> np.ndarray:
        """Compute per-atom RMSF over trajectory."""
        if not positions_list:
            return np.array([])
        arr = np.stack(positions_list, axis=0)
        mean_pos = np.mean(arr, axis=0)
        diff = arr - mean_pos
        rmsf = np.sqrt(np.mean(np.sum(diff * diff, axis=2), axis=0))
        return rmsf

    @staticmethod
    def calculate_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
        """计算两个坐标集之间的RMSD"""
        diff = coords1 - coords2
        return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))

    @staticmethod
    def calculate_group_rmsd(
        system_path: str,
        frame_indices: List[int],
        logger: Optional[logging.Logger] = None,
    ) -> np.ndarray:
        """为指定帧索引列表计算基于本组mean structure的RMSD序列

        Args:
            system_path: 系统目录路径
            frame_indices: 帧号列表（Frame_ID）
            logger: 可选的日志记录器

        Returns:
            RMSD序列数组
        """
        try:
            # 加载原子坐标数据
            stru_dir = os.path.join(system_path, "OUT.ABACUS", "STRU")
            if not os.path.exists(stru_dir):
                if logger:
                    logger.warning(f"STRU目录不存在: {stru_dir}")
                return np.array([])

            parser = StrUParser(exclude_hydrogen=True)
            all_frames = parser.parse_trajectory(stru_dir)

            if not all_frames:
                if logger:
                    logger.warning(f"未找到有效轨迹数据: {system_path}")
                return np.array([])

            # 按frame_id排序
            all_frames.sort(key=lambda x: x.frame_id)

            # 创建帧号到数组索引的映射
            frame_id_to_index = {
                frame.frame_id: idx for idx, frame in enumerate(all_frames)
            }

            # 将帧号映射到数组索引
            array_indices = []
            for frame_id in frame_indices:
                if frame_id in frame_id_to_index:
                    array_indices.append(frame_id_to_index[frame_id])
                else:
                    if logger:
                        logger.warning(
                            f"帧号 {frame_id} 在轨迹数据中不存在 (可用帧号范围: {min(frame_id_to_index.keys()) if frame_id_to_index else 'N/A'} - {max(frame_id_to_index.keys()) if frame_id_to_index else 'N/A'})"
                        )

            if len(array_indices) < 2:
                if logger:
                    logger.warning(f"有效帧数不足: {len(array_indices)}")
                return np.array([])

            # 提取指定帧的positions
            selected_positions = [
                all_frames[idx].positions.copy() for idx in array_indices
            ]

            # 计算本组的mean structure
            mean_structure, aligned_positions = RMSDCalculator.iterative_mean_structure(
                selected_positions, max_iter=20, tol=1e-6
            )

            # 计算每帧到mean structure的RMSD
            rmsds = []
            for pos in aligned_positions:
                rmsd = RMSDCalculator.calculate_rmsd(pos, mean_structure)
                rmsds.append(rmsd)

            return np.array(rmsds, dtype=float)

        except Exception as e:
            if logger:
                logger.error(f"计算组RMSD时出错: {e}")
            return np.array([])


class PCAReducer:
    """PCA降维处理器类，支持按累计方差贡献率降维"""

    def __init__(self, pca_variance_ratio: float = 0.90):
        self.pca_variance_ratio = pca_variance_ratio

    def apply_pca_reduction(
        self, vector_matrix: np.ndarray
    ) -> Tuple[np.ndarray, Optional[object]]:
        """应用PCA降维到指定累计方差贡献率"""
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            raise ImportError(
                "scikit-learn is required for PCA functionality. Please install it with: pip install scikit-learn"
            )

        try:
            # 使用相关系数矩阵PCA，不使用白化
            pca = PCA(
                n_components=self.pca_variance_ratio, whiten=False, random_state=42
            )
            reduced = pca.fit_transform(vector_matrix)

            return reduced, pca
        except Exception as e:
            logger = logging.getLogger(__name__)
            ErrorHandler.log_detailed_error(
                logger,
                e,
                "PCA降维过程中出错",
                additional_info={
                    "输入矩阵形状": (
                        vector_matrix.shape
                        if hasattr(vector_matrix, "shape")
                        else "未知"
                    ),
                    "PCA累计方差贡献率": self.pca_variance_ratio,
                    "矩阵类型": type(vector_matrix).__name__,
                },
            )


class SystemAnalyser:
    def __init__(
        self,
        include_hydrogen: bool = False,
        sample_ratio: float = 0.1,
        power_p: float = 0.5,
        pca_variance_ratio: float = 0.90,
    ):
        self.include_hydrogen = include_hydrogen
        self.sample_ratio = sample_ratio
        self.power_p = power_p
        self.pca_variance_ratio = pca_variance_ratio
        self.parser = StrUParser(exclude_hydrogen=not include_hydrogen)
        self.pca_reducer = PCAReducer(pca_variance_ratio)
        self.logger = logging.getLogger(__name__)

    def _extract_sampled_frames(
        self,
        frames: List,
        comprehensive_matrix: np.ndarray,
        pre_sampled_frames: Optional[List[int]],
        metrics: TrajectoryMetrics,
    ) -> Tuple[List[int], int]:
        """统一提取采样帧的逻辑

        Args:
            frames: 帧列表
            comprehensive_matrix: 综合向量矩阵
            pre_sampled_frames: 预采样帧列表
            metrics: 度量对象

        Returns:
            (sampled_indices, swap_count)元组
        """
        swap_count = 0

        if pre_sampled_frames is not None and len(pre_sampled_frames) > 0:
            # 复用采样结果：保持输入顺序过滤存在的帧
            existing_ids = {f.frame_id: idx for idx, f in enumerate(frames)}
            sampled_indices = [
                existing_ids[fid] for fid in pre_sampled_frames if fid in existing_ids
            ]
            if not sampled_indices:
                # 回退到重新采样
                pre_sampled_frames = None
            else:
                metrics.sampled_frames = [frames[i].frame_id for i in sampled_indices]
                return sampled_indices, swap_count

        if pre_sampled_frames is None or (
            pre_sampled_frames is not None and len(pre_sampled_frames) == 0
        ):
            k = max(2, int(round(self.sample_ratio * metrics.num_frames)))
            if k < metrics.num_frames:
                sampled_indices, swap_count, _ = PowerMeanSampler.select_frames(
                    comprehensive_matrix, k, p=self.power_p
                )
                metrics.sampled_frames = [frames[i].frame_id for i in sampled_indices]
            else:
                sampled_indices = list(range(len(frames)))
                metrics.sampled_frames = [f.frame_id for f in frames]

        return sampled_indices, swap_count

    def analyse_system(
        self,
        system_dir: str,
        pre_sampled_frames: Optional[List[int]] = None,
        pre_stru_files: Optional[List[str]] = None,
    ) -> Optional[Tuple]:
        system_info = self._extract_system_info(system_dir)
        if not system_info:
            return None
        system_name, mol_id, conf, temperature = system_info
        stru_dir = os.path.join(system_dir, "OUT.ABACUS", "STRU")
        if not os.path.exists(stru_dir):
            self.logger.warning(f"STRU目录不存在: {stru_dir}")
            return None
        frames = self.parser.parse_trajectory(stru_dir, pre_files=pre_stru_files)
        if ValidationUtils.is_empty(frames):
            self.logger.warning(f"未找到有效轨迹数据: {system_dir}")
            return None
        metrics = TrajectoryMetrics(system_name, mol_id, conf, temperature, system_dir)
        metrics.num_frames = len(frames)
        distance_vectors = []
        for frame in frames:
            dist_vec = MetricCalculator.calculate_distance_vectors(frame.positions)
            # treat None or empty numpy arrays as empty
            if ValidationUtils.is_empty(dist_vec):
                continue
            distance_vectors.append(dist_vec)
        if ValidationUtils.is_empty(distance_vectors):
            self.logger.warning(f"无法计算距离向量: {system_dir}")
            return None
        min_dim = min(len(vec) for vec in distance_vectors)
        # Only keep vectors that have at least min_dim elements, and truncate them to min_dim
        vector_matrix = [
            vec[:min_dim]
            for vec in distance_vectors
            if not ValidationUtils.is_empty(vec) and len(vec) >= min_dim
        ]
        if ValidationUtils.is_empty(vector_matrix):
            self.logger.warning(f"距离向量维度不一致: {system_dir}")
            return None

        vector_matrix = np.array(vector_matrix)

        # 应用PCA降维
        reduced_matrix, pca_model = self.pca_reducer.apply_pca_reduction(vector_matrix)
        # 设置PCA相关字段
        metrics.pca_variance_ratio = float(self.pca_variance_ratio)
        if pca_model is not None:
            metrics.pca_components = pca_model.n_components_
        else:
            metrics.pca_components = 0

        # 构建包含能量与PCA分量的综合向量（不做磁盘缓存，重算成本低）
        comprehensive_matrix, pca_components_data = self.build_comprehensive_vectors(
            frames, reduced_matrix
        )

        # 设置综合向量相关字段

        # 在综合向量空间中计算指标
        original_metrics = MetricCalculator.compute_all_metrics(comprehensive_matrix)
        metrics.set_original_metrics(original_metrics)

        # pca_components_data 已为处理后的构象分量（与综合向量一致的缩放）

        # 统一采样逻辑
        sampled_indices, swap_count = self._extract_sampled_frames(
            frames, comprehensive_matrix, pre_sampled_frames, metrics
        )

        # 计算采样后的指标
        sampled_vectors = (
            comprehensive_matrix[sampled_indices]
            if len(sampled_indices) > 0
            else comprehensive_matrix
        )

        # 计算采样帧的RMSD值
        sampled_frame_ids = (
            [frames[i].frame_id for i in sampled_indices] if sampled_indices else []
        )
        sampled_rmsd = []
        if sampled_frame_ids:
            try:
                sampled_rmsd = RMSDCalculator.calculate_group_rmsd(
                    system_dir, sampled_frame_ids, self.logger
                )
            except Exception as e:
                self.logger.warning(f"RMSD计算失败: {str(e)}")
                sampled_rmsd = []

        from ..core.metrics import MetricsToolkit

        sampled_metrics = MetricsToolkit.adapt_sampling_metrics(
            sampled_vectors, comprehensive_matrix, sampled_rmsd
        )
        metrics.set_sampled_metrics(sampled_metrics)

        return metrics, frames, swap_count, pca_components_data, []

    def analyse_system_sampling_only(
        self, system_dir: str, pre_sampled_frames: Optional[List[int]] = None
    ) -> Optional[Tuple]:
        """仅执行采样算法的轻量级分析版本，不计算各项统计指标"""
        system_info = self._extract_system_info(system_dir)
        if not system_info:
            return None
        system_name, mol_id, conf, temperature = system_info
        stru_dir = os.path.join(system_dir, "OUT.ABACUS", "STRU")
        if not os.path.exists(stru_dir):
            self.logger.warning(f"STRU目录不存在: {stru_dir}")
            return None

        frames = self.parser.parse_trajectory(stru_dir)
        if ValidationUtils.is_empty(frames):
            self.logger.warning(f"未找到有效轨迹数据: {system_dir}")
            return None

        # 创建简化的metrics对象，只包含基本信息
        metrics = TrajectoryMetrics(system_name, mol_id, conf, temperature, system_dir)
        metrics.num_frames = len(frames)

        # 计算距离向量用于采样
        distance_vectors = []
        for frame in frames:
            dist_vec = MetricCalculator.calculate_distance_vectors(frame.positions)
            if ValidationUtils.is_empty(dist_vec):
                continue
            distance_vectors.append(dist_vec)

        if ValidationUtils.is_empty(distance_vectors):
            self.logger.warning(f"无法计算距离向量: {system_dir}")
            return None

        min_dim = min(len(vec) for vec in distance_vectors)
        vector_matrix = [
            vec[:min_dim]
            for vec in distance_vectors
            if not ValidationUtils.is_empty(vec) and len(vec) >= min_dim
        ]
        if ValidationUtils.is_empty(vector_matrix):
            self.logger.warning(f"距离向量维度不一致: {system_dir}")
            return None

        vector_matrix = np.array(vector_matrix)

        # 应用PCA降维（用于采样）
        reduced_matrix, pca_model = self.pca_reducer.apply_pca_reduction(vector_matrix)

        # 构建简化的综合向量（仅包含能量和PCA）
        comprehensive_matrix, _ = self.build_comprehensive_vectors(
            frames, reduced_matrix
        )

        # 统一采样逻辑
        sampled_indices, swap_count = self._extract_sampled_frames(
            frames, comprehensive_matrix, pre_sampled_frames, metrics
        )

        # 返回简化的结果：只包含metrics和frames
        return metrics, frames

    def build_comprehensive_vectors(
        self, frames: List, reduced_matrix: np.ndarray
    ) -> Tuple[np.ndarray, List[dict]]:
        """构建包含能量和PCA分量的综合向量，进行中心化和标准化缩放。

        归一化流程：
        1. 中心化：每个分量减去该分量的均值
        2. 缩放：按 sqrt(2*方差) 进行缩放，确保总方差=1

        具体规则：
        - 能量：E_centered = E - mean(E)，然后 E_proc = E_centered / sqrt(2*Var(E_centered))
        - PCA：PC_centered = PC - mean(PC)，然后 PC_proc = PC_centered / sqrt(2*Sum(Var(PC_centered)))

        返回：
        - comprehensive_matrix: [n_valid, 1+dim]，首列 E_proc，后续列为 PC_proc。
        - pca_components_data: 每帧的处理后 PC 分量字典（用于 CSV 保存）。
        """
        comprehensive_vectors: List[np.ndarray] = []
        pca_components_data: List[dict] = []

        # 收集所有有效帧的能量数据
        energies: List[float] = []
        valid_frames: List[int] = []
        for i, frame in enumerate(frames):
            if i < len(reduced_matrix) and getattr(frame, "energy", None) is not None:
                energies.append(float(frame.energy))
                valid_frames.append(i)

        if not valid_frames:
            self.logger.warning("没有找到包含能量数据的帧，使用原始PCA向量")
            return reduced_matrix, []

        # 能量 z-score 并缩放到目标方差 0.5
        energies_arr = np.array(energies, dtype=float)
        energy_mean = float(np.mean(energies_arr))
        energy_std = float(np.std(energies_arr))
        if energy_std > 0:
            energies_z = (energies_arr - energy_mean) / energy_std
        else:
            energies_z = energies_arr - energy_mean
        var_energy = float(np.var(energies_z)) if energies_z.size > 0 else 0.0
        if var_energy > 0:
            energy_scale = 1.0 / np.sqrt(2.0 * var_energy)
            energies_processed = energies_z * energy_scale
        else:
            energies_processed = np.zeros_like(energies_z)

        # PCA 中心化和方差计算
        pca_centered_matrix = None
        if len(valid_frames) > 0 and getattr(reduced_matrix, "ndim", 0) == 2:
            pca_matrix_valid = reduced_matrix[valid_frames]
            # 明确中心化：每个分量减去其均值
            pca_means = np.mean(pca_matrix_valid, axis=0)
            pca_centered_matrix = pca_matrix_valid - pca_means
            # 计算中心化后数据的方差
            pca_variances = (
                np.var(pca_centered_matrix, axis=0)
                if pca_centered_matrix.size > 0
                else np.array([])
            )
            pca_total_variance = (
                float(np.sum(pca_variances)) if pca_variances.size > 0 else 0.0
            )
        else:
            pca_total_variance = 0.0
        if pca_total_variance > 0:
            pca_scale = 1.0 / np.sqrt(2.0 * pca_total_variance)
        else:
            pca_scale = 0.0

        # 构建综合向量与导出数据
        for idx, frame_idx in enumerate(valid_frames):
            frame = frames[frame_idx]
            # 处理后的能量
            frame.energy_standardized = float(energies_processed[idx])

            # 处理后的 PCA 分量
            if pca_centered_matrix is not None:
                # 使用中心化后的PCA分量
                centered_pcs = pca_centered_matrix[idx]
                pca_components = centered_pcs * pca_scale
            else:
                # 降级处理：使用原始PCA分量
                raw_pcs = reduced_matrix[frame_idx]
                pca_components = raw_pcs * pca_scale

            combined_vector = np.concatenate(
                ([energies_processed[idx]], pca_components.astype(float))
            )
            comprehensive_vectors.append(combined_vector)

            item = {
                "system": (frame.system_name if hasattr(frame, "system_name") else ""),
                "frame": frame.frame_id,
            }
            for pc_num, val in enumerate(pca_components, start=1):
                item[f"PC{pc_num}"] = float(val)
            pca_components_data.append(item)

        comprehensive_matrix = np.array(comprehensive_vectors, dtype=float)
        return comprehensive_matrix, pca_components_data

    def _extract_system_info(
        self, system_dir: str
    ) -> Optional[Tuple[str, str, str, str]]:
        system_name = os.path.basename(system_dir)
        match = re.match(r"struct_mol_(\d+)_conf_(\d+)_T(\d+)K", system_name)
        if not match:
            self.logger = logging.getLogger(__name__)
            self.logger.warning(f"目录名格式不正确: {system_name}")
            return None
        mol_id, conf, temperature = match.groups()
        return system_name, mol_id, conf, temperature
