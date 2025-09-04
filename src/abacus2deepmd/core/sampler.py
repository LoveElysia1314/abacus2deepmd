#!/usr/bin/env python

from typing import List, Tuple

import numpy as np

from ..utils.common import CommonUtils


class MathUtils:
    """Mathematical utility functions"""

    @staticmethod
    def power_mean(values: np.ndarray, p: float, default: float = 0.0) -> float:
        """Calculate power mean of values

        Args:
            values: Input array of values
            p: Power parameter
            default: Default value to return for invalid inputs

        Returns:
            Power mean value
        """
        if len(values) == 0:
            return default

        if abs(p) < 0.05:  # Treat p close to 0 as geometric mean
            # Geometric mean for p=0
            return np.exp(np.mean(np.log(values + 1e-15)))
        else:
            return np.power(np.mean(np.power(values, p)), 1.0 / p)

    # Import safe_divide from CommonUtils to avoid duplication
    safe_divide = CommonUtils.safe_divide

    # Constants for numerical stability
    DISTANCE_EPSILON = 1e-12


class PowerMeanSampler:
    @staticmethod
    def select_frames(
        points: np.ndarray,
        k: int,
        p: float = 0.5,
    ) -> Tuple[List[int], int, float]:
        n = len(points)
        if k >= n:
            return list(range(n)), 0, 0.0
        selected = PowerMeanSampler._initialize_seeds(points, k)
        selected = PowerMeanSampler._incremental_selection(points, selected, k, p)
        selected, swap_count, improve_ratio = PowerMeanSampler._swap_optimization(
            points, selected, p
        )
        return selected, swap_count, improve_ratio

    @staticmethod
    def _initialize_seeds(points: np.ndarray, k: int) -> List[int]:
        """Initialize seeds for greedy selection.

        Select the point farthest from the origin (mean) as the initial seed.
        Since the comprehensive vectors are standardized, the mean is at origin.

        Args:
            points: Input points array (standardized comprehensive vectors)
            k: Number of points to select

        Returns:
            List of initial seed indices
        """
        # Calculate distances from origin (L2 norm for each point)
        distances_from_origin = np.linalg.norm(points, axis=1)
        # Select the point with maximum distance from origin
        first_seed = np.argmax(distances_from_origin)
        return [first_seed]

    @staticmethod
    def _incremental_selection(
        points: np.ndarray, selected: List[int], k: int, p: float
    ) -> List[int]:
        n = len(points)
        if n == 0:
            return selected

        X = np.asarray(points, dtype=np.float64)
        n = X.shape[0]

        # 候选集合（未选索引），使用 numpy 数组便于切片
        all_idx = np.arange(n, dtype=np.int64)
        if len(selected) > 0:
            mask_init = np.ones(n, dtype=bool)
            mask_init[selected] = False
            candidates = all_idx[mask_init]
        else:
            candidates = all_idx.copy()

        # 预计算每个点的范数平方，用于快速计算 d2 = ||x||^2 + ||s||^2 - 2 x·s
        norms_sq = np.einsum("ij,ij->i", X, X)
        eps = MathUtils.DISTANCE_EPSILON
        eps2 = eps * eps

        # 维护累计值，只对候选索引维护（和 candidates 对齐的一维数组）
        if abs(p) < 0.05:  # Treat p close to 0 as geometric mean
            sum_acc_cand = np.zeros(
                candidates.size, dtype=np.float64
            )  # 存放 0.5*Σ log(d2)
        else:
            sum_acc_cand = np.zeros(
                candidates.size, dtype=np.float64
            )  # 存放 Σ d2^(p/2)

        # 使用当前已选点进行初始化累计
        for s in selected:
            s_norm = norms_sq[s]
            # 使用全量 GEMV 再切片，避免 X[candidates] 拷贝
            x_dot_s_all = X.dot(X[s])
            x_dot_s = x_dot_s_all[candidates]
            d2 = norms_sq[candidates] + s_norm - 2.0 * x_dot_s
            d2 = np.maximum(d2, eps2)
            if abs(p) < 0.05:  # Treat p close to 0 as geometric mean
                sum_acc_cand += 0.5 * np.log(d2 + 1e-30)
            else:
                sum_acc_cand += np.power(d2, 0.5 * p)

        # 计算自动批大小（经验公式）：batch = clamp(round(sqrt(n/k)), 4, 64)
        # 对于很小的 k，后续会再用剩余量进行限制，确保安全。
        denom = max(k, 1)
        batch_auto = int(round(np.sqrt(n / denom)))
        batch_auto = max(4, min(64, batch_auto))

        # 主循环：精确微批，每批内用哨兵屏蔽，批末一次性压缩
        while len(selected) < k and candidates.size > 0:
            remaining_k = k - len(selected)
            if remaining_k <= 0:
                break
            # 当前批大小受剩余量与候选数限制
            batch_size = min(batch_auto, remaining_k, max(1, candidates.size))

            # 批内：维护一个布尔存活掩码，标记本批内已选的候选位置
            alive_mask = np.ones(candidates.size, dtype=bool)
            # 哨兵值：p>=0 用 -inf，使其不会被再次选中；p<0 用 +inf。
            sentinel = -np.inf if p >= 0 else np.inf

            picks = 0
            while picks < batch_size and len(selected) < k and alive_mask.any():
                # 仅在 alive 上进行选择
                # 为了避免分支，直接在 sum_acc_cand 上选择并借助活跃掩码控制
                # 临时将非活跃位置设为哨兵，选择后再恢复（更快可行）
                tmp = sum_acc_cand.copy()
                tmp[~alive_mask] = sentinel
                best_local = int(np.argmin(tmp)) if p < 0 else int(np.argmax(tmp))
                best_idx = int(candidates[best_local])

                # 记录选中并屏蔽该位置
                selected.append(best_idx)
                alive_mask[best_local] = False
                picks += 1

                # 增量更新：仅对仍存活的候选进行
                if len(selected) <= k and alive_mask.any():
                    s = best_idx
                    s_norm = norms_sq[s]
                    x_dot_s_all = X.dot(X[s])
                    x_dot_s = x_dot_s_all[candidates]
                    d2 = norms_sq[candidates] + s_norm - 2.0 * x_dot_s
                    d2 = np.maximum(d2, eps2)
                    if abs(p) < 0.05:  # Treat p close to 0 as geometric mean
                        sum_acc_cand[alive_mask] += 0.5 * np.log(d2[alive_mask] + 1e-30)
                    else:
                        sum_acc_cand[alive_mask] += np.power(d2[alive_mask], 0.5 * p)

            # 批末：压缩候选与累计数组，仅保留仍存活的候选
            if alive_mask.size > 0 and not alive_mask.all():
                candidates = candidates[alive_mask]
                sum_acc_cand = sum_acc_cand[alive_mask]

        return selected

    @staticmethod
    def _swap_optimization(
        points: np.ndarray, selected: List[int], p: float
    ) -> Tuple[List[int], int, float]:
        """
        交换优化阶段，去除 cdist，采用向量化实现，并复用选点阶段已计算的距离和累计数据。
        """
        if len(selected) < 2:
            return selected, 0, 0.0
        n = len(points)
        selected = selected.copy()
        not_selected = np.setdiff1d(np.arange(n), selected)
        X = np.asarray(points, dtype=np.float64)
        sel_idx = np.array(selected, dtype=np.int64)
        sel_points = X[sel_idx]
        norms_sq = np.einsum("ij,ij->i", X, X)
        eps = MathUtils.DISTANCE_EPSILON
        eps2 = eps * eps

        # 计算已选点两两距离（向量化）
        sel_dot = sel_points @ sel_points.T
        sel_norms = norms_sq[sel_idx]
        d2_mat = sel_norms[:, None] + sel_norms[None, :] - 2.0 * sel_dot
        np.fill_diagonal(d2_mat, np.inf)
        pair_dists = np.sqrt(np.maximum(d2_mat, eps2)).flatten()
        pair_dists = pair_dists[np.isfinite(pair_dists)]
        initial_obj = MathUtils.power_mean(pair_dists, p)
        swap_count = 0
        improved = True

        # 预计算所有点与已选点的距离平方（n x k）
        dot_all_sel = X @ sel_points.T
        d2_all_sel = norms_sq[:, None] + sel_norms[None, :] - 2.0 * dot_all_sel
        d2_all_sel = np.maximum(d2_all_sel, eps2)

        while improved:
            improved = False
            best_improvement = 0.0
            best_swap = None
            # 枚举已选点 i_idx, i
            for i_idx, i in enumerate(sel_idx):
                # 取 i 与其他已选点的距离
                d2_i_others = d2_all_sel[i, np.arange(len(sel_idx)) != i_idx]
                dist_i = np.sqrt(d2_i_others)
                old_contrib = MathUtils.power_mean(dist_i, p)
                # 枚举未选点 j
                for j in not_selected:
                    d2_j_others = d2_all_sel[j, np.arange(len(sel_idx)) != i_idx]
                    dist_j = np.sqrt(d2_j_others)
                    new_contrib = MathUtils.power_mean(dist_j, p)
                    improvement = new_contrib - old_contrib
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_swap = (i_idx, j)
            if best_swap and best_improvement > MathUtils.DISTANCE_EPSILON:
                i_idx, j = best_swap
                old_point = sel_idx[i_idx]
                sel_idx[i_idx] = j
                not_selected = np.setdiff1d(not_selected, [j])
                not_selected = np.append(not_selected, old_point)
                # 更新 sel_points, dot_all_sel, d2_all_sel
                sel_points = X[sel_idx]
                sel_norms = norms_sq[sel_idx]
                dot_all_sel = X @ sel_points.T
                d2_all_sel = norms_sq[:, None] + sel_norms[None, :] - 2.0 * dot_all_sel
                d2_all_sel = np.maximum(d2_all_sel, eps2)
                swap_count += 1
                improved = True
        # 重新计算最终目标值提升比率
        sel_dot = sel_points @ sel_points.T
        d2_mat = sel_norms[:, None] + sel_norms[None, :] - 2.0 * sel_dot
        np.fill_diagonal(d2_mat, np.inf)
        pair_dists = np.sqrt(np.maximum(d2_mat, eps2)).flatten()
        pair_dists = pair_dists[np.isfinite(pair_dists)]
        final_obj = MathUtils.power_mean(pair_dists, p)
        improve_ratio = (
            (final_obj - initial_obj) / initial_obj if initial_obj > 0 else 0.0
        )
        return sel_idx.tolist(), swap_count, improve_ratio


class SamplingStrategy:
    @staticmethod
    def uniform_sample_indices(n: int, k: int) -> np.ndarray:
        """Generate uniformly spaced indices for sampling.

        Args:
            n: Total number of items
            k: Number of samples to select

        Returns:
            Array of selected indices
        """
        if k >= n:
            return np.arange(n)
        return np.round(np.linspace(0, n - 1, k)).astype(int)


# Statistical analysis utilities (merged from math_utils.py)
def calculate_improvement(sample_val: float, baseline_mean: float) -> float:
    """Calculate percentage improvement over baseline.

    Args:
        sample_val: Sample value
        baseline_mean: Baseline mean value
        baseline_sem: Baseline standard error (unused in current implementation)

    Returns:
        Improvement percentage
    """
    if np.isnan(sample_val) or np.isnan(baseline_mean) or baseline_mean == 0:
        return np.nan
    improvement = (sample_val - baseline_mean) / abs(baseline_mean) * 100
    return improvement
