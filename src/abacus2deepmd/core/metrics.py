#!/usr/bin/env python

from typing import Dict, List, Any, Callable, Sequence, Optional, Union
from dataclasses import dataclass
import json

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from sklearn.decomposition import PCA


# Data classes for metrics
@dataclass
class BasicDistanceMetrics:
    ANND: float
    MPD: float


@dataclass
class DiversityMetrics:
    total_variance: float
    pca_variance_ratio: float
    energy_coverage_ratio: float


@dataclass
class DistributionSimilarity:
    js_divergence: float


# Metrics registry classes
@dataclass
class MetricSpec:
    key: str
    category: str  # identity | scale | core_distance | diversity | distribution | pca
    formatter: Optional[Callable[[Any], str]] = None  # Optional formatter


# Schema version
SCHEMA_VERSION = "summary-v2"

# Group order
GROUP_ORDER = [
    "identity",  # Basic identification
    "scale",  # Scale/dimension
    "core_distance",  # Core structural distance metrics
    "diversity",  # Diversity & coverage & energy
    "distribution",  # Distribution / sampling similarity
    "pca",  # PCA overview
]

# Formatting utilities
_float6 = lambda v: (
    ""
    if v is None
    else (f"{float(v):.6f}" if not (isinstance(v, float) and v != v) else "")
)
_int = lambda v: "" if v is None else str(int(v))
_json_list = lambda v: json.dumps(v, ensure_ascii=False) if v is not None else ""
_passthrough = lambda v: "" if v is None else str(v)

# Metrics registry
REGISTRY: List[MetricSpec] = [
    # identity
    MetricSpec("system", "identity", _passthrough),
    MetricSpec("mol_id", "identity", _passthrough),
    MetricSpec("conf", "identity", _passthrough),
    MetricSpec("temperature", "identity", _passthrough),
    # scale
    MetricSpec("num_frames", "scale", _int),
    # core distance
    MetricSpec("rmsd_mean", "core_distance", _float6),
    MetricSpec("ANND", "core_distance", _float6),
    MetricSpec("MPD", "core_distance", _float6),
    # diversity & energy
    MetricSpec("total_variance", "diversity", _float6),
    MetricSpec("energy_coverage_ratio", "diversity", _float6),
    # distribution similarity (order swapped before PCA)
    MetricSpec("js_divergence", "distribution", _float6),
    # PCA
    MetricSpec("pca_components", "pca", _int),
    MetricSpec("pca_variance_ratio", "pca", _float6),
    MetricSpec("pca_cumulative_variance_ratio", "pca", _float6),
]

# Pre-built: group -> specs list
_GROUP_TO_SPECS: Dict[str, List[MetricSpec]] = {g: [] for g in GROUP_ORDER}
for spec in REGISTRY:
    _GROUP_TO_SPECS.setdefault(spec.category, []).append(spec)


class MetricsToolkit:
    """Unified static methods for metric calculations."""

    # ---- Basic distance metrics ----
    @staticmethod
    def compute_basic_distance_metrics(vectors: np.ndarray) -> BasicDistanceMetrics:
        if vectors is None or len(vectors) < 2:
            return BasicDistanceMetrics(np.nan, np.nan)
        try:
            # Reuse logic similar to sampling_compare_demo but centralized
            dists = squareform(pdist(vectors, metric="euclidean"))
            np.fill_diagonal(dists, np.inf)
            valid = dists[dists != np.inf]
            if len(valid) == 0:
                return BasicDistanceMetrics(np.nan, np.nan)
            annd = (
                float(np.mean(np.min(dists, axis=1)))
                if np.any(np.isfinite(np.min(dists, axis=1)))
                else np.nan
            )
            mpd = float(np.mean(valid)) if len(valid) else np.nan
            return BasicDistanceMetrics(annd, mpd)
        except Exception:
            return BasicDistanceMetrics(np.nan, np.nan)

    # ---- Diversity metrics ----
    @staticmethod
    def compute_diversity_metrics(
        vectors: np.ndarray, full_vectors: np.ndarray = None
    ) -> DiversityMetrics:
        if vectors is None or len(vectors) < 2:
            return DiversityMetrics(np.nan, np.nan, np.nan)
        # Drop rows with any NaN
        if np.any(np.isnan(vectors)):
            mask = ~np.any(np.isnan(vectors), axis=1)
            vectors = vectors[mask]
            if len(vectors) < 2:
                return DiversityMetrics(np.nan, np.nan, np.nan)
        try:
            # Direct variance sum coverage (since total variance is normalized to 1)
            variances = np.var(vectors, axis=0)
            total_variance = float(np.sum(variances))
            pca_variance_ratio = total_variance  # For compatibility
            # Energy coverage ratio (range_S / range_X)
            energy_coverage_ratio = np.nan
            if (
                vectors.shape[1] > 0
                and full_vectors is not None
                and full_vectors.shape[1] > 0
            ):
                energy_col_s = vectors[:, 0]
                energy_col_x = full_vectors[:, 0]
                if not np.all(np.isnan(energy_col_s)) and not np.all(
                    np.isnan(energy_col_x)
                ):
                    valid_s = energy_col_s[~np.isnan(energy_col_s)]
                    valid_x = energy_col_x[~np.isnan(energy_col_x)]
                    if len(valid_s) > 0 and len(valid_x) > 0:
                        range_s = np.ptp(valid_s)
                        range_x = np.ptp(valid_x)
                        energy_coverage_ratio = (
                            range_s / range_x if range_x > 0 else np.nan
                        )
            return DiversityMetrics(
                total_variance, pca_variance_ratio, energy_coverage_ratio
            )
        except Exception:
            return DiversityMetrics(np.nan, np.nan, np.nan)

    # ---- Distribution similarity ----
    @staticmethod
    def compute_distribution_similarity(
        sample_vectors: np.ndarray, full_vectors: np.ndarray
    ) -> DistributionSimilarity:
        if (
            sample_vectors is None
            or full_vectors is None
            or len(sample_vectors) < 2
            or len(full_vectors) < 2
        ):
            return DistributionSimilarity(np.nan)
        # Clean NaNs
        if np.any(np.isnan(sample_vectors)):
            sample_vectors = sample_vectors[~np.any(np.isnan(sample_vectors), axis=1)]
        if np.any(np.isnan(full_vectors)):
            full_vectors = full_vectors[~np.any(np.isnan(full_vectors), axis=1)]
        if len(sample_vectors) < 2 or len(full_vectors) < 2:
            return DistributionSimilarity(np.nan)
        try:
            pca = PCA(n_components=min(3, sample_vectors.shape[1]))
            sample_pca = pca.fit_transform(sample_vectors)
            full_pca = pca.transform(full_vectors)
            js_components = []
            for i in range(sample_pca.shape[1]):
                s_hist, _ = np.histogram(sample_pca[:, i], bins=20, density=True)
                f_hist, _ = np.histogram(full_pca[:, i], bins=20, density=True)
                s_hist = s_hist / (np.sum(s_hist) + 1e-10)
                f_hist = f_hist / (np.sum(f_hist) + 1e-10)
                m = 0.5 * (s_hist + f_hist)
                js = 0.5 * (
                    entropy(s_hist + 1e-10, m + 1e-10)
                    + entropy(f_hist + 1e-10, m + 1e-10)
                )
                js_components.append(js)
            js_divergence = float(np.mean(js_components))
            return DistributionSimilarity(js_divergence)
        except Exception:
            return DistributionSimilarity(np.nan)

    # ---- RMSD summary ----
    @staticmethod
    def summarize_rmsd(values: Sequence[float]) -> float:
        arr = np.array(values, dtype=float)
        if arr.size == 0 or np.all(np.isnan(arr)):
            return np.nan
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            return np.nan
        return float(np.mean(valid))

    @staticmethod
    def adapt_sampling_metrics(
        selected_vectors: np.ndarray,
        full_vectors: np.ndarray,
        rmsd_values: Optional[Union[np.ndarray, List[float]]] = None,
    ) -> Dict[str, Any]:
        """Adapt sampling metrics for comparison analysis."""
        try:
            basic = MetricsToolkit.compute_basic_distance_metrics(selected_vectors)
            diversity = MetricsToolkit.compute_diversity_metrics(
                selected_vectors, full_vectors
            )
            similarity = MetricsToolkit.compute_distribution_similarity(
                selected_vectors, full_vectors
            )
            rmsd_mean = MetricsToolkit.summarize_rmsd(
                rmsd_values if rmsd_values is not None else []
            )
            return {
                "ANND": basic.ANND,
                "MPD": basic.MPD,
                "Total_Variance": diversity.total_variance,
                "Energy_Coverage_Ratio": diversity.energy_coverage_ratio,
                "JS_Divergence": similarity.js_divergence,
                "RMSD_Mean": rmsd_mean,
            }
        except Exception:
            return {}


# Registry functions


class MetricCalculator:
    @staticmethod
    def calculate_distance_vectors(positions: np.ndarray) -> np.ndarray:
        from .system_analyser import ValidationUtils

        # positions is expected to be a 2D numpy array; be robust if it's None or too short
        if (
            ValidationUtils.is_empty(positions)
            or getattr(positions, "shape", [0])[0] < 2
        ):
            return np.array([])
        # Remove L2 normalization step, directly return raw distance vectors
        return pdist(positions)

    @staticmethod
    def compute_all_metrics(vector_matrix: np.ndarray) -> Dict[str, object]:
        from .system_analyser import ValidationUtils

        """Unified calculation of distance-related metrics.

        Refactored to call MetricsToolkit.compute_basic_distance_metrics to eliminate duplication.
        Maintains original return structure for backward compatibility.
        """
        if (
            ValidationUtils.is_empty(vector_matrix)
            or getattr(vector_matrix, "shape", [0])[0] < 2
        ):
            return {"global_mean": 0.0, "ANND": 0.0, "MPD": 0.0}

        global_mean = float(np.mean(vector_matrix))
        basic = MetricsToolkit.compute_basic_distance_metrics(vector_matrix)
        return {
            "global_mean": global_mean,
            "ANND": basic.ANND,
            "MPD": basic.MPD,
        }


class TrajectoryMetrics:
    def __init__(
        self,
        system_name: str,
        mol_id: str,
        conf: str,
        temperature: str,
        system_path: str = "",
    ):
        self.system_name = system_name
        self.mol_id = mol_id
        self.conf = conf
        self.temperature = temperature
        self.system_path = system_path  # Add system path
        self.num_frames = 0
        self.ANND = 0.0
        self.MPD = 0.0
        self.sampled_frames = []
        # RMSD related fields
        self.rmsd_mean = 0.0  # RMSD mean (overall metric)
        # PCA related fields
        self.pca_variance_ratio = 0.0  # PCA target variance contribution rate
        self.pca_components = 0  # PCA principal component count
        # Comprehensive vector related fields
        # Average conformation coordinates (for subsequent conformation unification)
        self.mean_structure = (
            None  # Average conformation coordinates, numpy array shape (n_atoms, 3)
        )
        # Level 4: Diversity / distribution similarity extension fields (optional output)
        self.total_variance = None
        self.energy_coverage_ratio = None
        self.js_divergence = None

    def set_original_metrics(self, metrics_data: Dict[str, float]):
        self.ANND = metrics_data["ANND"]
        self.MPD = metrics_data["MPD"]

    def set_sampled_metrics(self, metrics_data: Dict[str, float]):
        """Set metrics data after sampling"""
        self.ANND = metrics_data.get("ANND", 0.0)
        self.MPD = metrics_data.get("MPD", 0.0)
        self.total_variance = metrics_data.get("Total_Variance", np.nan)
        self.energy_coverage_ratio = metrics_data.get("Energy_Coverage_Ratio", np.nan)
        self.js_divergence = metrics_data.get("JS_Divergence", np.nan)
        self.rmsd_mean = metrics_data.get("RMSD_Mean", 0.0)
