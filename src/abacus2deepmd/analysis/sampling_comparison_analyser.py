#!/usr/bin/env python
"""
Sampling effect comparison analyzer
Function: Compare the effects of intelligent sampling algorithm with random/uniform sampling
Refactored: Core utility functions migrated to unified modules, retaining business logic

Optimization features:
1. Dual statistics: Calculate both absolute and relative statistical data simultaneously
2. Relative value normalization: Use intelligent sampling as 1.0 baseline, convert random and uniform sampling metrics to ratios between 0-1
3. Eliminate system differences: Each system normalized independently, ensuring equal contribution of each system to final results
4. Enhanced visualization: Charts use relative value coordinates and error bars, labels display absolute value information
5. Data integrity: CSV files contain complete statistical data for both Abs (absolute values) and Rel (relative values)
"""

import os
import numpy as np
import pandas as pd
import json
import logging
import glob
from typing import List
import warnings
import matplotlib.pyplot as plt

from abacus2deepmd.utils import LoggerManager
from abacus2deepmd.core.metrics import MetricsToolkit
from abacus2deepmd.core.sampler import SamplingStrategy, calculate_improvement
from abacus2deepmd.core.system_analyser import RMSDCalculator

warnings.filterwarnings("ignore")


class SamplingComparisonAnalyser:
    """Sampling effect comparison analyzer class"""

    def __init__(self, error_log_file: str = None, error_logger: logging.Logger = None):
        self.logger = LoggerManager.create_logger(__name__)

        # Use the passed error logger, or create a new one
        if error_logger:
            self.error_logger = error_logger
        elif error_log_file:
            self.error_logger = LoggerManager.create_logger_with_error_log(
                name=f"{__name__}.errors",
                level=logging.ERROR,
                add_console=False,
                error_log_file=error_log_file,
                log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                date_format="%Y-%m-%d %H:%M:%S",
            )
        else:
            self.error_logger = None

    def _calculate_axis_limits(
        self, values, errors, margin_percent=0.05, min_limit=None
    ):
        """Calculate dynamic range of axis to ensure error bars are fully displayed

        Args:
            values: List of data values
            errors: List of error values
            margin_percent: Margin percentage
            min_limit: Minimum limit value (e.g., x-axis starts from 0)

        Returns:
            (min_limit, max_limit): Axis range
        """
        # Filter NaN values
        valid_values = [v for v in values if not pd.isna(v)]
        valid_errors = [e for e in errors if not pd.isna(e)]

        if not valid_values or not valid_errors:
            return (0, 1)  # Default range

        # Calculate boundaries
        data_min = min(valid_values)
        data_max = max(valid_values)
        max_error = max(valid_errors)

        axis_min = data_min - max_error
        axis_max = data_max + max_error

        # Apply minimum limit
        if min_limit is not None:
            axis_min = max(axis_min, min_limit)

        # Calculate margin
        data_range = axis_max - axis_min
        if data_range > 0:
            margin = margin_percent * data_range
            axis_min -= margin
            axis_max += margin

        return (axis_min, axis_max)

    def analyse_sampling_compare(
        self, result_dir=None, workers: int = -1, force_recompute: bool = False
    ):
        """Main function for sampling effect comparison analysis, output results by type"""
        # Auto-locate result directory
        if result_dir is None:
            dirs = sorted(glob.glob("analysis_results/run_*"), reverse=True)
            if not dirs:
                self.logger.warning(
                    "No analysis results found, please run the main program first."
                )
                return
            result_dir = dirs[0]

        # Load system path mapping
        targets_file = os.path.join(result_dir, "analysis_targets.json")
        system_paths = {}
        if os.path.exists(targets_file):
            try:
                with open(targets_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for mol_data in data.get("molecules", {}).values():
                        for sys_name, sys_data in mol_data.get("systems", {}).items():
                            system_paths[sys_name] = sys_data.get("system_path", "")
            except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
                self.logger.warning(f"Failed to load system path mapping: {e}")

        # Compatibility for new and old single system directory
        single_dir_candidates = [
            os.path.join(result_dir, "single_analysis"),
            os.path.join(result_dir, "single_analysis_results"),
        ]
        single_dir = None
        for c in single_dir_candidates:
            if os.path.isdir(c):
                single_dir = c
                break
        if single_dir is None:
            self.logger.warning("single_analysis directory not found")
            return
        # Compatibility for new and old file naming: frame_*.csv / frame_metrics_*.csv
        files = glob.glob(os.path.join(single_dir, "frame_*.csv")) + glob.glob(
            os.path.join(single_dir, "frame_metrics_*.csv")
        )

        if not files:
            self.logger.warning("Frame metrics files not found")
            return

        # Sort files by system name to ensure consistent processing order
        from ..utils.common import CommonUtils

        def get_system_name_from_file(f):
            base = os.path.basename(f)
            if base.startswith("frame_metrics_"):
                return base[len("frame_metrics_") : -4]
            elif base.startswith("frame_"):
                return base[len("frame_") : -4]
            else:
                return base

        files.sort(
            key=lambda f: CommonUtils.sort_system_names([get_system_name_from_file(f)])[
                0
            ]
        )

        # Create categorized output folder
        cache_dir = os.path.join(result_dir, "sampling_comparison")
        os.makedirs(cache_dir, exist_ok=True)

        # If force recompute, ignore cache
        if force_recompute:
            self.logger.info("Force recompute: ignore all cache, recalculate directly")
            sampled_cache = None
            random_cache = None
            uniform_cache = None
        else:
            # Read cached csv
            sampled_cache = None
            random_cache = None
            uniform_cache = None
            if os.path.exists(os.path.join(cache_dir, "sampled.csv")):
                sampled_cache = pd.read_csv(os.path.join(cache_dir, "sampled.csv"))
            if os.path.exists(os.path.join(cache_dir, "random.csv")):
                random_cache = pd.read_csv(os.path.join(cache_dir, "random.csv"))
            if os.path.exists(os.path.join(cache_dir, "uniform.csv")):
                uniform_cache = pd.read_csv(os.path.join(cache_dir, "uniform.csv"))

        from abacus2deepmd.utils.common import run_parallel_tasks
        import functools

        sampled_rows = []
        random_rows = []
        uniform_rows = []
        cache_hit = 0
        cache_miss = 0
        tasks = []
        for f in files:
            base = os.path.basename(f)
            if base.startswith("frame_metrics_"):
                system = base[len("frame_metrics_") : -4]
            elif base.startswith("frame_"):
                system = base[len("frame_") : -4]
            else:
                continue
            sampled_row = None
            random_row = None
            uniform_row = None
            if sampled_cache is not None:
                match = sampled_cache[sampled_cache["System"] == system]
                if not match.empty:
                    sampled_row = match.iloc[0].to_dict()
            if random_cache is not None:
                match = random_cache[random_cache["System"] == system]
                if not match.empty:
                    random_row = match.iloc[0].to_dict()
            if uniform_cache is not None:
                match = uniform_cache[uniform_cache["System"] == system]
                if not match.empty:
                    uniform_row = match.iloc[0].to_dict()

            if sampled_row and random_row and uniform_row:
                sampled_rows.append(sampled_row)
                random_rows.append(random_row)
                uniform_rows.append(uniform_row)
                cache_hit += 1
                self.logger.debug(f"System {system} cache hit, skip calculation")
                continue
            else:
                self.logger.debug(
                    f"System {system} cache miss: sampled={sampled_row is not None}, random={random_row is not None}, uniform={uniform_row is not None}"
                )

            # Cache miss, add to parallel tasks
            tasks.append(f)

        # Parallel processing of cache miss files
        if tasks:
            worker = functools.partial(
                self._analyze_single_system, system_paths=system_paths
            )
            results = run_parallel_tasks(
                tasks,
                worker_fn=worker,
                workers=workers,
                logger=self.logger,
                error_logger=self.error_logger,
                desc="Sampling comparison analysis",
            )
            for row in results:
                if not row:
                    continue
                cache_miss += 1
                base_info = {
                    "System": row.get("System", ""),
                    "Sample_Ratio": row.get("Sample_Ratio", ""),
                    "Total_Frames": row.get("Total_Frames", ""),
                    "Sampled_Frames": row.get("Sampled_Frames", ""),
                }
                sampled_rows.append(
                    {
                        **base_info,
                        "ANND": row.get("ANND_sampled"),
                        "MPD": row.get("MPD_sampled"),
                        "Total_Variance": row.get("Total_Variance_sampled"),
                        "Energy_Coverage_Ratio": row.get(
                            "Energy_Coverage_Ratio_sampled"
                        ),
                        "JS_Divergence": row.get("JS_Divergence_sampled"),
                        "RMSD_Mean": row.get("RMSD_Mean_sampled"),
                    }
                )
                random_rows.append(
                    {
                        **base_info,
                        "ANND": row.get("ANND_random"),
                        "MPD": row.get("MPD_random"),
                        "Total_Variance": row.get("Total_Variance_random"),
                        "Energy_Coverage_Ratio": row.get(
                            "Energy_Coverage_Ratio_random"
                        ),
                        "JS_Divergence": row.get("JS_Divergence_random"),
                        "RMSD_Mean": row.get("RMSD_Mean_random"),
                    }
                )
                uniform_rows.append(
                    {
                        **base_info,
                        "ANND": row.get("ANND_uniform"),
                        "MPD": row.get("MPD_uniform"),
                        "Total_Variance": row.get("Total_Variance_uniform"),
                        "Energy_Coverage_Ratio": row.get(
                            "Energy_Coverage_Ratio_uniform"
                        ),
                        "JS_Divergence": row.get("JS_Divergence_uniform"),
                        "RMSD_Mean": row.get("RMSD_Mean_uniform"),
                    }
                )

        # Sort the results
        from abacus2deepmd.utils.common import CommonUtils

        sampled_rows_sorted = sorted(
            sampled_rows, key=lambda x: CommonUtils.sort_system_names([x["System"]])[0]
        )
        random_rows_sorted = sorted(
            random_rows, key=lambda x: CommonUtils.sort_system_names([x["System"]])[0]
        )
        uniform_rows_sorted = sorted(
            uniform_rows, key=lambda x: CommonUtils.sort_system_names([x["System"]])[0]
        )

        pd.DataFrame(sampled_rows_sorted).to_csv(
            os.path.join(cache_dir, "sampled.csv"), index=False
        )
        pd.DataFrame(random_rows_sorted).to_csv(
            os.path.join(cache_dir, "random.csv"), index=False
        )
        pd.DataFrame(uniform_rows_sorted).to_csv(
            os.path.join(cache_dir, "uniform.csv"), index=False
        )

        total = cache_hit + cache_miss
        if cache_hit == total:
            self.logger.info(f"All {total} systems cache hit, skip all calculations.")
        elif cache_hit > 0:
            self.logger.info(
                f"{cache_hit} systems cache hit, {cache_miss} systems recalculated."
            )
        else:
            self.logger.info(f"All {total} systems recalculated.")

        self.logger.info(f"Sampling effect categorized results saved to {result_dir}")

        # Generate summary table
        if sampled_rows:
            self.logger.info(
                "Generating sampling method comparison summary table (including absolute and relative values)..."
            )
            self._create_summary_table([row for row in sampled_rows], result_dir)

        # Generate visualization charts
        if sampled_rows:
            self.logger.info(
                "Generating sampling method comparison visualization charts (using relative values for plotting)..."
            )
            self._create_comparison_plots(result_dir, len(sampled_rows))

        # Output unified completion statistics
        success_count = len(sampled_rows)  # Number of successfully analyzed systems
        total_systems = len(files)  # Total number of systems found
        self.logger.info(
            f"Sampling effect comparison analysis completed: Total {success_count}/{total_systems} | Success {success_count} | Failed {total_systems - success_count}"
        )

    def _analyze_single_system(self, file_path, system_paths):
        """Analyze data for a single system"""
        try:
            df = pd.read_csv(file_path)
            system = (
                os.path.basename(file_path)
                .replace("frame_metrics_", "")
                .replace(".csv", "")
                .replace("frame_", "")
            )
            system_path = system_paths.get(system, "")

            # Prepare data
            vector_cols = [
                col for col in df.columns if (col == "E_Proc" or col.startswith("C"))
            ]
            vectors = df[vector_cols].values
            selected = df["Selected"] == 1
            k = selected.sum()
            n = len(df)
            sample_ratio = k / n if n > 0 else 0

            if k == 0 or n == 0:
                return None

            # Get frame indices
            frame_indices = df["Frame_ID"].values
            sampled_indices = frame_indices[selected]

            # Recalculate RMSD for sampled group (based on this group's mean structure)
            sampled_rmsd = []
            if system_path:
                sampled_rmsd = self._calculate_group_rmsd(
                    system_path, sampled_indices.tolist()
                )
                if len(sampled_rmsd) == 0:
                    self.logger.warning(f"Unable to calculate RMSD for sampled group")
            else:
                self.logger.warning(
                    f"System path {system} not found, unable to calculate RMSD"
                )

            # Sampling algorithm results
            sampled_metrics = MetricsToolkit.adapt_sampling_metrics(
                vectors[selected],
                vectors,
                sampled_rmsd if len(sampled_rmsd) > 0 else [],
            )

            # Random sampling comparison
            rand_metrics = self._run_random_sampling_comparison(
                vectors, df, system_path, k, n
            )

            # Uniform sampling comparison
            uniform_metrics = self._run_uniform_sampling_comparison(
                vectors, df, system_path, k, n
            )

            # Build result row
            return self._build_result_row(
                system,
                sample_ratio,
                n,
                k,
                sampled_metrics,
                rand_metrics,
                uniform_metrics,
            )

        except Exception as e:
            # Record concise information in error log
            if self.error_logger:
                self.error_logger.error(
                    f"Sampling analysis failed - System: {system} | "
                    f"File: {os.path.basename(file_path)} | "
                    f"Error: {type(e).__name__}: {str(e)}"
                )

            # Record brief information in main log
            self.logger.error(
                f"Error analyzing system {os.path.basename(file_path)}: {type(e).__name__}"
            )
            return None

    def _run_random_sampling_comparison(self, vectors, df, system_path, k, n):
        """Run random sampling comparison (single run, fixed seed=42), return single metrics dictionary

        Parameter simplification: remove unused selected_mask / sampled_indices.
        """
        frame_indices = df["Frame_ID"].values
        rng = np.random.default_rng(42)
        idx = rng.choice(n, k, replace=False)
        sel_vectors = vectors[idx]
        sel_frame_indices = frame_indices[idx]

        # Recalculate random group RMSD
        rand_rmsd = []
        if system_path:
            rand_rmsd = self._calculate_group_rmsd(
                system_path, sel_frame_indices.tolist()
            )
            if len(rand_rmsd) == 0:
                self.logger.warning("Unable to calculate random group RMSD")
        else:
            self.logger.warning(
                "System path not found, unable to calculate random group RMSD"
            )

        return MetricsToolkit.adapt_sampling_metrics(
            sel_vectors, vectors, rand_rmsd if len(rand_rmsd) > 0 else []
        )

    def _run_uniform_sampling_comparison(self, vectors, df, system_path, k, n):
        """Run uniform sampling comparison"""
        if k == 0:
            return {}

        frame_indices = df["Frame_ID"].values
        idx_uniform = SamplingStrategy.uniform_sample_indices(n, k)
        sel_vectors = vectors[idx_uniform]
        sel_frame_indices = frame_indices[idx_uniform]

        # Recalculate uniform group RMSD
        uniform_rmsd = []
        if system_path:
            uniform_rmsd = self._calculate_group_rmsd(
                system_path, sel_frame_indices.tolist()
            )
            if len(uniform_rmsd) == 0:
                self.logger.warning("Unable to calculate uniform group RMSD")
        else:
            self.logger.warning(
                "System path not found, unable to calculate uniform group RMSD"
            )

        return MetricsToolkit.adapt_sampling_metrics(
            sel_vectors, vectors, uniform_rmsd if len(uniform_rmsd) > 0 else []
        )

    def _build_result_row(
        self, system, sample_ratio, n, k, sampled_metrics, rand_metrics, uniform_metrics
    ):
        """Build result row data"""
        # Random sampling single result directly uses rand_metrics

        return {
            # Basic information
            "System": system,
            "Sample_Ratio": sample_ratio,
            "Total_Frames": n,
            "Sampled_Frames": k,
            # Sampling algorithm results
            "ANND_sampled": sampled_metrics.get("ANND"),
            "MPD_sampled": sampled_metrics.get("MPD"),
            "Total_Variance_sampled": sampled_metrics.get("Total_Variance"),
            "Energy_Coverage_Ratio_sampled": sampled_metrics.get(
                "Energy_Coverage_Ratio"
            ),
            "JS_Divergence_sampled": sampled_metrics.get("JS_Divergence"),
            "RMSD_Mean_sampled": sampled_metrics.get("RMSD_Mean"),
            # Random sampling statistics
            "ANND_random": rand_metrics.get("ANND"),
            "MPD_random": rand_metrics.get("MPD"),
            "Total_Variance_random": rand_metrics.get("Total_Variance"),
            "Energy_Coverage_Ratio_random": rand_metrics.get("Energy_Coverage_Ratio"),
            "JS_Divergence_random": rand_metrics.get("JS_Divergence"),
            "RMSD_Mean_random": rand_metrics.get("RMSD_Mean"),
            # Uniform sampling results
            "ANND_uniform": uniform_metrics.get("ANND"),
            "MPD_uniform": uniform_metrics.get("MPD"),
            "Total_Variance_uniform": uniform_metrics.get("Total_Variance"),
            "Energy_Coverage_Ratio_uniform": uniform_metrics.get(
                "Energy_Coverage_Ratio"
            ),
            "JS_Divergence_uniform": uniform_metrics.get("JS_Divergence"),
            "RMSD_Mean_uniform": uniform_metrics.get("RMSD_Mean"),
            # Improvement percentage
            "ANND_improvement_pct": calculate_improvement(
                sampled_metrics.get("ANND"), rand_metrics.get("ANND")
            ),
            "RMSD_improvement_pct": calculate_improvement(
                sampled_metrics.get("RMSD_Mean"), rand_metrics.get("RMSD_Mean")
            ),
            # Statistical significance
            # Single random comparison cannot perform significance test, set to NaN
            "ANND_p_value": np.nan,
            "RMSD_p_value": np.nan,
            # Improvement relative to uniform sampling
            "ANND_vs_uniform_pct": calculate_improvement(
                sampled_metrics.get("ANND"), uniform_metrics.get("ANND")
            ),
            "RMSD_vs_uniform_pct": calculate_improvement(
                sampled_metrics.get("RMSD_Mean"), uniform_metrics.get("RMSD_Mean")
            ),
        }

    def _create_summary_table(self, rows, result_dir):
        """Create summary table (containing absolute and relative values)"""
        import pandas as pd
        import numpy as np
        import os

        # New and old cache directory compatibility
        cache_dir_new = os.path.join(result_dir, "sampling_comparison")
        cache_dir_legacy = os.path.join(result_dir, "sampling_comparison_cache")
        if os.path.isdir(cache_dir_new):
            cache_dir = cache_dir_new
        else:
            cache_dir = cache_dir_legacy
        sampled_path = os.path.join(cache_dir, "sampled.csv")
        random_path = os.path.join(cache_dir, "random.csv")
        uniform_path = os.path.join(cache_dir, "uniform.csv")

        if not (
            os.path.exists(sampled_path)
            and os.path.exists(random_path)
            and os.path.exists(uniform_path)
        ):
            self.logger.warning(
                "Three sampling type cache files incomplete, unable to generate summary table"
            )
            return

        df_sampled = pd.read_csv(sampled_path)
        df_random = pd.read_csv(random_path)
        df_uniform = pd.read_csv(uniform_path)

        # Merge three tables by System field
        df_merged = df_sampled.merge(
            df_random, on="System", suffixes=("_sampled", "_random")
        )
        # uniform suffix needs to be added manually
        df_uniform = df_uniform.add_suffix("_uniform")
        df_uniform = df_uniform.rename(columns={"System_uniform": "System"})
        df_merged = df_merged.merge(df_uniform, on="System", how="left")

        ordered = [
            "RMSD_Mean",
            "ANND",
            "MPD",
            "Total_Variance",
            "Energy_Coverage_Ratio",
            "JS_Divergence",
        ]
        metrics_to_summarize = []
        for m in ordered:
            metrics_to_summarize.append(
                (m, f"{m}_sampled", f"{m}_random", f"{m}_uniform")
            )

        self.logger.info(
            f"Start processing {len(df_merged)} systems data, calculating absolute and relative values..."
        )

        summary_rows_final = []
        for metric_name, sampled_col, random_col, uniform_col in metrics_to_summarize:
            # Get original values for each system
            sampled_values = []
            random_values = []
            uniform_values = []

            for _, row in df_merged.iterrows():
                system = row["System"]
                sampled_val = row[sampled_col]
                random_val = row[random_col]
                uniform_val = row[uniform_col]

                # Skip NaN values
                if pd.isna(sampled_val) or sampled_val == 0:
                    continue

                # Collect original absolute values
                sampled_values.append(sampled_val)
                random_values.append(random_val if not pd.isna(random_val) else np.nan)
                uniform_values.append(
                    uniform_val if not pd.isna(uniform_val) else np.nan
                )

            # Calculate absolute value statistics
            sampled_means = np.array([v for v in sampled_values if not pd.isna(v)])
            random_means = np.array([v for v in random_values if not pd.isna(v)])
            uniform_means = np.array([v for v in uniform_values if not pd.isna(v)])

            # Calculate relative values (with smart sampling as baseline, between 0-1)
            sampled_rel_values = [1.0] * len(
                sampled_values
            )  # Smart sampling baseline is 1.0
            random_rel_values = [
                (r / s) if not pd.isna(r) and not pd.isna(s) and s != 0 else np.nan
                for r, s in zip(random_values, sampled_values)
            ]
            uniform_rel_values = [
                (u / s) if not pd.isna(u) and not pd.isna(s) and s != 0 else np.nan
                for u, s in zip(uniform_values, sampled_values)
            ]

            # Calculate relative value statistics
            sampled_rel_means = np.array(
                [v for v in sampled_rel_values if not pd.isna(v)]
            )
            random_rel_means = np.array(
                [v for v in random_rel_values if not pd.isna(v)]
            )
            uniform_rel_means = np.array(
                [v for v in uniform_rel_values if not pd.isna(v)]
            )

            row = {"Metric": metric_name}

            # Absolute value statistics
            row.update(
                {
                    "Sampled_Mean_Abs": (
                        np.mean(sampled_means) if len(sampled_means) > 0 else np.nan
                    ),
                    "Sampled_Sem_Abs": (
                        np.std(sampled_means, ddof=1) / np.sqrt(len(sampled_means))
                        if len(sampled_means) >= 2
                        else np.nan
                    ),
                    "Random_Mean_Abs": (
                        np.mean(random_means) if len(random_means) > 0 else np.nan
                    ),
                    "Random_Sem_Abs": (
                        np.std(random_means, ddof=1) / np.sqrt(len(random_means))
                        if len(random_means) >= 2
                        else np.nan
                    ),
                    "Uniform_Mean_Abs": (
                        np.mean(uniform_means) if len(uniform_means) > 0 else np.nan
                    ),
                    "Uniform_Sem_Abs": (
                        np.std(uniform_means, ddof=1) / np.sqrt(len(uniform_means))
                        if len(uniform_means) >= 2
                        else np.nan
                    ),
                }
            )

            # Relative value statistics
            row.update(
                {
                    "Sampled_Mean_Rel": (
                        np.mean(sampled_rel_means)
                        if len(sampled_rel_means) > 0
                        else np.nan
                    ),
                    "Sampled_Sem_Rel": (
                        np.std(sampled_rel_means, ddof=1)
                        / np.sqrt(len(sampled_rel_means))
                        if len(sampled_rel_means) >= 2
                        else np.nan
                    ),
                    "Random_Mean_Rel": (
                        np.mean(random_rel_means)
                        if len(random_rel_means) > 0
                        else np.nan
                    ),
                    "Random_Sem_Rel": (
                        np.std(random_rel_means, ddof=1)
                        / np.sqrt(len(random_rel_means))
                        if len(random_rel_means) >= 2
                        else np.nan
                    ),
                    "Uniform_Mean_Rel": (
                        np.mean(uniform_rel_means)
                        if len(uniform_rel_means) > 0
                        else np.nan
                    ),
                    "Uniform_Sem_Rel": (
                        np.std(uniform_rel_means, ddof=1)
                        / np.sqrt(len(uniform_rel_means))
                        if len(uniform_rel_means) >= 2
                        else np.nan
                    ),
                }
            )

            summary_rows_final.append(row)

        summary_df = pd.DataFrame(summary_rows_final)

        # Save summary results (keep original filename)
        summary_path = os.path.join(result_dir, "sampling_methods_comparison.csv")
        summary_df.to_csv(summary_path, index=False)
        self.logger.info(f"Sampling method summary saved: {summary_path}")
        self.logger.info(
            "Contains absolute value (Abs) and relative value (Rel) two sets of statistical data, relative value with smart sampling as 1.0 baseline (between 0-1)"
        )

    def _create_comparison_plots(self, result_dir, num_systems):
        """Create sampling method comparison visualization charts (using relative values for plotting)"""
        import matplotlib.pyplot as plt

        # Read summary data
        summary_path = os.path.join(result_dir, "sampling_methods_comparison.csv")
        if not os.path.exists(summary_path):
            self.logger.warning(
                "Summary data file not found, unable to generate charts"
            )
            return

        df = pd.read_csv(summary_path)

        # Metric name mapping
        metric_names = {
            "RMSD_Mean": "RMSD Mean",
            "ANND": "Average Nearest Neighbor Distance",
            "MPD": "Mean Pairwise Distance",
            "Total_Variance": "Total Variance",
            "Energy_Coverage_Ratio": "Energy Coverage Ratio",
            "JS_Divergence": "JS Divergence",
        }

        # Filter valid metrics
        valid_data = []
        for _, row in df.iterrows():
            metric = row["Metric"]
            if (
                metric in metric_names
                and not pd.isna(row["Sampled_Mean_Rel"])
                and row["Sampled_Mean_Rel"] != 0
            ):
                sampled_mean_rel = row["Sampled_Mean_Rel"]
                random_mean_rel = row["Random_Mean_Rel"]
                uniform_mean_rel = row["Uniform_Mean_Rel"]

                # Use relative values for comparison
                sampled_pct = (
                    sampled_mean_rel if not pd.isna(sampled_mean_rel) else np.nan
                )
                random_pct = random_mean_rel if not pd.isna(random_mean_rel) else np.nan
                uniform_pct = (
                    uniform_mean_rel if not pd.isna(uniform_mean_rel) else np.nan
                )

                valid_data.append(
                    {
                        "metric": metric,
                        "name": metric_names[metric],
                        "sampled_pct": sampled_pct,
                        "random_pct": random_pct,
                        "uniform_pct": uniform_pct,
                        "sampled_mean_abs": row[
                            "Sampled_Mean_Abs"
                        ],  # Keep absolute values for labels
                        "random_mean_abs": row["Random_Mean_Abs"],
                        "uniform_mean_abs": row["Uniform_Mean_Abs"],
                        "sampled_sem_rel": row["Sampled_Sem_Rel"],
                        "random_sem_rel": row["Random_Sem_Rel"],
                        "uniform_sem_rel": row["Uniform_Sem_Rel"],
                    }
                )

        if not valid_data:
            self.logger.warning("No valid metric data available for plotting")
            return

        # Create mean comparison plot (using relative values)
        self._create_mean_comparison_plot_relative(valid_data, result_dir, num_systems)

    def _create_mean_comparison_plot_relative(
        self, valid_data, result_dir, num_systems
    ):
        """Create mean comparison horizontal bar chart (relative value version)"""

        fig, ax = plt.subplots(figsize=(14, 10))

        # Prepare data
        metrics = [item["name"] for item in valid_data]
        sampled_values = [item["sampled_pct"] for item in valid_data]
        random_values = [item["random_pct"] for item in valid_data]
        uniform_values = [item["uniform_pct"] for item in valid_data]

        # Get actual values for labels
        sampled_actual = [item["sampled_mean_abs"] for item in valid_data]
        random_actual = [item["random_mean_abs"] for item in valid_data]
        uniform_actual = [item["uniform_mean_abs"] for item in valid_data]

        # Use relative value standard errors for error bars
        sampled_sem_rel = [item["sampled_sem_rel"] for item in valid_data]
        random_sem_rel = [item["random_sem_rel"] for item in valid_data]
        uniform_sem_rel = [item["uniform_sem_rel"] for item in valid_data]

        # Calculate bar positions
        y_pos = np.arange(len(metrics))
        bar_height = 0.25

        # Draw horizontal bar chart (add error bars)
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        bars1 = ax.barh(
            y_pos - bar_height,
            sampled_values,
            bar_height,
            label="Smart Sampling (Baseline)",
            color=colors[0],
            alpha=0.8,
            xerr=sampled_sem_rel,
            capsize=3,
            error_kw={"elinewidth": 1, "ecolor": colors[0]},
        )
        bars2 = ax.barh(
            y_pos,
            random_values,
            bar_height,
            label="Random Sampling",
            color=colors[1],
            alpha=0.8,
            xerr=random_sem_rel,
            capsize=3,
            error_kw={"elinewidth": 1, "ecolor": colors[1]},
        )
        bars3 = ax.barh(
            y_pos + bar_height,
            uniform_values,
            bar_height,
            label="Uniform Sampling",
            color=colors[2],
            alpha=0.8,
            xerr=uniform_sem_rel,
            capsize=3,
            error_kw={"elinewidth": 1, "ecolor": colors[2]},
        )

        # Add baseline
        ax.axvline(x=1.0, color="red", linestyle="--", alpha=0.7, linewidth=1)
        ax.text(
            1.02,
            len(metrics) - 0.5,
            "Smart Sampling\nBaseline (1.0)",
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # Set labels and title
        ax.set_yticks(y_pos)
        ax.set_yticklabels(metrics, fontsize=11)
        ax.set_xlabel("Relative Performance (Ratio to Smart Sampling)", fontsize=12)
        ax.set_title(
            f"Sampling Methods Performance Comparison - Relative Values\n(Systems: {num_systems}, Mean ± Sem, Baseline: Smart Sampling = 1.0)",
            fontsize=16,
            fontweight="bold",
        )
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10)
        ax.grid(True, alpha=0.3, axis="x")

        # Dynamically adjust x-axis range to ensure error bars are fully displayed
        all_values = sampled_values + random_values + uniform_values
        all_errors = sampled_sem_rel + random_sem_rel + uniform_sem_rel

        x_min, x_max = self._calculate_axis_limits(
            all_values, all_errors, margin_percent=0.05, min_limit=0
        )
        ax.set_xlim(x_min, x_max)

        # Add value labels (relative value, actual value)
        def add_value_labels(bars, rel_values, actual_values):
            for bar, rel_val, act_val in zip(bars, rel_values, actual_values):
                if not pd.isna(rel_val) and not pd.isna(act_val):
                    width = bar.get_width()
                    label_text = f"({rel_val:.3f}, {act_val:.3f})"
                    # Place label inside the bar
                    ax.text(
                        width / 2,
                        bar.get_y() + bar.get_height() / 2,
                        label_text,
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white",
                        fontweight="bold",
                    )

        add_value_labels(bars1, sampled_values, sampled_actual)
        add_value_labels(bars2, random_values, random_actual)
        add_value_labels(bars3, uniform_values, uniform_actual)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)  # Leave more space for legend

        # Save chart
        plots_dir = os.path.join(result_dir, "sampling_comparison_plots")
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir, "sampling_performance_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Sampling performance comparison chart saved: {plot_path}")
        self.logger.info(
            "Chart uses relative value coordinates and error bars, labels show absolute value information"
        )

    def _calculate_group_rmsd(
        self, system_path: str, frame_indices: List[int]
    ) -> np.ndarray:
        return RMSDCalculator.calculate_group_rmsd(
            system_path, frame_indices, self.logger
        )


# Compatibility function
def analyse_sampling_compare(
    result_dir=None,
    workers: int = -1,
    error_log_file: str = None,
    error_logger: logging.Logger = None,
    force_recompute: bool = False,
):
    """Compatibility function, maintain backward compatibility"""
    analyser = SamplingComparisonAnalyser(
        error_log_file=error_log_file, error_logger=error_logger
    )
    return analyser.analyse_sampling_compare(
        result_dir=result_dir, workers=workers, force_recompute=force_recompute
    )
