#!/usr/bin/env python
"""
Power Parameter Tester - Test sampling effect comparison for different power_p parameters
Read systems from default paths, sample using different p values respectively, then generate comprehensive comparison charts
Support parallel analysis of multiple systems, calculate mean and standard error, and draw line charts with error bars

Optimization features:
1. Relative value normalization: Use p=0 as 100% baseline, convert each system metrics to relative values, eliminate absolute value differences between different systems
2. Parallel optimization: Use ProcessPoolExecutor to avoid Python GIL limitations, support multi-core parallel analysis
3. Random system selection: Support random selection of test systems (seed=42) and control maximum system count
4. Enhanced visualization: Charts display relative change percentages, highlighting the impact of p values on sampling effects
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import concurrent.futures

from abacus2deepmd.io.path_manager import PathManager
from abacus2deepmd.io.file_utils import lightweight_discover_systems
from abacus2deepmd.core.analysis_orchestrator import (
    AnalysisOrchestrator,
    AnalysisConfig,
)
from abacus2deepmd.core.system_analyser import SystemAnalyser
from abacus2deepmd.utils.logmanager import LoggerManager


class PowerParameterTester:
    """Test sampling effect of different power_p parameters"""

    def __init__(
        self,
        output_base_dir: str = "power_test_results",
        max_workers: int = -1,
        max_systems: Optional[int] = None,
        sample_ratio: float = 0.1,
        pca_variance_ratio: float = 0.90,
        search_paths: Optional[List[str]] = None,
        include_project: bool = False,
        force_recompute: bool = False,
        logger=None,
    ):
        self.output_base_dir = output_base_dir
        self.max_workers = max_workers
        self.max_systems = max_systems
        self.sample_ratio = sample_ratio
        self.pca_variance_ratio = pca_variance_ratio
        self.search_paths = search_paths
        self.include_project = include_project
        self.force_recompute = force_recompute
        self.logger = logger or LoggerManager.create_logger(__name__)
        self.results = []
        self.num_systems = None

    def _calculate_axis_limits(
        self, values, errors, margin_percent=0.05, min_limit=None, max_limit=None
    ):
        """Calculate dynamic range of axis to ensure error bars are fully displayed

        Args:
            values: Data value list
            errors: Error value list
            margin_percent: Margin percentage
            min_limit: Minimum limit value
            max_limit: Maximum limit value

        Returns:
            (min_limit, max_limit): Axis range
        """
        # Filter NaN values
        valid_values = [v for v in values if not pd.isna(v)]
        valid_errors = [e for e in errors if not pd.isna(e)]

        if not valid_values or not valid_errors:
            return (-1, 1)  # Default range

        # Calculate boundaries
        data_min = min(valid_values)
        data_max = max(valid_values)
        max_error = max(valid_errors)

        axis_min = data_min - max_error
        axis_max = data_max + max_error

        # Apply limits
        if min_limit is not None:
            axis_min = max(axis_min, min_limit)
        if max_limit is not None:
            axis_max = min(axis_max, max_limit)

        # Calculate margins
        data_range = axis_max - axis_min
        if data_range > 0:
            margin = margin_percent * data_range
            axis_min -= margin
            axis_max += margin

        return (axis_min, axis_max)

    def discover_systems(self) -> List:
        """Automatically discover systems"""
        # Prefer to use the search path parsed by the main program; otherwise fall back to the default path (the parent directory of the package directory)
        if self.search_paths and len(self.search_paths) > 0:
            search_paths = self.search_paths
        else:
            search_paths = [
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
            ]

        self.logger.info(f"Search paths: {search_paths}")
        records = lightweight_discover_systems(
            search_paths, include_project=self.include_project
        )

        if not records:
            self.logger.warning(
                "No systems found, please ensure there are ABACUS output directories that meet the standards"
            )
            return []

        self.logger.info(f"Discovered {len(records)} systems")

        # If maximum number of systems is set, perform random selection
        if self.max_systems is not None and len(records) > self.max_systems:
            # Set random seed to ensure reproducibility
            np.random.seed(42)
            selected_indices = np.random.choice(
                len(records), size=self.max_systems, replace=False
            )
            records = [records[i] for i in selected_indices]
            self.logger.info(
                f"Randomly selected {self.max_systems} systems for testing"
            )

        return records

    def run_single_test(
        self, system_record, power_p: float, test_name: str
    ) -> Dict[str, Any]:
        """Run test for a single p value"""
        try:
            # Configure analysis parameters - use passed parameters (inherit from main program)
            # Use temporary directory as output directory since we don't need to save intermediate files
            import tempfile

            temp_dir = tempfile.mkdtemp()
            config = AnalysisConfig(
                sample_ratio=self.sample_ratio,
                power_p=power_p,
                pca_variance_ratio=self.pca_variance_ratio,
                workers=self.max_workers,
                output_dir=temp_dir,  # Use temporary directory
                search_paths=[system_record.system_path],
                force_recompute=self.force_recompute,
                steps=[1],  # Only execute sampling step
            )

            # Create analyzer
            orchestrator = AnalysisOrchestrator(config)
            analyser = SystemAnalyser(
                sample_ratio=config.sample_ratio,
                power_p=config.power_p,
                pca_variance_ratio=config.pca_variance_ratio,
            )

            # Execute analysis
            path_manager = PathManager(output_dir=temp_dir)
            path_manager.set_output_dir_for_params(config.as_param_dict())

            # Analyze system
            analysis_result = analyser.analyse_system(
                system_record.system_path, pre_sampled_frames=[]
            )

            if analysis_result:
                # Unpack analysis results
                metrics, frames, swap_count, pca_components_data, _ = analysis_result

                # Extract key metrics
                extracted_metrics = self.extract_metrics_from_result(
                    metrics, power_p, test_name
                )
                return extracted_metrics
            else:
                return (None, f"Analysis failed: {test_name}")

        except Exception as e:
            return (None, f"Test failed {test_name}: {str(e)}")
        finally:
            # Clean up temporary directory
            import shutil

            try:
                if "temp_dir" in locals():
                    shutil.rmtree(temp_dir)
            except Exception as e:
                pass

    def extract_metrics_from_result(
        self, metrics, power_p: float, test_name: str
    ) -> Dict[str, Any]:
        """Extract key metrics from TrajectoryMetrics object"""
        try:
            # Extract metrics from metrics object
            return {
                "test_name": test_name,
                "power_p": power_p,
                "system_id": getattr(
                    metrics, "system_name", "unknown"
                ),  # Use system_id as unified field name
                "RMSD_Mean": getattr(metrics, "rmsd_mean", np.nan),
                "ANND": getattr(metrics, "ANND", np.nan),
                "MPD": getattr(metrics, "MPD", np.nan),
                "Total_Variance": getattr(metrics, "total_variance", np.nan),
                "Energy_Coverage_Ratio": getattr(
                    metrics, "energy_coverage_ratio", np.nan
                ),
                "JS_Divergence": getattr(metrics, "js_divergence", np.nan),
                "sampled_frames": len(getattr(metrics, "sampled_frames", [])),
                "total_frames": getattr(metrics, "num_frames", 0),
            }
        except Exception as e:
            self.logger.error(f"Failed to extract metrics: {str(e)}")
            return None

    def run_single_test_task(self, system_record, power_p: float):
        """Run test task for single (system, p-value) combination"""
        test_name = f"Power_p_{power_p:.1f}"
        result = self.run_single_test(system_record, power_p, test_name)

        # If the result is an error tuple, return directly
        if isinstance(result, tuple) and len(result) > 0 and result[0] is None:
            return result

        return result

    def aggregate_results(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Aggregate results from multiple systems, convert to relative values (with p=0 as 100% baseline)"""
        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        metrics = [
            "RMSD_Mean",
            "ANND",
            "MPD",
            "Total_Variance",
            "Energy_Coverage_Ratio",
            "JS_Divergence",
        ]

        # Group by system, convert each system's metrics to relative values (with p=0 as baseline)
        normalized_results = []

        for system_id in df["system_id"].unique():
            system_df = df[df["system_id"] == system_id].copy()

            # Get baseline values for p=0
            baseline_row = system_df[system_df["power_p"] == 0.0]
            if baseline_row.empty:
                self.logger.warning(
                    f"System {system_id} lacks p=0 baseline data, skipping this system"
                )
                continue

            baseline_values = baseline_row[metrics].iloc[0]

            # Convert all metrics to relative values (percentages)
            for metric in metrics:
                baseline_val = baseline_values[metric]
                if pd.notna(baseline_val) and baseline_val != 0:
                    system_df[metric] = (system_df[metric] / baseline_val) * 100
                else:
                    # If baseline value is 0 or NaN, keep original value
                    self.logger.warning(
                        f"System {system_id} metric {metric} has baseline value of 0 or NaN, keeping original value"
                    )

            normalized_results.append(system_df)

        if not normalized_results:
            self.logger.error("No valid normalization results")
            return pd.DataFrame()

        # Merge all normalized results
        normalized_df = pd.concat(normalized_results, ignore_index=True)

        # Group by power_p, calculate mean and standard error
        def calc_sem(x):
            """Calculate standard error"""
            if len(x) <= 1:
                return np.nan
            return x.std(ddof=1) / np.sqrt(len(x))

        # Calculate mean and standard error for each metric separately
        result_dfs = []
        for metric in metrics:
            grouped = (
                normalized_df.groupby("power_p")[metric]
                .agg(["mean", calc_sem])
                .reset_index()
            )
            grouped.columns = ["power_p", f"{metric}_mean", f"{metric}_sem"]
            result_dfs.append(grouped.set_index("power_p"))

        # Merge all results
        if result_dfs:
            aggregated = result_dfs[0]
            for df in result_dfs[1:]:
                aggregated = aggregated.join(df, how="outer")
            aggregated = aggregated.reset_index()
        else:
            aggregated = pd.DataFrame()

        return aggregated

    def run_all_tests(self):
        """Run all tests, generate comparison charts only - using fine-grained task allocation"""
        self.logger.info("Starting Power parameter test analysis...")

        # Discover systems
        systems = self.discover_systems()
        if not systems:
            self.logger.warning(
                "No available systems discovered, skipping Power parameter test"
            )
            return

        self.num_systems = len(systems)

        # Generate all (system, p-value) combination tasks
        p_values = np.arange(-1.0, 1.1, 0.1)
        p_values = np.round(p_values, 1)  # Fix floating point precision issues

        # Create task list: each task is a (system, p-value) combination
        tasks = []
        for system in systems:
            for p in p_values:
                tasks.append((system, p))

        self.logger.info(
            f"Power parameter test tasks prepared: {len(systems)} systems, {len(p_values)} p-values, {len(tasks)} total tasks"
        )

        # Determine actual number of workers to use
        if self.max_workers == -1:
            import multiprocessing

            actual_workers = multiprocessing.cpu_count()
        else:
            actual_workers = max(1, self.max_workers)

        self.logger.info(f"Using {actual_workers} processes for parallel testing")

        # Create progress reporter
        from ..utils.progress import ProgressReporter

        reporter = ProgressReporter(
            total=len(tasks),
            logger=self.logger,
            desc="Power parameter test",
            error_logger=self.logger,  # Use the same logger as error_logger
        )

        # Execute all tasks using task pool
        all_results = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=actual_workers
        ) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self.run_single_test_task, system, p): (system, p)
                for system, p in tasks
            }

            # Collect results
            for future in concurrent.futures.as_completed(future_to_task):
                system, p = future_to_task[future]
                try:
                    result = future.result()
                    if result and isinstance(result, dict):
                        all_results.append(result)
                        reporter.item_done(success=True)
                    else:
                        # Handle error results
                        error_msg = "Analysis failed"
                        if isinstance(result, tuple) and len(result) > 1:
                            error_msg = result[1]
                        reporter.item_done(success=False, error=error_msg)
                except Exception as e:
                    reporter.item_done(success=False, error=str(e))

        # Complete progress reporting
        reporter.finish()

        if not all_results:
            self.logger.warning("No valid test results obtained")
            return

        self.logger.info(
            f"Obtained {len(all_results)} valid test results, starting to generate comparison charts..."
        )

        # Aggregate results
        aggregated_df = self.aggregate_results(all_results)

        # Generate comparison charts
        self.generate_comparison_plot(aggregated_df)

        self.logger.info("Power parameter test analysis completed!")

    def save_results(self, results: List[Dict[str, Any]]):
        """Save test results - only save charts, not CSV/JSON"""
        # Do not save CSV and JSON, only save charts
        pass

    def generate_comparison_plot(self, aggregated_df: pd.DataFrame):
        """Generate comprehensive comparison charts with error bars, showing percentage changes relative to p=0"""
        if aggregated_df.empty:
            self.logger.warning("aggregated_df is empty, cannot generate chart")
            return

        import matplotlib.pyplot as plt

        # Set Chinese font
        plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        # Metric name mapping
        metric_names = {
            "RMSD_Mean": "RMSD Mean",
            "ANND": "Average Nearest Neighbor Distance",
            "MPD": "Mean Pairwise Distance",
            "Total_Variance": "Total Variance",
            "Energy_Coverage_Ratio": "Energy Coverage Ratio",
            "JS_Divergence": "JS Divergence",
        }

        # Create chart
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        metrics = [
            "RMSD_Mean",
            "ANND",
            "MPD",
            "Total_Variance",
            "Energy_Coverage_Ratio",
            "JS_Divergence",
        ]

        for idx, metric in enumerate(metrics):
            if idx >= 6:
                break

            ax = axes[idx]

            mean_col = f"{metric}_mean"
            sem_col = f"{metric}_sem"

            if mean_col in aggregated_df.columns and sem_col in aggregated_df.columns:
                p_values = aggregated_df["power_p"].values
                means = aggregated_df[mean_col].values
                sems = aggregated_df[sem_col].values

                # Filter NaN values
                valid_mask = ~pd.isna(means) & ~pd.isna(sems)
                if np.any(valid_mask):
                    ax.errorbar(
                        p_values[valid_mask],
                        means[valid_mask],
                        yerr=sems[valid_mask],
                        fmt="o-",
                        linewidth=2,
                        markersize=6,
                        capsize=5,
                        alpha=0.8,
                        color="blue",
                    )

                    # Add p=0 100% baseline
                    ax.axhline(
                        y=100, color="red", linestyle="--", alpha=0.7, linewidth=1
                    )
                    ax.text(
                        0.02,
                        0.98,
                        "p=0 baseline (100%)",
                        transform=ax.transAxes,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                    )

                    # Annotate special points
                    for p, val in zip(p_values[valid_mask], means[valid_mask]):
                        if p in [-1.0, 0.0, 1.0]:  # Annotate key points
                            ax.annotate(
                                f"{val:.1f}%",
                                (p, val),
                                textcoords="offset points",
                                xytext=(0, 10),
                                ha="center",
                                fontsize=9,
                            )

            ax.set_xlabel("Power Parameter (p)")
            ax.set_ylabel(f"{metric_names[metric]} (% of p=0)")
            ax.set_title(
                f"{metric_names[metric]} vs Power Parameter\n(Relative to p=0 baseline)"
            )
            ax.grid(True, alpha=0.3)

            # Dynamically adjust x-axis range to ensure error bars are fully displayed
            if mean_col in aggregated_df.columns and sem_col in aggregated_df.columns:
                p_vals = aggregated_df["power_p"].values
                mean_vals = aggregated_df[mean_col].values
                sem_vals = aggregated_df[sem_col].values

                # Filter NaN values
                valid_mask = ~pd.isna(mean_vals) & ~pd.isna(sem_vals)
                if np.any(valid_mask):
                    # Calculate boundaries for all points and error bars
                    x_min, x_max = self._calculate_axis_limits(
                        p_vals[valid_mask], sem_vals[valid_mask], margin_percent=0.05
                    )
                    ax.set_xlim(x_min, x_max)

        # Hide extra subplots
        for idx in range(len(metrics), 6):
            axes[idx].set_visible(False)

        plt.suptitle(
            f"Power Parameter Comparison Results - Relative Values\n(Sample Ratio: {self.sample_ratio}, Systems: {self.num_systems}, Baseline: p=0)",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()

        # Create power analysis plots directory
        plots_dir = os.path.join(self.output_base_dir, "power_analysis_plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Save chart - include sampling rate information
        plot_filename = (
            f"power_parameter_comparison_r{self.sample_ratio}_multi_system_relative.png"
        )
        plot_path = os.path.join(plots_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Comparison chart saved: {plot_path}")
        self.logger.info(
            "Chart shows percentage changes relative to p=0 baseline, eliminating absolute value differences between different systems"
        )


def run_power_parameter_test(
    output_base_dir: str = "power_test_results",
    max_workers: int = -1,
    max_systems: Optional[int] = None,
    sample_ratio: float = 0.1,
    pca_variance_ratio: float = 0.90,
    search_paths: Optional[List[str]] = None,
    include_project: bool = False,
    force_recompute: bool = False,
    logger=None,
):
    """Convenient function to run power_p parameter test"""
    tester = PowerParameterTester(
        output_base_dir=output_base_dir,
        max_workers=max_workers,
        max_systems=max_systems,
        sample_ratio=sample_ratio,
        pca_variance_ratio=pca_variance_ratio,
        search_paths=search_paths,
        include_project=include_project,
        force_recompute=force_recompute,
        logger=logger,
    )
    tester.run_all_tests()
    return tester
