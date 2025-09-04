#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ABACUS main analyzer program - Refactored version
Function: Batch analysis of ABACUS molecular dynamics trajectories, focusing on sampling, system analysis and deepmd conversion
"""

import os
import sys
import time
import argparse
import traceback
import json
import numpy as np
from typing import List, Dict, Tuple

# Import custom modules (changed to package internal absolute imports)
from abacus2deepmd.io.path_manager import PathManager
from abacus2deepmd.io.file_utils import lightweight_discover_systems
from abacus2deepmd.core.system_analyser import ErrorHandler
from abacus2deepmd.utils.common import run_parallel_tasks_streaming
from abacus2deepmd.core.analysis_orchestrator import (
    AnalysisOrchestrator,
    AnalysisConfig,
)
from abacus2deepmd.io.result_saver import _deepmd_export_worker, DeepMDTaskBuilder
from abacus2deepmd.analysis.sampling_comparison_analyser import analyse_sampling_compare
from abacus2deepmd.analysis.power_parameter_tester import run_power_parameter_test
from abacus2deepmd.utils.progress import ProgressReporter
import signal


# Multi-process work context


class WorkflowExecutor:
    """Workflow executor - Extract common methods to reduce redundancy"""

    def __init__(self, orchestrator: AnalysisOrchestrator):
        self.orchestrator = orchestrator

    def execute_sampling_step(
        self, config: AnalysisConfig
    ) -> Tuple[List[tuple], PathManager, str]:
        """Execute sampling step"""
        search_paths = self.orchestrator.resolve_search_paths()
        records = lightweight_discover_systems(
            search_paths, include_project=config.include_project
        )
        if not records:
            self.orchestrator.logger.error("Lightweight discovery found no systems")
            return [], None, ""

        # Output directory
        path_manager, actual_output_dir = self.orchestrator.setup_output_directory()

        # Build mol_systems for path_manager compatibility in saving sampling information
        mol_systems: Dict[str, List[str]] = {}
        for rec in records:
            mol_systems.setdefault(rec.mol_id, []).append(rec.system_path)

        # Fix: Ensure analysis_targets.json is loaded correctly to enable reuse
        self.orchestrator.setup_analysis_targets(path_manager, search_paths)

        # Use orchestrator's execute_analysis method directly
        analysis_results = self.orchestrator.execute_analysis(path_manager)

        # Save results
        self.orchestrator.save_results(analysis_results, path_manager)

        return analysis_results, path_manager, actual_output_dir

    def execute_deepmd_step(
        self,
        analysis_results: List[tuple],
        path_manager: PathManager,
        actual_output_dir: str,
    ) -> None:
        """Execute DeepMD export step (parallelized implementation)"""
        self.orchestrator.logger.info(
            "Starting independent DeepMD data export (parallel support)..."
        )

        # Use new task builder
        tasks = DeepMDTaskBuilder.build_deepmd_tasks(
            analysis_results,
            path_manager,
            actual_output_dir,
            self.orchestrator.config.force_recompute,
        )

        if not tasks:
            self.orchestrator.logger.warning("No tasks available for DeepMD export")
            return

        workers = self.orchestrator.determine_workers()
        if workers <= 1:
            # Fallback to sequential mode
            reporter = ProgressReporter(
                total=len(tasks),
                logger=self.orchestrator.logger,
                desc="DeepMD export (sequential)",
                error_logger=self.orchestrator.error_logger,
            )
            success = 0
            for task in tasks:
                try:
                    result = _deepmd_export_worker(task)
                    if result and isinstance(result, tuple) and result[0]:
                        success += 1
                        # Complete DeepMD export task (no individual system info output)
                        reporter.item_done(success=True)
                    else:
                        error_msg = (
                            result[1]
                            if result and isinstance(result, tuple) and len(result) > 1
                            else "unknown error"
                        )
                        reporter.item_done(success=False, error=error_msg)
                except Exception as e:
                    reporter.item_done(success=False, error=e)
            reporter.finish()
            return

        self.orchestrator.logger.info(
            f"DeepMD export parallel start, tasks {len(tasks)}, workers={workers}"
        )
        results = run_parallel_tasks_streaming(
            tasks=tasks,
            worker_fn=_deepmd_export_worker,
            workers=workers,
            logger=self.orchestrator.logger,
            error_logger=self.orchestrator.error_logger,
            desc="DeepMD export",
            json_save_callback=None,  # DeepMD export doesn't need JSON save
            single_analysis_save_callback=None,  # DeepMD export doesn't need single_analysis save
        )

    def execute_sampling_compare_step(self, actual_output_dir: str) -> None:
        """Execute sampling comparison step"""
        self.orchestrator.logger.info("Starting sampling comparison analysis...")

        try:
            # Get worker count
            workers = self.orchestrator.determine_workers()

            # Check if force recompute is needed
            force_recompute = self.orchestrator.config.force_recompute

            # Check cache validity, force recompute if cache is invalid
            if not force_recompute:
                cache_dir = os.path.join(actual_output_dir, "sampling_comparison")
                if os.path.exists(cache_dir):
                    cache_valid = self._check_cache_validity(cache_dir)
                    if not cache_valid:
                        force_recompute = True
                        self.orchestrator.logger.info(
                            "Detected cache file format mismatch, force recompute mode enabled"
                        )

            # If force recompute, clear sampling comparison cache directory
            if force_recompute:
                import shutil

                cache_dir = os.path.join(actual_output_dir, "sampling_comparison")
                if os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir)
                    self.orchestrator.logger.info(
                        "Force recompute: sampling comparison cache directory cleared"
                    )

            # Execute comparison analysis - use orchestrator's error logger
            analyse_sampling_compare(
                result_dir=actual_output_dir,
                workers=workers,
                error_logger=self.orchestrator.error_logger,
                force_recompute=force_recompute,
            )

        except ImportError as e:
            ErrorHandler.log_simple_error(
                self.orchestrator.logger,
                e,
                "Unable to import sampling comparison module",
            )
        except Exception as e:
            ErrorHandler.log_simple_error(
                self.orchestrator.logger, e, "Sampling comparison analysis failed"
            )

    def execute_power_parameter_test_step(
        self,
        max_workers: int = -1,
        max_systems: int = None,
        sample_ratio: float = 0.1,
        pca_variance_ratio: float = 0.90,
        search_paths: List[str] = None,
        include_project: bool = False,
        force_recompute: bool = False,
    ) -> None:
        """Execute power parameter testing step"""
        self.orchestrator.logger.info("Starting Power parameter testing analysis...")

        try:
            # Get output directory - use analysis_results as base directory, not subdirectory
            path_manager, _ = self.orchestrator.setup_output_directory()
            output_base_dir = os.path.dirname(
                path_manager.output_dir
            )  # Get analysis_results directory

            # Execute power parameter testing, using passed computation parameters
            run_power_parameter_test(
                output_base_dir=output_base_dir,
                max_workers=max_workers,
                max_systems=max_systems,
                sample_ratio=sample_ratio,
                pca_variance_ratio=pca_variance_ratio,
                search_paths=search_paths,
                include_project=include_project,
                force_recompute=force_recompute,
                logger=self.orchestrator.logger,
            )

        except Exception as e:
            ErrorHandler.log_simple_error(
                self.orchestrator.logger, e, "Power parameter testing failed"
            )

    def _check_cache_validity(self, cache_dir: str) -> bool:
        """Check validity of sampling comparison cache files"""
        import os
        import pandas as pd

        required_files = ["sampled.csv", "random.csv", "uniform.csv"]
        required_columns = ["System", "Total_Variance", "Energy_Coverage_Ratio"]

        for file_name in required_files:
            file_path = os.path.join(cache_dir, file_name)
            if not os.path.exists(file_path):
                return False

            try:
                df = pd.read_csv(file_path)
                if not all(col in df.columns for col in required_columns):
                    self.orchestrator.logger.warning(
                        f"Cache file {file_name} missing required columns"
                    )
                    return False
            except Exception as e:
                self.orchestrator.logger.warning(
                    f"Failed to read cache file {file_name}: {e}"
                )
                return False

        return True

    def load_existing_results_for_deepmd(
        self, config: AnalysisConfig
    ) -> Tuple[List[tuple], PathManager, str]:
        """Load existing analysis results for DeepMD export, ensure correct parsing of system_path and sampled_frames"""
        self.orchestrator.logger.info("Loading existing results for DeepMD export...")

        # Set output directory
        path_manager, actual_output_dir = self.orchestrator.setup_output_directory()

        # Use new build method
        analysis_results = path_manager.build_results_for_existing_targets()

        return analysis_results, path_manager, actual_output_dir

    def load_existing_results_for_compare(
        self, config: AnalysisConfig
    ) -> Tuple[List[tuple], PathManager, str]:
        """Load existing analysis results for sampling comparison"""
        self.orchestrator.logger.info(
            "Preparing sampling effect comparison (from existing result directory)..."
        )

        # Set output directory
        path_manager, actual_output_dir = self.orchestrator.setup_output_directory()

        # Check if necessary files exist
        targets_file = os.path.join(actual_output_dir, "analysis_targets.json")
        if not os.path.exists(targets_file):
            self.orchestrator.logger.warning(
                f"analysis_targets.json file not found: {targets_file}"
            )
            self.orchestrator.logger.info(
                "Sampling comparison will try to find available data from result directory"
            )

        # For sampling comparison, we return an empty analysis results list,
        # because analyse_sampling_compare function will work directly from result directory
        analysis_results = []  # Empty list indicates need to load data from directory

        return analysis_results, path_manager, actual_output_dir

    def finalize_analysis(
        self,
        analysis_results: List[tuple],
        path_manager: PathManager,
        start_time: float,
        output_dir: str,
    ) -> None:
        """Complete analysis and output statistics"""
        # Save analysis target status
        try:
            current_analysis_params = self.orchestrator.config.as_param_dict()
            path_manager.save_analysis_targets(current_analysis_params)
        except Exception as e:
            ErrorHandler.log_simple_error(
                self.orchestrator.logger, e, "Save analysis target failed"
            )

        # Output final statistics
        elapsed = time.time() - start_time

        # Adjust statistics logic based on current step
        if self.orchestrator.current_step == 2:
            # Step 2: DeepMD export, use path_manager targets as total
            total_targets = len(path_manager.targets) if path_manager else 0
            analysed_count = len(analysis_results) if analysis_results else 0
        elif self.orchestrator.current_step == 3:
            # Step 3: Sampling comparison, use path_manager targets as total
            total_targets = len(path_manager.targets) if path_manager else 0
            analysed_count = (
                len(path_manager.targets) if path_manager else 0
            )  # Sampling comparison analyzes all available systems
        elif self.orchestrator.current_step == 4:
            # Step 4: Power parameter testing, no specific statistics needed
            total_targets = 0
            analysed_count = 0
        else:
            # Other steps: use path_manager targets
            total_targets = len(path_manager.targets) if path_manager else 0
            analysed_count = len(analysis_results) if analysis_results else 0

        self.orchestrator.logger.info("=" * 60)
        self.orchestrator.logger.info(
            f"Analysis completed! Actually analyzed {analysed_count}/{total_targets} systems"
        )

        if analysis_results:
            # Output sampling statistics
            swap_counts = [result[2] for result in analysis_results if len(result) > 2]
            if swap_counts:
                self.orchestrator.logger.info("Sampling optimization statistics:")
                self.orchestrator.logger.info(
                    "  Average swap count: %.2f", np.mean(swap_counts)
                )
                self.orchestrator.logger.info(
                    f"  Total swap count: {int(sum(swap_counts))}"
                )

        self.orchestrator.logger.info(
            f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)"
        )
        self.orchestrator.logger.info(f"Result directory: {output_dir}")

        self.orchestrator.logger.info("=" * 60)


class MainApp:
    """Main application class - refactored version"""

    def __init__(self):
        self.orchestrator = None
        self.workflow_executor = None

    def run(self) -> None:
        """Run main program"""
        start_time = time.time()

        # Set up signal handler for graceful program interruption handling
        self._setup_signal_handlers()

        try:
            # Parse arguments and create configuration
            config, args = self._parse_arguments_to_config()

            # Create orchestrator
            self.orchestrator = AnalysisOrchestrator(config)

            # Create workflow executor
            self.workflow_executor = WorkflowExecutor(self.orchestrator)

            # Set up logging
            self.orchestrator.setup_logging()

            # Log startup information
            self._log_startup_info(config)

            # Execute main analysis workflow
            self._execute_workflow(config, start_time)

        except KeyboardInterrupt:
            # Handle Ctrl+C interrupt
            if self.orchestrator and self.orchestrator.logger:
                self.orchestrator.logger.warning(
                    "Interrupt signal received, saving current progress..."
                )
                self._emergency_save_progress()
                self.orchestrator.logger.info("Progress saved, program exiting")
            else:
                print("\nInterrupt signal received, exiting...", file=sys.stderr)
        except Exception as e:
            if self.orchestrator and self.orchestrator.logger:
                self.orchestrator.logger.error(
                    f"Main program execution error: {str(e)}"
                )
                self.orchestrator.logger.error(
                    f"Detailed error information: {traceback.format_exc()}"
                )
                # Try to save current progress
                self._emergency_save_progress()
            else:
                print(f"Main program execution error: {str(e)}", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
        finally:
            if self.orchestrator:
                self.orchestrator.cleanup_logging()

    def _setup_signal_handlers(self) -> None:
        """Set up signal handler"""
        try:

            def signal_handler(signum, frame):
                """Signal handler"""
                if self.orchestrator and self.orchestrator.logger:
                    self.orchestrator.logger.warning(
                        f"Received signal {signum}, saving current progress..."
                    )
                    self._emergency_save_progress()
                    self.orchestrator.logger.info("Progress saved, program exiting")
                else:
                    print(f"\nReceived signal {signum}, exiting...", file=sys.stderr)
                sys.exit(1)

            # Register signal handler
            signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
            signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

        except ImportError:
            # Windows may not have signal module, or some platforms don't support it
            pass
        except Exception as e:
            # If signal handler setup fails, continue execution
            if self.orchestrator and self.orchestrator.logger:
                ErrorHandler.log_simple_error(
                    self.orchestrator.logger, e, "Failed to set signal handler"
                )

    def _emergency_save_progress(self) -> None:
        """Emergency save current progress"""
        try:
            if self.orchestrator and hasattr(self.orchestrator, "current_path_manager"):
                path_manager = self.orchestrator.current_path_manager
                if path_manager and path_manager.targets:
                    # Save current analysis target status
                    current_analysis_params = self.orchestrator.config.as_param_dict()
                    path_manager.save_analysis_targets(current_analysis_params)
                    if self.orchestrator.logger:
                        self.orchestrator.logger.info(
                            "Emergency save progress completed"
                        )
        except Exception as e:
            if self.orchestrator and self.orchestrator.logger:
                ErrorHandler.log_simple_error(
                    self.orchestrator.logger, e, "Emergency save progress failed"
                )
            else:
                print(f"Emergency save progress failed: {e}", file=sys.stderr)

    def _parse_steps_argument(self, steps_str: str) -> List[int]:
        """Parse step parameters, support multiple formats:
        - Single number: "1"
        - List format: "[1,2,4]"
        - Range format: "[1,3-4]" means [1,3,4]
        """
        if not steps_str:
            return []

        # Remove whitespace characters
        steps_str = steps_str.strip()

        # Handle single number case
        if steps_str.isdigit():
            return [int(steps_str)]

        # Handle list format [1,2,4] or [1,3-4]
        if steps_str.startswith("[") and steps_str.endswith("]"):
            content = steps_str[1:-1].strip()
            if not content:
                return []

            parts = [part.strip() for part in content.split(",")]
            result = []

            for part in parts:
                if "-" in part:
                    # Handle range format, such as "3-4"
                    range_parts = [p.strip() for p in part.split("-")]
                    if (
                        len(range_parts) == 2
                        and range_parts[0].isdigit()
                        and range_parts[1].isdigit()
                    ):
                        start = int(range_parts[0])
                        end = int(range_parts[1])
                        result.extend(range(start, end + 1))
                    else:
                        raise ValueError(f"Invalid range format: {part}")
                elif part.isdigit():
                    # Handle single number
                    result.append(int(part))
                else:
                    raise ValueError(f"Invalid step format: {part}")

            return result

        # If not list format, try as single number
        if steps_str.isdigit():
            return [int(steps_str)]

        raise ValueError(
            f"Invalid step parameter format: {steps_str}. Supported formats: 1 or [1,2,4] or [1,3-4]"
        )

    def _parse_arguments_to_config(self) -> Tuple[AnalysisConfig, argparse.Namespace]:
        """Parse command line arguments and create config object"""
        parser = argparse.ArgumentParser(
            description="abacus2deepmd: Advanced analysis tool specifically for processing ABACUS molecular dynamics trajectories, supporting intelligent conformation sampling and DeepMD data export",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
            Process step description:
            1 - Sampling: Execute system discovery, sampling
            2 - Independent DeepMD export: Only export existing sampling results to DeepMD format
            3 - Sampling effect comparison: Execute comparison analysis of different sampling methods
            4 - Power parameter test: Test sampling effect comparison of different power_p parameters
            """,
        )

        # Core parameters
        parser.add_argument(
            "-r", "--sample_ratio", type=float, default=0.1, help="Sampling ratio"
        )
        parser.add_argument(
            "-p",
            "--power_p",
            type=float,
            default=-0.5,
            help="p value of power mean distance",
        )
        parser.add_argument(
            "-v",
            "--pca_variance_ratio",
            type=float,
            default=0.90,
            help="PCA dimensionality reduction cumulative variance contribution rate",
        )

        # Runtime configuration
        parser.add_argument(
            "-w",
            "--workers",
            type=int,
            default=-1,
            help="Number of parallel worker processes",
        )
        parser.add_argument(
            "-o",
            "--output_dir",
            type=str,
            default="analysis_results",
            help="Output root directory",
        )
        parser.add_argument(
            "-s",
            "--search_path",
            nargs="*",
            default=None,
            help="Recursive search paths",
        )
        parser.add_argument(
            "-i",
            "--include_project",
            action="store_true",
            help="Allow searching project own directory",
        )
        parser.add_argument(
            "-f", "--force_recompute", action="store_true", help="Force recompute"
        )

        # Process control
        parser.add_argument(
            "--steps",
            type=str,
            default="[1,2,3]",
            help='Analysis steps to execute: 1=Sampling, 2=DeepMD export, 3=Sampling effect comparison, 4=Power parameter test. Default: [1,2,3]. Supported formats: single number(1), list([1,2,4]), range([1,3-4]). Example: --steps 1 or --steps "[1,3-4]"',
        )

        # Power parameter test related parameters
        parser.add_argument(
            "--max_systems",
            type=int,
            default=64,
            help="Maximum number of systems used in Power parameter test (None means use all discovered systems)",
        )

        args = parser.parse_args()

        # Determine steps to execute
        steps_to_run = []

        if args.steps is not None:
            # User explicitly specified steps, use new parsing function
            steps_to_run = self._parse_steps_argument(args.steps)
        else:
            # Default execute all steps
            steps_to_run = [1, 2, 3, 4]

        # Validate step numbers
        valid_steps = [1, 2, 3, 4]
        invalid_steps = [s for s in steps_to_run if s not in valid_steps]
        if invalid_steps:
            raise ValueError(
                f"Invalid step numbers: {invalid_steps}. Valid steps: {valid_steps}"
            )

        config = AnalysisConfig(
            sample_ratio=args.sample_ratio,
            power_p=args.power_p,
            pca_variance_ratio=args.pca_variance_ratio,
            workers=args.workers,
            output_dir=args.output_dir,
            search_paths=args.search_path or [],
            include_project=args.include_project,
            force_recompute=args.force_recompute,
            steps=steps_to_run,
            max_systems=args.max_systems,
        )

        return config, args

    def _log_startup_info(self, config: AnalysisConfig) -> None:
        """Log startup information"""
        search_paths = self.orchestrator.resolve_search_paths()
        search_info = f"Search paths: {search_paths if search_paths else '(parent directory of current directory)'}"

        # Step information
        step_names = {
            1: "Sampling",
            2: "DeepMD export",
            3: "Sampling effect comparison",
            4: "Power parameter test",
        }
        steps_info = [
            f"{step}({step_names.get(step, 'Unknown')})" for step in config.steps
        ]
        steps_str = ",".join(steps_info)

        workers = self.orchestrator.determine_workers()

        self.orchestrator.logger.info(
            f"ABACUS main analyzer started [Execution steps: {steps_str}] | Sampling ratio: {config.sample_ratio} | Worker processes: {workers}"
        )
        self.orchestrator.logger.info(search_info)
        self.orchestrator.logger.info(
            f"Project directory shielding: {'Off' if config.include_project else 'On'}"
        )
        self.orchestrator.logger.info(
            f"Force recompute: {'Yes' if config.force_recompute else 'No'}"
        )

    def _execute_workflow(self, config: AnalysisConfig, start_time: float) -> None:
        """Execute main workflow (based on step list)"""
        self.orchestrator.logger.info(
            f"Start executing analysis process, steps: {config.steps}"
        )

        # Initialize shared data
        analysis_results = []
        path_manager = None
        actual_output_dir = None

        # Step 1: Sampling
        if 1 in config.steps:
            self.orchestrator.current_step = 1
            self.orchestrator.logger.info("Execute step 1: Sampling")
            analysis_results, path_manager, actual_output_dir = (
                self.workflow_executor.execute_sampling_step(config)
            )

        # Step 2: DeepMD export
        if 2 in config.steps:
            self.orchestrator.current_step = 2
            self.orchestrator.logger.info("Execute step 2: DeepMD export")
            if analysis_results:  # If step 1 has been executed, use its results
                self.workflow_executor.execute_deepmd_step(
                    analysis_results, path_manager, actual_output_dir
                )
            else:  # If only execute step 2, load existing data
                analysis_results, path_manager, actual_output_dir = (
                    self.workflow_executor.load_existing_results_for_deepmd(config)
                )
                if analysis_results:
                    self.workflow_executor.execute_deepmd_step(
                        analysis_results, path_manager, actual_output_dir
                    )
                else:
                    self.orchestrator.logger.warning(
                        "Step 2: No available sampling data found, skip DeepMD export"
                    )

        # Step 3: Sampling effect comparison
        if 3 in config.steps:
            self.orchestrator.current_step = 3
            self.orchestrator.logger.info("Execute step 3: Sampling effect comparison")
            if analysis_results:  # If previous steps have been executed
                self.workflow_executor.execute_sampling_compare_step(actual_output_dir)
            else:  # If only execute step 3, need to load existing data
                analysis_results, path_manager, actual_output_dir = (
                    self.workflow_executor.load_existing_results_for_compare(config)
                )
                if (
                    path_manager
                ):  # As long as path_manager exists, sampling comparison can be executed
                    self.workflow_executor.execute_sampling_compare_step(
                        actual_output_dir
                    )
                else:
                    self.orchestrator.logger.warning(
                        "Step 3: No available analysis data found, skip sampling effect comparison"
                    )

        # Step 4: Power parameter test
        if 4 in config.steps:
            self.orchestrator.current_step = 4
            self.orchestrator.logger.info("Execute step 4: Power parameter test")
            # Power parameter test can run independently, does not depend on previous steps
            self.workflow_executor.execute_power_parameter_test_step(
                max_workers=config.workers,
                max_systems=getattr(config, "max_systems", None),
                sample_ratio=config.sample_ratio,
                pca_variance_ratio=config.pca_variance_ratio,
                search_paths=self.orchestrator.resolve_search_paths(),
                include_project=config.include_project,
                force_recompute=config.force_recompute,
            )

        # Final statistics and cleanup
        if analysis_results and path_manager:
            self.workflow_executor.finalize_analysis(
                analysis_results, path_manager, start_time, actual_output_dir
            )


def main():
    app = MainApp()
    app.run()


if __name__ == "__main__":
    main()
