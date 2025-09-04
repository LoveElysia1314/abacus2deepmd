#!/usr/bin/env python

import logging
import os
import warnings
from typing import Optional, List, Dict, Tuple
import dpdata  # type: ignore
from ..io.stru_parser import FrameData, StrUParser
from ..io.file_utils import FileUtils, DIR_SINGLE, FRAME_PREFIX

# Suppress dpdata duplicate type registration warnings
warnings.filterwarnings(
    "ignore", message="Data type move is registered twice", category=UserWarning
)


class ResultSaver:
    """Result saver class responsible for saving analysis results to CSV files"""

    @staticmethod
    def should_skip_analysis(output_dir: str, system_name: str) -> bool:
        """Check if system analysis should be skipped

        Args:
            output_dir: Output directory
            system_name: System name

        Returns:
            Returns True if corresponding file exists in single_analysis and analysis_targets.json exists
        """
        # Directory candidates
        single_dir_candidates = [os.path.join(output_dir, DIR_SINGLE)]
        single_analysis_dir = None
        for c in single_dir_candidates:
            if os.path.isdir(c):
                single_analysis_dir = c
                break
        if single_analysis_dir is None:
            return False
        # Filename
        frame_metrics_file = os.path.join(
            single_analysis_dir, f"{FRAME_PREFIX}{system_name}.csv"
        )

        # Check analysis_targets.json file
        targets_file = os.path.join(output_dir, "analysis_targets.json")

        # Skip analysis if both files exist
        return os.path.exists(frame_metrics_file) and os.path.exists(
            targets_file
        )  # -------------------- Streaming / Per-System Saving Enhancements --------------------

    @staticmethod
    def save_single_system(
        output_dir: str,
        result: Tuple,
        sampling_only: bool = False,
        flush_targets_hook: Optional[callable] = None,
    ) -> None:
        """Stream save all available results for a single system (called immediately after system completion).

        Automatically degrades based on current mode and available data:
          - Full mode (result length >=5): write frame_metrics
          - Sampling only mode/data insufficient: only write sampling frame information (sampling frames have mainly been refreshed in analysis_targets.json by caller)

        Args:
            output_dir: run_* directory
            result: tuple returned by analyse_system
            sampling_only: whether sampling_only mode
            flush_targets_hook: optional callback for caller to refresh analysis_targets.json after success
        """
        logger = logging.getLogger(__name__)
        if not result:
            return
        try:
            metrics = result[0]
            # sampling_only mode returns only (metrics, frames)
            frames = result[1] if len(result) > 1 else []
            # Full mode expects 5 elements
            pca_components_data = (
                result[3] if not sampling_only and len(result) > 3 else None
            )

            # 1) frame_{system}.csv (full mode only)
            if (not sampling_only) and frames and pca_components_data is not None:
                try:
                    sampled_frames = [
                        fid for fid in getattr(metrics, "sampled_frames", [])
                    ]
                    # Only save CSV file when sampled frames are not empty
                    if sampled_frames:
                        ResultSaver.save_frame_metrics(
                            output_dir=output_dir,
                            system_name=metrics.system_name,
                            frames=frames,
                            sampled_frames=sampled_frames,
                            pca_components_data=pca_components_data,
                        )
                except Exception as fe:
                    logger.warning(
                        f"Single system frame metrics write failed {metrics.system_name}: {fe}"
                    )

            # 4) Optional refresh analysis_targets.json (frequency controlled by orchestrator hook)
            if flush_targets_hook:
                try:
                    flush_targets_hook()
                except Exception as he:
                    logger.warning(f"analysis_targets.json refresh failed: {he}")
        except Exception as e:
            logger.error(f"Streaming save system results failed: {e}")

    @staticmethod
    def save_frame_metrics(
        output_dir: str,
        system_name: str,
        frames: List[FrameData],
        sampled_frames: List[int],
        pca_components_data: List[Dict] = None,
    ) -> None:
        """Save individual frame metrics to CSV file, with comprehensive vector components"""
        single_analysis_dir = os.path.join(output_dir, DIR_SINGLE)
        FileUtils.ensure_dir(single_analysis_dir)
        csv_path = os.path.join(single_analysis_dir, f"{FRAME_PREFIX}{system_name}.csv")

        headers = ["Frame_ID", "Selected", "E_Proc"]
        max_pc = 0
        if pca_components_data:
            for item in pca_components_data:
                for key in item.keys():
                    if key.startswith("PC"):
                        pc_num = int(key[2:])
                        max_pc = max(max_pc, pc_num)
            headers.extend([f"C{i}" for i in range(1, max_pc + 1)])

        sampled_set = set(sampled_frames)
        pca_lookup = {}
        if pca_components_data:
            for item in pca_components_data:
                frame_id = item.get("frame")
                if frame_id is not None:
                    pca_lookup[frame_id] = item

        rows = []
        for i, frame in enumerate(frames):
            selected = 1 if frame.frame_id in sampled_set else 0
            row = [frame.frame_id, selected]
            energy_standardized = (
                frame.energy_standardized
                if frame.energy_standardized is not None
                else ""
            )
            row.append(energy_standardized)
            if pca_components_data and frame.frame_id in pca_lookup:
                pca_item = pca_lookup[frame.frame_id]
                for pc_num in range(1, max_pc + 1):
                    pc_key = f"PC{pc_num}"
                    pc_value = pca_item.get(pc_key, 0.0)
                    row.append(f"{pc_value:.6f}")
            elif pca_components_data:
                for pc_num in range(1, max_pc + 1):
                    row.append("0.000000")
            rows.append(row)

        FileUtils.safe_write_csv(csv_path, rows, headers=headers, encoding="utf-8-sig")

    # Old incremental and sampling record aggregation functions removed

    # ---- DeepMD Export Functionality (merged from deepmd_exporter.py) ----
    ALL_TYPE_MAP = [
        "H",
        "He",
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "K",
        "Ca",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Ge",
        "As",
        "Se",
        "Br",
        "Kr",
        "Rb",
        "Sr",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        "Tl",
        "Pb",
        "Bi",
        "Po",
        "At",
        "Rn",
        "Fr",
        "Ra",
        "Ac",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu",
        "Am",
        "Cm",
        "Bk",
        "Cf",
        "Es",
        "Fm",
        "Md",
        "No",
        "Lr",
        "Rf",
        "Db",
        "Sg",
        "Bh",
        "Hs",
        "Mt",
        "Ds",
        "Rg",
        "Cn",
        "Nh",
        "Fl",
        "Mc",
        "Lv",
        "Ts",
        "Og",
    ]

    @staticmethod
    def export_sampled_frames_to_deepmd(
        system_path: str,
        sampled_frame_ids: List[int],
        output_root: str,
        system_name: str,
        md_dumpfreq: int,
        logger: Optional[logging.Logger] = None,
        force: bool = False,
    ) -> Optional[str]:
        """Export sampled frames to DeepMD format

        Args:
            system_path: ABACUS output directory path
            sampled_frame_ids: List of sampled frame IDs
            output_root: Output root directory
            system_name: System name
            md_dumpfreq: MD dump frequency, used for frame ID to index mapping
            logger: Logger instance
            force: Whether to force re-export

        Returns:
            Output directory path on success, None on failure
        """
        if logger is None:
            logger = logging.getLogger(__name__)

        if not sampled_frame_ids:
            logger.debug(f"[deepmd-export] {system_name} no sampled frames, skipping")
            return None

        target_dir = os.path.join(output_root, system_name)
        marker_file = os.path.join(target_dir, "export.done")

        if os.path.isdir(target_dir) and os.path.exists(marker_file) and not force:
            logger.debug(
                f"[deepmd-export] {system_name} already exists and not force overwrite, skipping"
            )
            return target_dir

        try:
            # Load dpdata system
            ls = dpdata.LabeledSystem(
                system_path, fmt="abacus/lcao/md", type_map=ResultSaver.ALL_TYPE_MAP
            )

            # Use md_dumpfreq for frame ID to index conversion
            logger.debug(
                f"[deepmd-export] {system_name} using efficient mapping (md_dumpfreq={md_dumpfreq})"
            )
            valid_indices = []
            for frame_id in sampled_frame_ids:
                dpdata_idx = frame_id // md_dumpfreq
                valid_indices.append(dpdata_idx)

            if not valid_indices:
                logger.warning(
                    f"[deepmd-export] {system_name} no sampled frames, skipping"
                )
                return None

            # Export to DeepMD format
            sub_ls = ls[valid_indices]
            os.makedirs(target_dir, exist_ok=True)
            sub_ls.to_deepmd_npy(target_dir)

            # Write marker file
            with open(marker_file, "w", encoding="utf-8") as f:
                f.write(f"frames={len(valid_indices)}\n")

            logger.debug(
                f"[deepmd-export] export {system_name} successful, frames={len(valid_indices)} -> {target_dir}"
            )
            return target_dir

        except Exception as e:
            logger.error(f"[deepmd-export] export {system_name} failed: {e}")
            return None

    # ---- DeepMD Export Worker Function ----


def _deepmd_export_worker(task: tuple):
    """DeepMD export parallel worker function

    Parameters
    ----------
    task: tuple(system_path, system_name, sampled_frame_ids, output_root, force, md_dumpfreq)
    Returns
    -------
    tuple: (system_name, error_msg) - error_msg is None on success, system_name is None on failure
    """
    try:
        # Validate input parameters
        if not isinstance(task, tuple) or len(task) != 6:
            return (None, "Invalid task format")

        system_path, system_name, sampled_frame_ids, output_root, force, md_dumpfreq = (
            task
        )

        if not sampled_frame_ids:
            return (None, "No sampled_frame_ids")

        if not system_path or not system_name:
            return (None, "Missing system_path or system_name")

        import os
        import dpdata

        ls = dpdata.LabeledSystem(system_path, fmt="abacus/lcao/md")
        result = ResultSaver.export_sampled_frames_to_deepmd(
            system_path=system_path,
            sampled_frame_ids=sampled_frame_ids,
            output_root=output_root,
            system_name=system_name,
            md_dumpfreq=md_dumpfreq,  # Pass md_dumpfreq
        )

        if result is None:
            return (None, "Export function returned None")

        return (system_name, None)
    except Exception as e:
        return (None, f"{type(e).__name__}: {e}")


class DeepMDTaskBuilder:
    """DeepMD task builder"""

    @staticmethod
    def resolve_md_dumpfreq(system_name: str, system_path: str, path_manager) -> int:
        """Parse md_dumpfreq parameter

        Args:
            system_name: System name
            system_path: System path
            path_manager: PathManager instance

        Returns:
            md_dumpfreq value
        """
        md_dumpfreq = None
        try:
            # Find corresponding md_dumpfreq from path_manager's targets
            for target in path_manager.targets:
                if target.system_name == system_name:
                    md_dumpfreq = getattr(target, "md_dumpfreq", None)
                    break
            if md_dumpfreq is None:
                # Fallback: Parse INPUT file directly
                parser = StrUParser()
                input_file = os.path.join(system_path, "INPUT")
                md_dumpfreq, _ = parser.parse_md_parameters(input_file)
        except Exception as e:
            # Default value
            md_dumpfreq = 10

        return md_dumpfreq

    @staticmethod
    def build_deepmd_tasks(
        analysis_results: List[tuple], path_manager, actual_output_dir: str, force: bool
    ) -> List[tuple]:
        """Build DeepMD export task list

        Args:
            analysis_results: Analysis result list
            path_manager: PathManager instance
            actual_output_dir: Output directory
            force: Whether to force recompute

        Returns:
            Task list, each task is (system_path, system_name, sampled_frame_ids, output_root, force, md_dumpfreq)
        """
        tasks = []
        out_root = os.path.join(actual_output_dir, "deepmd_npy_per_system")
        os.makedirs(out_root, exist_ok=True)

        for result in analysis_results:
            if not result or len(result) < 2:
                continue
            metrics_obj = result[0]
            system_name = getattr(metrics_obj, "system_name", None)
            system_path = getattr(metrics_obj, "system_path", None)

            # Get sampled_frames from path_manager.targets instead of metrics object
            sampled_frame_ids = []
            if system_name:
                for target in path_manager.targets:
                    if target.system_name == system_name:
                        sampled_frame_ids = target.sampled_frames or []
                        break

            if (
                system_name
                and system_path
                and sampled_frame_ids
                and len(sampled_frame_ids) > 0
            ):
                # Get md_dumpfreq
                md_dumpfreq = DeepMDTaskBuilder.resolve_md_dumpfreq(
                    system_name, system_path, path_manager
                )

                tasks.append(
                    (
                        system_path,
                        system_name,
                        sampled_frame_ids,
                        out_root,
                        True,  # Always force re-export, don't use cache
                        md_dumpfreq,
                    )
                )

        # Sort by system name to ensure task order matches JSON
        from ..utils.common import CommonUtils

        tasks.sort(
            key=lambda task: CommonUtils.sort_system_names([task[1]])[0]
        )  # task[1] is system_name

        return tasks
