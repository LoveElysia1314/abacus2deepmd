#!/usr/bin/env python

import glob
import json
import logging
import os
import copy
import csv
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from ..utils.common import CommonUtils
from .file_utils import (
    FileUtils,
    LightweightSystemRecord,
    lightweight_discover_systems,
    AnalysisTarget,
    SCHEMA_VERSION,
    SAMPLING_ALGORITHM_VERSION,
    DIR_SINGLE,
    DIR_SAMPLING_COMP,
    FRAME_PREFIX,
)
from ..io.result_saver import ResultSaver
from .stru_parser import StrUParser
from ..core.metrics import TrajectoryMetrics


class PathManager:
    def __init__(self, output_dir: str = None):
        if output_dir is None:
            output_dir = os.path.join(FileUtils.get_project_root(), "analysis_results")
        self.base_output_dir = output_dir
        self.output_dir = output_dir  # Will be updated in set_output_dir_for_params
        self.targets_file = None  # Will be set in set_output_dir_for_params
        self.logger = logging.getLogger(__name__)
        self.targets: List[AnalysisTarget] = []
        self.mol_groups: Dict[str, List[AnalysisTarget]] = {}

    def set_output_dir_for_params(self, analysis_params: Dict[str, Any]) -> str:
        """Set output directory according to analysis parameters, achieve parameter isolation"""

        key_params = {
            "sample_ratio": analysis_params.get("sample_ratio", 0.1),
            "power_p": analysis_params.get("power_p", -0.5),
            "pca_variance_ratio": analysis_params.get("pca_variance_ratio", 0.90),
        }

        # Generate concise directory name based on short options (use short option letters, no case distinction)
        # Mapping: sample_ratio -> r, power_p -> p, pca_variance_ratio -> v
        short_map = {
            "sample_ratio": "r",
            "power_p": "p",
            "pca_variance_ratio": "v",
        }

        # Keep fixed order: r, p, v
        ordered_keys = ["sample_ratio", "power_p", "pca_variance_ratio"]
        parts = []
        for key in ordered_keys:
            value = key_params.get(key)
            short = short_map.get(key, key)
            # All values are float, use general format
            formatted = f"{value:g}"
            parts.append(f"{short}{formatted}")

        dir_name = "run_" + "_".join(parts)
        self.output_dir = os.path.join(self.base_output_dir, dir_name)

        # Update related file paths
        self.targets_file = os.path.join(self.output_dir, "analysis_targets.json")
        # Subdirectory creation
        for sub in (DIR_SINGLE, DIR_SAMPLING_COMP):
            os.makedirs(os.path.join(self.output_dir, sub), exist_ok=True)

        # Ensure directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        self.logger.debug(f"Set parameter-specific output directory: {self.output_dir}")
        return self.output_dir

    def check_existing_complete_results(self) -> bool:
        """Check if the current parameter directory already has complete analysis results"""
        if not self.output_dir or not os.path.exists(self.output_dir):
            return False

        # Check key result files
        targets_file = self.targets_file

        if not os.path.exists(targets_file):
            self.logger.debug(
                "Key result file does not exist, needs to be recalculated"
            )
            return False

        # Check completion status in analysis_targets.json
        try:
            with open(targets_file, encoding="utf-8") as f:
                data = json.load(f)

            # Check if all systems are completed
            total_systems = 0
            completed_systems = 0

            for mol_data in data.get("molecules", {}).values():
                for system_name, system_data in mol_data.get("systems", {}).items():
                    total_systems += 1
                    # Check if there is a corresponding CSV file in the single_analysis directory
                    if ResultSaver.should_skip_analysis(self.output_dir, system_name):
                        completed_systems += 1

            if total_systems == 0:
                self.logger.debug("No system records in target file")
                return False

            # If all systems are completed, return True to take the fast path
            if completed_systems == total_systems:
                return True
            else:
                return False

        except Exception as e:
            self.logger.warning(f"Error checking existing results: {str(e)}")
            return False

    def load_from_discovery(
        self, records_or_paths: Union[List[LightweightSystemRecord], List[str]]
    ) -> None:
        """Load analysis targets from discovery results

        Args:
                records_or_paths: List of LightweightSystemRecord or search paths
        """
        # Determine if records or search_paths are passed
        if records_or_paths and isinstance(
            records_or_paths[0], LightweightSystemRecord
        ):
            # Passed records
            records = records_or_paths
        else:
            # Passed search_paths, need to discover
            records = lightweight_discover_systems(records_or_paths)

        self.targets.clear()
        self.mol_groups.clear()
        total_targets = 0
        skipped_count = 0

        for record in records:
            # Get STRU file list
            stru_files = glob.glob(os.path.join(record.stru_dir, "STRU_MD_*"))
            if not stru_files:
                self.logger.warning(
                    f"System {record.system_name} did not find STRU_MD files"
                )
                skipped_count += 1
                continue

            # Parse MD parameters and calculate expected frame count
            md_dumpfreq = 1  # Will be stored in target object
            md_nstep = 1  # Local variable for calculating expected_frame_count
            expected_frame_count = 0
            try:
                parser = StrUParser()
                input_file = os.path.join(
                    record.system_path, "INPUT"
                )  # Modified to parse from system_path/INPUT
                md_dumpfreq, md_nstep = parser.parse_md_parameters(input_file)
                expected_frame_count = md_nstep // md_dumpfreq + 1
                self.logger.debug(
                    f"System {record.system_name}: md_dumpfreq={md_dumpfreq}, md_nstep={md_nstep}, expected_frame_count={expected_frame_count}"
                )
            except Exception as e:
                self.logger.warning(
                    f"System {record.system_name} failed to parse MD parameters, using default values: {e}"
                )

            # Get STRU file list
            stru_files = glob.glob(os.path.join(record.stru_dir, "STRU_MD_*"))
            if not stru_files:
                self.logger.warning(
                    f"System {record.system_name} did not find STRU_MD files"
                )
                skipped_count += 1
                continue

            # Apply md_dumpfreq filtering
            try:
                original_count = len(stru_files)
                stru_files = parser.select_frames_by_md_dumpfreq(
                    stru_files, md_dumpfreq
                )
                if not stru_files:
                    self.logger.warning(
                        f"System {record.system_name} has no available frames after filtering by md_dumpfreq={md_dumpfreq} (original {original_count} frames)"
                    )
                    skipped_count += 1
                    continue
                if len(stru_files) != original_count:
                    self.logger.debug(
                        f"System {record.system_name}: Filtered {len(stru_files)}/{original_count} frames by md_dumpfreq={md_dumpfreq} (excluding 0 frames)"
                    )
            except Exception as e:
                self.logger.warning(
                    f"System {record.system_name} failed to apply md_dumpfreq filtering, using all frames: {e}"
                )

            # Check if actual frame count meets expectations
            actual_frame_count = len(stru_files)
            if actual_frame_count < expected_frame_count:
                # Skip silently to reduce log redundancy without specific system information
                skipped_count += 1
                continue

            target = AnalysisTarget(
                system_path=record.system_path,
                mol_id=record.mol_id,
                conf=record.conf,
                temperature=record.temperature,
                stru_files=stru_files,
                creation_time=getattr(record, "ctime", 0),
                expected_frame_count=expected_frame_count,
                md_dumpfreq=md_dumpfreq,
            )
            self.targets.append(target)
            total_targets += 1

        # Group by mol_id
        self._rebuild_mol_groups()

        self.logger.debug(
            f"PathManager loaded {len(self.mol_groups)} molecules with {total_targets} targets"
        )
        if skipped_count > 0:
            self.logger.info(
                f"Skipped {skipped_count} systems due to insufficient frame count"
            )

    def deduplicate_targets(self) -> None:
        """Remove duplicate analysis targets, keeping the one with latest modification time based on system name (system_key)"""
        if not self.targets:
            return

        # Group by system_key
        system_groups = {}
        for target in self.targets:
            key = target.system_key  # mol_id_conf_id_T_tempK
            if key not in system_groups:
                system_groups[key] = []
            system_groups[key].append(target)

        # Deduplication: keep the one with latest modification time for each group
        deduplicated_targets = []
        duplicates_removed = 0

        for key, targets_group in system_groups.items():
            if len(targets_group) == 1:
                deduplicated_targets.extend(targets_group)
            else:
                # Multiple duplicates, select the one with latest modification time
                latest_target = max(targets_group, key=lambda t: t.creation_time)
                deduplicated_targets.append(latest_target)
                duplicates_removed += len(targets_group) - 1

                # Record removed duplicate items
                removed_paths = [
                    t.system_path for t in targets_group if t != latest_target
                ]
                self.logger.info(
                    f"Deduplication: System {key} keeps {latest_target.system_path}"
                )
                for removed_path in removed_paths:
                    self.logger.info(f"  Removed duplicate: {removed_path}")

        # Update targets and mol_groups
        self.targets = deduplicated_targets
        self._rebuild_mol_groups()

        if duplicates_removed > 0:
            self.logger.info(
                f"Deduplication completed: Removed {duplicates_removed} duplicate targets, kept {len(self.targets)}"
            )

    def _rebuild_mol_groups(self) -> None:
        """Rebuild molecule groups"""
        self.mol_groups.clear()
        for target in self.targets:
            mol_id = target.mol_id
            if mol_id not in self.mol_groups:
                self.mol_groups[mol_id] = []
            self.mol_groups[mol_id].append(target)

    def load_targets(self) -> bool:
        if not os.path.exists(self.targets_file):
            return False
        try:
            with open(self.targets_file, encoding="utf-8") as f:
                targets_data = json.load(f)
            self.targets.clear()
            self.mol_groups.clear()
            for target_dict in targets_data:
                target_dict.pop("created_at", None)
                target = AnalysisTarget(**target_dict)
                self.targets.append(target)
                mol_id = target.mol_id
                if mol_id not in self.mol_groups:
                    self.mol_groups[mol_id] = []
                self.mol_groups[mol_id].append(target)
            self.logger.debug(f"Loaded {len(self.targets)} analysis targets from file")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load targets: {str(e)}")
            return False

    def get_all_targets(self) -> List[AnalysisTarget]:
        return self.targets.copy()

    def build_results_for_existing_targets(
        self, restrict_systems: Optional[List[str]] = None
    ) -> List[tuple]:
        """Build analysis results list from existing analysis_targets.json

        Args:
            restrict_systems: List of restricted system names, processes all systems when None

        Returns:
            Analysis results list [(metrics, sampled_frames), ...]
        """
        # Delayed import to avoid circular dependency
        targets_file = os.path.join(self.output_dir, "analysis_targets.json")
        if not os.path.exists(targets_file):
            self.logger.warning(f"analysis_targets.json file not found: {targets_file}")
            return []

        try:
            with open(targets_file, "r", encoding="utf-8") as f:
                targets_data = json.load(f)

            analysis_results = []
            molecules = targets_data.get("molecules", {})

            for mol_id, mol_data in molecules.items():
                systems = mol_data.get("systems", {})
                for system_name, system_data in systems.items():
                    if restrict_systems and system_name not in restrict_systems:
                        continue

                    system_path = system_data.get("system_path", "")
                    sampled_frames = []
                    # robustly parse sampled_frames (may be str or list)
                    sampled_frames_raw = system_data.get("sampled_frames", [])
                    if isinstance(sampled_frames_raw, str):
                        try:
                            sampled_frames = json.loads(sampled_frames_raw)
                        except Exception:
                            self.logger.warning(
                                f"Unable to parse sampled frame data: {system_name}, content: {sampled_frames_raw}"
                            )
                            sampled_frames = []
                    elif isinstance(sampled_frames_raw, list):
                        sampled_frames = sampled_frames_raw
                    else:
                        sampled_frames = []
                    # Filter non-int types
                    sampled_frames = [
                        int(x)
                        for x in sampled_frames
                        if isinstance(x, int)
                        or (isinstance(x, float) and x.is_integer())
                    ]
                    if not system_path or not sampled_frames:
                        self.logger.warning(
                            f"Incomplete system data, skipping: {system_name}"
                        )
                        continue

                    # Parse system information
                    mol_id_parsed, conf, temperature = CommonUtils.parse_system_name(
                        system_name
                    )

                    # Create TrajectoryMetrics object
                    metrics = TrajectoryMetrics(
                        system_name=system_name,
                        mol_id=mol_id_parsed,
                        conf=conf,
                        temperature=temperature,
                        system_path=system_path,
                    )
                    metrics.sampled_frames = sampled_frames
                    metrics.num_frames = len(sampled_frames)
                    # Construct analysis result tuple (format: (metrics, sampled_frames))
                    result = (metrics, sampled_frames)
                    analysis_results.append(result)

        except Exception as e:
            self.logger.error(f"Failed to read analysis_targets.json file: {e}")
            return []

        return analysis_results

    def save_analysis_targets(self, analysis_params: Dict[str, Any] = None) -> str:
        """Save / incrementally update analysis_targets.json (schema v2)

        Goals:
          1. Do not overwrite existing non-empty sampled results with empty frames (unless new value is non-empty)
          2. Unify sampled_frames as List[int]
          3. Write system-level metadata: frame_count / sampled_count / status / integrity / sampled_origin
          4. Record sampling parameters (power_p_at_sampling / pca_ratio_at_sampling)
          5. Write schema_version / writer_version / params_hash in metadata
        """

        if not self.targets_file:
            raise ValueError(
                "Output directory not set, please call set_output_dir_for_params first"
            )

        targets_file = self.targets_file
        temp_file = targets_file + ".tmp"

        try:

            # Empty file protection
            if not os.path.exists(targets_file) or os.path.getsize(targets_file) == 0:
                old_data = None
            else:
                try:
                    with open(targets_file, "r", encoding="utf-8") as f:
                        old_raw = f.read().strip()
                    if not old_raw:
                        old_data = None
                    else:
                        old_data = json.loads(old_raw)
                        old_data = self._migrate_targets_data(old_data)
                except Exception:
                    # If old file is corrupted, backup and restart
                    try:
                        corrupt_backup = targets_file + ".corrupt.bak"
                        os.replace(targets_file, corrupt_backup)
                        self.logger.warning(
                            f"Detected corrupted analysis_targets.json, backed up as {corrupt_backup} and regenerating"
                        )
                    except Exception:
                        pass
                    old_data = None

            data = (
                copy.deepcopy(old_data)
                if old_data
                else {"metadata": {}, "summary": {}, "molecules": {}}
            )

            # metadata - prevent None override
            data["metadata"] = {
                "schema_version": SCHEMA_VERSION,
                "sampling_algorithm_version": SAMPLING_ALGORITHM_VERSION,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "generator": "ABACUS-STRU-Analyser",
                "writer_version": "1.1.0",
                "analysis_params": analysis_params or {},
                "output_directory": self.output_dir,
            }

            # summary - prevent None override
            old_summary = data.get("summary", {})
            data["summary"] = {
                "total_molecules": (
                    len(self.mol_groups)
                    if self.mol_groups
                    else old_summary.get("total_molecules", 0)
                ),
                "total_systems": (
                    len(self.targets)
                    if self.targets
                    else old_summary.get("total_systems", 0)
                ),
            }

            # systems
            # Sort molecule IDs to ensure consistent output order
            from ..utils.common import CommonUtils

            sorted_mol_ids = CommonUtils.sort_system_names(list(self.mol_groups.keys()))

            for mol_id in sorted_mol_ids:
                targets = self.mol_groups[mol_id]
                mol_entry = data["molecules"].setdefault(
                    mol_id, {"molecule_id": mol_id, "system_count": 0, "systems": {}}
                )
                mol_entry["system_count"] = len(targets)

                # Sort system names
                sorted_system_names = CommonUtils.sort_system_names(
                    [t.system_name for t in targets]
                )
                system_name_to_target = {t.system_name: t for t in targets}

                for sys_name in sorted_system_names:
                    target = system_name_to_target[sys_name]
                    sys_entry = mol_entry["systems"].get(sys_name, {})

                    # Old sampled frames
                    old_frames_raw = sys_entry.get("sampled_frames", [])
                    if isinstance(old_frames_raw, str):
                        try:
                            old_frames = json.loads(old_frames_raw)
                        except Exception:
                            old_frames = []
                    else:
                        old_frames = old_frames_raw or []

                    new_frames = target.sampled_frames or []
                    if (not new_frames) and old_frames:
                        # Retain old values
                        final_frames = old_frames
                    else:
                        final_frames = sorted({x for x in new_frames if x >= 0})

                    frame_count = len(target.stru_files)
                    sampled_count = len(final_frames)

                    # Store sampled_frames in compact string format to reduce file size
                    sampled_frames_serialized = json.dumps(
                        final_frames, separators=(",", ":")
                    )
                    # Construct structured system_entry, only keep system-related fields (whitelist)
                    system_entry = {
                        "system_path": target.system_path,
                        "frame_count": frame_count,
                        "expected_frame_count": target.expected_frame_count,
                        "md_dumpfreq": target.md_dumpfreq,  # Add md_dumpfreq field
                        "sampled_frames": sampled_frames_serialized,
                        "sampled_count": sampled_count,
                    }
                    # Remove fields with None values
                    system_entry = {
                        k: v for k, v in system_entry.items() if v is not None
                    }
                    mol_entry["systems"][sys_name] = system_entry

            # Atomic write
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(temp_file, targets_file)

            # self.logger.info(f"Analysis targets saved to: {targets_file}")
            self.logger.debug(
                f"Summary: {len(self.mol_groups)} molecules, {len(self.targets)} systems"
            )
            return targets_file

        except Exception as e:
            self.logger.error(f"Failed to save analysis targets: {str(e)}")
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e2:
                    self.logger.warning(
                        f"Failed to remove temporary file {temp_file}: {e2}"
                    )
            raise

    def save_analysis_targets_incremental(
        self, target: AnalysisTarget, analysis_params: Dict[str, Any] = None
    ) -> None:
        """Incrementally save analysis results for a single system to analysis_targets.json

        Args:
            target: Analysis target to save
            analysis_params: Analysis parameters (optional, used when creating file for the first time)
        """
        if not self.targets_file:
            self.logger.debug("targets_file not set, skipping incremental save")
            return

        targets_file = self.targets_file
        temp_file = targets_file + ".tmp"

        try:
            # Read existing data
            if os.path.exists(targets_file) and os.path.getsize(targets_file) > 0:
                try:
                    with open(targets_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    data = self._migrate_targets_data(data)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to read existing analysis_targets.json, will recreate: {e}"
                    )
                    data = None
            else:
                data = None

            # Initialize data structure
            if data is None:
                data = {
                    "metadata": {
                        "schema_version": SCHEMA_VERSION,
                        "sampling_algorithm_version": SAMPLING_ALGORITHM_VERSION,
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                        "generator": "ABACUS-STRU-Analyser",
                        "writer_version": "1.1.0",
                        "analysis_params": analysis_params or {},
                        "output_directory": self.output_dir,
                    },
                    "summary": {
                        "total_molecules": len(self.mol_groups),
                        "total_systems": len(self.targets),
                    },
                    "molecules": {},
                }

            # Ensure molecule entry exists
            mol_id = target.mol_id
            if mol_id not in data["molecules"]:
                data["molecules"][mol_id] = {
                    "molecule_id": mol_id,
                    "system_count": 0,
                    "systems": {},
                }

            mol_entry = data["molecules"][mol_id]

            # Update system information
            sys_name = target.system_name
            frame_count = len(target.stru_files)
            sampled_count = len(target.sampled_frames) if target.sampled_frames else 0

            # Store sampled_frames in compact string format as needed
            sampled_frames_serialized = json.dumps(
                target.sampled_frames or [], separators=(",", ":")
            )

            system_entry = {
                "system_path": target.system_path,
                "frame_count": frame_count,
                "expected_frame_count": target.expected_frame_count,
                "md_dumpfreq": target.md_dumpfreq,  # Add md_dumpfreq field
                "sampled_frames": sampled_frames_serialized,
                "sampled_count": sampled_count,
            }
            # Remove fields with None values
            system_entry = {k: v for k, v in system_entry.items() if v is not None}

            mol_entry["systems"][sys_name] = system_entry
            mol_entry["system_count"] = len(mol_entry["systems"])

            # Update summary information
            data["summary"]["total_systems"] = len(self.targets)

            # Atomic write
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(temp_file, targets_file)

            self.logger.debug(
                f"Incrementally saved system {sys_name} to analysis_targets.json"
            )

        except Exception as e:
            self.logger.warning(
                f"Failed to incrementally save system {target.system_name}: {e}"
            )
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass

    # ---------------- schema v2 helper methods ----------------
    def _migrate_targets_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate old schema (<=1) to schema v2 basic structure."""
        try:
            meta = data.get("metadata", {})
            schema_v = meta.get("schema_version") or meta.get("version")
            if schema_v == SCHEMA_VERSION:
                return data  # Already latest
            # Old version: need to convert sampled_frames string to list; add fields
            molecules = data.get("molecules")
            if not molecules:
                # Old format may be a direct list (supported by old load_targets logic) - keep as is
                return data
            for mol in molecules.values():
                systems = mol.get("systems", {})
                for sys_name, sys_entry in systems.items():
                    raw = sys_entry.get("sampled_frames", [])
                    if isinstance(raw, str):
                        try:
                            sys_entry["sampled_frames"] = json.loads(raw)
                        except Exception:
                            sys_entry["sampled_frames"] = []
                    if "sampled_count" not in sys_entry:
                        sf = sys_entry.get("sampled_frames") or []
                        sys_entry["sampled_count"] = len(sf)
                    # No longer add sampled_origin/status/integrity during migration
                    # For compatibility with subsequent processing, still keep as compact string format storage
                    if isinstance(sys_entry.get("sampled_frames"), list):
                        try:
                            sys_entry["sampled_frames"] = json.dumps(
                                sys_entry["sampled_frames"], separators=(",", ":")
                            )
                        except Exception:
                            pass
            data["metadata"]["schema_version"] = SCHEMA_VERSION
            data["metadata"]["writer_version"] = "1.1.0"
            return data
        except Exception:
            return data

    # _promote_status removed (simplified redundant status fields)

    def load_analysis_targets(self) -> bool:
        """Load analysis targets from new format analysis_targets.json"""
        targets_file = os.path.join(self.output_dir, "analysis_targets.json")
        if not os.path.exists(targets_file):
            return False

        try:
            with open(targets_file, encoding="utf-8") as f:
                analysis_data = json.load(f)
            # Migrate old schema
            analysis_data = self._migrate_targets_data(analysis_data)

            # Check format version
            if "metadata" not in analysis_data or "molecules" not in analysis_data:
                self.logger.warning(
                    "Analysis target file format incompatible, trying old format loading"
                )
                return self.load_targets()  # Fallback to old format

            # Check sampling algorithm version
            metadata = analysis_data.get("metadata", {})
            current_sampling_version = metadata.get("sampling_algorithm_version")
            if current_sampling_version != SAMPLING_ALGORITHM_VERSION:
                self.logger.warning(
                    f"Sampling algorithm version mismatch: file version {current_sampling_version} vs current version {SAMPLING_ALGORITHM_VERSION}, will clear file"
                )
                # Clear file
                try:
                    os.remove(targets_file)
                    self.logger.info("analysis_targets.json file cleared")
                except Exception as e:
                    self.logger.error(
                        f"Failed to clear analysis_targets.json file: {e}"
                    )
                return False

            self.targets.clear()
            self.mol_groups.clear()

            # Load molecule and system data
            for mol_id, mol_data in analysis_data["molecules"].items():
                targets_for_mol = []
                for system_name, system_data in mol_data["systems"].items():
                    # Parse mol_id, conf, temperature from system_name
                    # Format: struct_mol_{mol_id}_conf_{conf}_T{temperature}K
                    try:
                        parts = system_name.split("_")
                        if (
                            len(parts) >= 6
                            and parts[0] == "struct"
                            and parts[1] == "mol"
                            and parts[3] == "conf"
                        ):
                            parsed_mol_id = parts[2]
                            parsed_conf = parts[4]
                            temp_part = parts[5]  # T{temperature}K
                            if temp_part.startswith("T") and temp_part.endswith("K"):
                                parsed_temperature = temp_part[1:-1]  # Remove T and K
                            else:
                                parsed_temperature = "0"
                        else:
                            # Fallback to old parsing method
                            parsed_mol_id = system_data.get("mol_id", mol_id)
                            parsed_conf = system_data.get("conf", "0")
                            parsed_temperature = system_data.get("temperature", "0")
                    except Exception:
                        # Parsing failed, use default values
                        parsed_mol_id = system_data.get("mol_id", mol_id)
                        parsed_conf = system_data.get("conf", "0")
                        parsed_temperature = system_data.get("temperature", "0")

                    # Compatibility: If some fields are missing in JSON, use parsed values or safe defaults
                    # Handle sampled_frames: may be string format (compact mode) or list format
                    sampled_frames_data = system_data.get("sampled_frames") or []
                    if isinstance(sampled_frames_data, str):  # Compatibility
                        try:
                            sampled_frames_data = json.loads(sampled_frames_data)
                        except Exception:
                            sampled_frames_data = []
                    # Unified specification: sort and unique
                    if isinstance(sampled_frames_data, list):
                        sampled_frames_data = sorted(
                            {
                                int(x)
                                for x in sampled_frames_data
                                if isinstance(x, int) and x >= 0
                            }
                        )

                    target = AnalysisTarget(
                        system_path=system_data.get("system_path", ""),
                        mol_id=parsed_mol_id,
                        conf=parsed_conf,
                        temperature=parsed_temperature,
                        stru_files=[],  # will be repopulated in load_from_discovery
                        creation_time=system_data.get("creation_time", 0.0),
                        sampled_frames=sampled_frames_data,
                        expected_frame_count=system_data.get(
                            "expected_frame_count", 0
                        ),  # read from JSON or use default value
                        md_dumpfreq=system_data.get(
                            "md_dumpfreq", 1
                        ),  # backward compatibility: defaults to 1
                    )
                    self.targets.append(target)
                    targets_for_mol.append(target)

                if targets_for_mol:
                    self.mol_groups[mol_id] = targets_for_mol

            # Sort loaded targets to ensure consistency
            from ..utils.common import CommonUtils

            self.targets.sort(
                key=lambda t: CommonUtils.sort_system_names([t.system_name])[0]
            )

            # Rebuild mol_groups to maintain sorting
            self._rebuild_mol_groups()

            self.logger.debug(
                f"Loaded {len(self.targets)} analysis targets from new format"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to load analysis targets: {str(e)}")
            return False

    def load_sampled_frames_from_csv(self) -> None:
        """Load sampled frame information from single system metrics CSV, compatible with old and new directory/naming."""
        if not self.output_dir:
            self.logger.warning(
                "Output directory not set, cannot load sampled frame information"
            )
            return

        # Directory candidates
        candidates = [os.path.join(self.output_dir, DIR_SINGLE)]
        single_dir = None
        for c in candidates:
            if os.path.isdir(c):
                single_dir = c
                break
        if single_dir is None:
            self.logger.debug(
                "single_analysis directory not found, skipping sampled frame loading"
            )
            return

        loaded = 0
        for target in self.targets:
            # Filename
            csv_path = os.path.join(
                single_dir, f"{FRAME_PREFIX}{target.system_name}.csv"
            )
            if not os.path.exists(csv_path):
                continue
            try:
                sampled_frames = []
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(
                        [line for line in f if not line.startswith("#")]
                    )
                    for row in reader:
                        try:
                            if int(row.get("Selected", 0)) == 1:
                                sampled_frames.append(int(row["Frame_ID"]))
                        except Exception:
                            continue
                if sampled_frames:
                    target.sampled_frames = sampled_frames
                    loaded += 1
            except Exception as e:
                self.logger.warning(
                    f"Failed to parse sampled frames {target.system_name}: {e}"
                )
        if loaded:
            self.logger.debug(f"Sampled frame loading completed: {loaded} systems")
        else:
            self.logger.debug("No sampled frames loaded from CSV")

    def update_sampled_frames_from_results(self, analysis_results: List[tuple]) -> None:
        """Directly update sampled frame information to targets from analysis results"""
        if not analysis_results:
            return

        # Establish system_name -> sampled_frames mapping
        sampled_frames_map = {}
        for result in analysis_results:
            if len(result) >= 2:
                metrics = result[0]
                system_name = getattr(metrics, "system_name", "unknown")
                sampled_frames = getattr(metrics, "sampled_frames", [])
                sampled_frames_map[system_name] = sampled_frames

        # Synchronize sampled frames to PathManager.targets
        updated_count = 0
        for target in self.targets:
            if target.system_name in sampled_frames_map:
                new_sampled_frames = sampled_frames_map[target.system_name]
                if new_sampled_frames:  # Only update when sampled frames are not empty
                    target.sampled_frames = new_sampled_frames
                    updated_count += 1
                    self.logger.debug(
                        f"Update sampled frames: {target.system_name} ({len(target.sampled_frames)} frames)"
                    )

        self.logger.debug(
            f"Synchronize sampled frame information from analysis results: {updated_count} systems"
        )
