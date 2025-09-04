#!/usr/bin/env python

import glob
import json
import logging
import os
import re
import csv
import json as _json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..utils.common import CommonUtils

SCHEMA_VERSION = 2  # Current schema version of analysis_targets.json
SAMPLING_ALGORITHM_VERSION = "1.0.0"  # Sampling algorithm version

# ---------------- Directory and file naming conventions ----------------
DIR_SINGLE = "single_analysis"
DIR_SAMPLING_COMP = "sampling_comparison"
FRAME_PREFIX = "frame_"


class FileUtils:
    """File and directory operation utilities"""

    @staticmethod
    def ensure_dir(path: str) -> None:
        """Ensure directory exists, create if necessary

        Args:
            path: Directory path to ensure
        """
        CommonUtils.ensure_directory(path)

    @staticmethod
    def safe_write_csv(
        filepath: str,
        data: List[List[Any]],
        headers: Optional[List[str]] = None,
        encoding: str = "utf-8-sig",
    ) -> bool:
        """Safely write data to CSV file

        Args:
            filepath: Path to CSV file
            data: Data rows to write
            headers: Optional header row
            encoding: File encoding

        Returns:
            True if successful, False otherwise
        """
        try:
            FileUtils.ensure_dir(os.path.dirname(filepath))
            with open(filepath, "w", newline="", encoding=encoding) as f:
                writer = csv.writer(f)
                if headers:
                    writer.writerow(headers)
                writer.writerows(data)
            return True
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to write CSV file {filepath}: {e}")
            return False

    @staticmethod
    def get_project_root() -> str:
        """Get the project root directory

        Returns:
            Path to project root directory
        """
        return CommonUtils.get_project_root()


SYSTEM_DIR_PATTERN = re.compile(r"struct_mol_(\d+)_conf_(\d+)_T(\d+)K$")


@dataclass
class LightweightSystemRecord:
    system_path: str
    mol_id: str
    conf: str
    temperature: str
    stru_dir: str
    frame_count: int
    md_dumpfreq: int = 1
    sampled_frames: Optional[List[int]] = None  # May be filled by subsequent scheduler

    @property
    def system_name(self) -> str:
        return f"struct_mol_{self.mol_id}_conf_{self.conf}_T{self.temperature}K"

    @property
    def key(self) -> str:
        return f"{self.mol_id}:{self.conf}:{self.temperature}"


def lightweight_discover_systems(
    search_paths: List[str], include_project: bool = False
) -> List[LightweightSystemRecord]:
    """Only perform directory-level scanning, match all struct_*_conf_*_T*K/OUT.ABACUS/STRU directories,
    for duplicate systems, select the one with the latest creation time. The rest of the analysis is left to the main process.
    """
    dedup_map: Dict[str, LightweightSystemRecord] = {}
    stru_dirs = []
    total_dirs = 0
    n_workers = 20
    # 分层扫描：先找体系目录，再判断STRU
    for base in search_paths:
        if not base:
            continue
        base_abs = os.path.abspath(base)
        pattern = os.path.join(base_abs, "struct_mol_*_conf_*_T*K")
        for sys_dir in glob.glob(pattern):
            stru_dir = os.path.join(sys_dir, "OUT.ABACUS", "STRU")
            if os.path.isdir(stru_dir):
                stru_dirs.append(stru_dir)
    total_dirs = len(stru_dirs)

    def get_record(stru_dir):
        parent = os.path.dirname(os.path.dirname(stru_dir))
        dir_name = os.path.basename(parent)
        m = SYSTEM_DIR_PATTERN.match(dir_name)
        if not m:
            return None
        mol_id, conf, temp = m.groups()
        try:
            ctime = os.path.getctime(stru_dir)
        except Exception:
            ctime = 0

        record = LightweightSystemRecord(
            system_path=parent,
            mol_id=mol_id,
            conf=conf,
            temperature=temp,
            stru_dir=stru_dir,
            frame_count=0,
            md_dumpfreq=1,
            sampled_frames=None,
        )
        record.ctime = ctime
        return (f"{mol_id}:{conf}:{temp}", record)

    # 并行获取ctime和生成record
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(get_record, d) for d in stru_dirs]
        for fut in as_completed(futures):
            res = fut.result()
            if res is None:
                continue
            key, record = res
            prev = dedup_map.get(key)
            if prev is None or record.ctime > getattr(prev, "ctime", 0):
                dedup_map[key] = record

    records = list(dedup_map.values())
    records.sort(key=lambda r: -getattr(r, "ctime", 0))
    logger = logging.getLogger(__name__)
    logger.debug(
        f"Lightweight discovery completed: {len(records)} systems after deduplication (scanned {total_dirs} directories)"
    )
    return records


@dataclass
class AnalysisTarget:
    system_path: str
    mol_id: str
    conf: str
    temperature: str
    stru_files: List[str]
    creation_time: float
    sampled_frames: List[int] = None  # List of sampled frame numbers
    expected_frame_count: int = 0  # Expected frame count: md_nstep // md_dumpfreq
    md_dumpfreq: int = 1  # MD dump frequency for efficient frame_id to index mapping

    def __post_init__(self):
        if self.sampled_frames is None:
            self.sampled_frames = []

    @property
    def system_name(self) -> str:
        return f"struct_mol_{self.mol_id}_conf_{self.conf}_T{self.temperature}K"

    @property
    def system_key(self) -> str:
        return f"mol_{self.mol_id}_conf_{self.conf}_T{self.temperature}K"


def load_sampling_reuse_map(targets_file: str) -> Dict[str, Dict[str, object]]:
    """Read existing analysis_targets.json, build reuse mapping.

    Returns:
        dict: system_name -> { 'sampled_frames': List[int] }
    """
    reuse = {}
    if not targets_file or not os.path.exists(targets_file):
        return reuse
    try:
        with open(targets_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        molecules = data.get("molecules", {})
        for mol in molecules.values():
            for sys_name, sys_data in mol.get("systems", {}).items():
                sampled_raw = sys_data.get("sampled_frames")
                if isinstance(sampled_raw, str):
                    try:
                        sampled_frames = _json.loads(sampled_raw)
                    except Exception:
                        sampled_frames = []
                else:
                    sampled_frames = sampled_raw or []
                if isinstance(sampled_frames, list):
                    sampled_frames = sorted(
                        {
                            int(x)
                            for x in sampled_frames
                            if isinstance(x, int) and x >= 0
                        }
                    )
                # Only add to reuse mapping when sampled frames are not empty
                if sampled_frames:
                    reuse[sys_name] = {
                        "sampled_frames": sampled_frames,
                        "system_path": sys_data.get("system_path", ""),
                        "params_hash_at_sampling": sys_data.get(
                            "params_hash_at_sampling"
                        ),
                    }
    except Exception as e:
        logger = logging.getLogger("src.io.path_manager")
        logger.warning(f"Failed to read sampling reuse information: {e}")
    return reuse
