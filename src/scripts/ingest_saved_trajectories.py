#!/usr/bin/env python3
"""Backfill runtime memory from existing saved trajectories.

Scans a directory tree for trajectories.jsonl files and ingests them into
the REx runtime memory bank, allowing the system to learn from previous
benchmark runs without re-running them.

Usage:
    python -m src.scripts.ingest_saved_trajectories /path/to/outputs \\
        --bank-dir /path/to/rex_experience_bank
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..rex.config import RexConfig, default_config
from ..rex.pipeline import promote_records


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("ingest_saved_trajectories")
    p.add_argument("outputs_dir", help="Path to outputs directory containing benchmark results")
    p.add_argument("--bank-dir", help="Path to REx seed bank (defaults to REX_EXPERIENCE_DIR env var)")
    p.add_argument("--runtime-dir", help="Path to runtime memory directory (defaults to REX_RUNTIME_DIR env var)")
    return p.parse_args()


def _infer_domain_from_path(path: Path) -> str | None:
    """Extract domain from path like outputs/tau_airline_rex or outputs/ace_retail_react.

    Expected naming conventions (set by run_project.sh):
      * tau_<env>_<controller>  → domain = tau_<env>
        Examples: tau_airline_baseline, tau_retail_rex
      * ace_*  → domain = ace
        Examples: ace_en, ace_results

    Returns None if the path doesn't match known patterns.
    """
    name = path.name
    if name.startswith("tau_"):
        parts = name.split("_")
        if len(parts) >= 3:
            env = parts[1]  # e.g., "airline", "retail"
            return f"tau_{env}"
        # Path starts with tau_ but lacks the _controller suffix
        return None
    if name.startswith("ace_"):
        return "ace"
    return None


def _collect_trajectory_files(outputs_dir: Path) -> Dict[str, List[Path]]:
    """Scan outputs_dir and group trajectories.jsonl by inferred domain.

    Returns dict mapping domain (e.g. "tau_airline") to list of .jsonl paths.
    """
    results: Dict[str, List[Path]] = {}
    if not outputs_dir.exists():
        return results

    for subdir in sorted(outputs_dir.iterdir()):
        if not subdir.is_dir():
            continue
        traj_path = subdir / "trajectories.jsonl"
        if not traj_path.exists():
            continue
        domain = _infer_domain_from_path(subdir)
        if domain is None:
            continue
        if domain not in results:
            results[domain] = []
        results[domain].append(traj_path)

    return results


def _load_records_from_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load all JSON objects from a JSONL file, skipping malformed lines."""
    records: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        records.append(obj)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"WARNING: failed to read {path}: {e}", file=sys.stderr)
    return records


def main() -> int:
    ns = _parse_args()
    outputs_dir = Path(ns.outputs_dir)

    # Gracefully handle missing outputs directory (common on first run).
    if not outputs_dir.exists():
        print(f"Outputs directory does not exist: {outputs_dir}")
        print("(This is normal on first run; no prior trajectories to backfill)")
        return 0

    cfg = default_config()
    if ns.bank_dir:
        cfg.bank_dir = Path(ns.bank_dir)
    if ns.runtime_dir:
        cfg.runtime_dir = Path(ns.runtime_dir)

    # Collect all trajectories grouped by domain.
    by_domain = _collect_trajectory_files(outputs_dir)
    if not by_domain:
        print(f"No trajectories found under {outputs_dir}")
        return 0

    total_promoted = 0
    total_skipped = 0
    total_rejected = 0

    for domain in sorted(by_domain.keys()):
        traj_paths = by_domain[domain]
        print(f"Ingesting {len(traj_paths)} trajectory files for domain '{domain}'")

        all_records: List[Dict[str, Any]] = []
        for path in traj_paths:
            records = _load_records_from_jsonl(path)
            all_records.extend(records)

        if not all_records:
            print(f"  No records found for domain '{domain}'")
            continue

        print(f"  Promoting {len(all_records)} records...")
        try:
            manifest = promote_records(
                all_records,
                domain=domain,
                config=cfg,
                runtime_dir=cfg.runtime_dir,
                seed_dir=cfg.bank_dir,
                allow_test_split=False,
            )
            promoted = manifest.get("promoted", 0)
            duplicates = manifest.get("duplicates_skipped", 0)
            rejected = manifest.get("rejected", [])
            blocked = manifest.get("blocked", [])

            total_promoted += promoted
            total_skipped += duplicates
            total_rejected += len(rejected) + len(blocked)

            print(f"    ✓ promoted={promoted}, duplicates={duplicates}, rejected={len(rejected)}, blocked={len(blocked)}")
        except Exception as e:
            print(f"    ✗ failed to promote records for domain '{domain}': {e}", file=sys.stderr)
            return 1

    print(f"\nBackfill summary:")
    print(f"  Total promoted: {total_promoted}")
    print(f"  Total skipped (duplicates): {total_skipped}")
    print(f"  Total rejected/blocked: {total_rejected}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
