#!/usr/bin/env python3
"""Explicit offline pipeline: raw trajectories → permanent experience memory.

This script is the ONLY sanctioned path for building permanent procedural
memory from collected trajectories. It must be run explicitly after collecting
baseline/act/react/rex trajectories. It does NOT run automatically at runtime.

Memory lifecycle
----------------
1. Offline data collection: runners produce ``outputs/<run>/trajectories.jsonl``
2. **This script**: scans trajectory files → distill → sanitize → deduplicate
   → write permanent experience cards into ``--output-bank``
3. Runtime (run_project.sh): REx reads ONLY the permanent experience bank; it
   never touches raw trajectory files.

Usage
-----
    python -m src.scripts.build_experience_memory \\
        --inputs outputs/ \\
        --output-bank permanent_memory/

    # Or for a single environment:
    python -m src.scripts.build_experience_memory \\
        --inputs outputs/tau_retail_baseline \\
        --output-bank outputs/experience_bank \\
        --domain retail

Optional flags
--------------
    --domain DOMAIN     Force all inputs into a single named domain (skips
                        path-based domain inference).
    --min-confidence F  Minimum distillation confidence (default: from config).
    --dry-run           Parse + distill but write nothing; reports what would
                        be promoted.
    --verbose           Print per-record distillation results.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..rex.config import RexConfig, default_config
from ..rex.distill import distill_record_to_card
from ..rex.memory_store import MemoryStore
from ..rex.memory_types import ProcessMemoryCard
from ..rex.sanitize import raw_sensitive_tokens


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="build_experience_memory",
        description=(
            "Offline pipeline: scan raw trajectories, distill procedural "
            "experiences, write permanent memory bank."
        ),
    )
    p.add_argument(
        "--inputs",
        required=True,
        help=(
            "Directory containing benchmark output sub-directories "
            "(each with trajectories.jsonl) OR a single trajectories.jsonl file."
        ),
    )
    p.add_argument(
        "--output-bank",
        required=True,
        help="Path to permanent experience bank directory (seed store).",
    )
    p.add_argument(
        "--domain",
        default=None,
        help=(
            "Force all inputs into this domain name.  When omitted, "
            "domain is inferred from the sub-directory name."
        ),
    )
    p.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Override minimum distillation confidence threshold (0–1).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and distill but do NOT write anything.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-record distillation outcome.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Domain inference (mirrors ingest_saved_trajectories for consistency)
# ---------------------------------------------------------------------------

def _infer_domain(path: Path) -> Optional[str]:
    """Infer domain from sub-directory name.

    Naming conventions expected from run_project.sh:
      tau_airline_baseline  → retail / airline (by env segment)
      tau_retail_rex        → retail
      ace_en                → ace
      ace_results           → ace
    """
    name = path.name
    if name.startswith("tau_"):
        parts = name.split("_")
        if len(parts) >= 2:
            return parts[1]          # "airline" or "retail"
        return "tau"
    if name.startswith("ace_"):
        return "ace"
    # Generic fallback: use the directory name as domain
    return name if name else None


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _collect_inputs(inputs_path: Path) -> List[Tuple[Path, str]]:
    """Return list of (jsonl_path, domain) pairs.

    Accepts:
      * A single ``trajectories.jsonl`` file.
      * A directory whose immediate children contain ``trajectories.jsonl``.
      * A directory that IS a run output (directly contains trajectories.jsonl).
    """
    if not inputs_path.exists():
        print(f"ERROR: inputs path does not exist: {inputs_path}", file=sys.stderr)
        return []

    # Direct file
    if inputs_path.is_file() and inputs_path.suffix == ".jsonl":
        domain = _infer_domain(inputs_path.parent) or "generic"
        return [(inputs_path, domain)]

    # Single run directory that directly holds trajectories.jsonl
    direct = inputs_path / "trajectories.jsonl"
    if direct.exists():
        domain = _infer_domain(inputs_path) or "generic"
        return [(direct, domain)]

    # Top-level outputs directory: iterate sub-directories
    results: List[Tuple[Path, str]] = []
    for subdir in sorted(inputs_path.iterdir()):
        if not subdir.is_dir():
            continue
        traj = subdir / "trajectories.jsonl"
        if not traj.exists():
            continue
        domain = _infer_domain(subdir) or "generic"
        results.append((traj, domain))

    return results


# ---------------------------------------------------------------------------
# Record loading
# ---------------------------------------------------------------------------

def _load_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        records.append(obj)
                except json.JSONDecodeError as exc:
                    print(f"  WARNING: skipped malformed JSON at line {lineno}: {exc}", file=sys.stderr)
    except OSError as exc:
        print(f"  ERROR: cannot read {path}: {exc}", file=sys.stderr)
    return records


# ---------------------------------------------------------------------------
# Forbidden-token harvest — prevents leaking task-specific IDs
# ---------------------------------------------------------------------------

def _harvest_forbidden_tokens(records: List[Dict[str, Any]]) -> List[str]:
    """Collect raw PII / ID tokens from the gold actions, user turns, etc.

    These are passed to the distiller so any cards that still reference a
    raw identifier are rejected.
    """
    tokens: set[str] = set()
    sensitive_fields = ("user_turns", "gold_actions", "task_id", "initial_user")
    for rec in records:
        for field in sensitive_fields:
            val = rec.get(field)
            if isinstance(val, str):
                tokens |= raw_sensitive_tokens(val)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, str):
                        tokens |= raw_sensitive_tokens(item)
    return [t for t in tokens if t]


# ---------------------------------------------------------------------------
# Distillation + writing
# ---------------------------------------------------------------------------

def _distill_and_collect(
    records: List[Dict[str, Any]],
    *,
    config: RexConfig,
    verbose: bool,
    forbidden_tokens: List[str],
) -> Tuple[List[ProcessMemoryCard], int, int]:
    """Distill records into ProcessMemoryCards.

    Returns (cards, n_rejected, n_skipped).
    """
    cards: List[ProcessMemoryCard] = []
    n_rejected = 0
    n_skipped = 0

    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            n_skipped += 1
            continue

        result = distill_record_to_card(
            record,
            config=config,
            forbidden_tokens=forbidden_tokens,
        )

        if result.card is None:
            reason = result.rejected_reason or "unknown"
            if reason in ("ineligible", "no_tool_calls"):
                n_skipped += 1
                if verbose:
                    print(f"    skip [{idx}]: {reason}")
            else:
                n_rejected += 1
                if verbose:
                    print(f"    reject [{idx}]: {reason}")
            continue

        cards.append(result.card)
        if verbose:
            diag = result.diagnostics or {}
            print(
                f"    ok [{idx}]: conf={diag.get('confidence', '?'):.2f}  "
                f"outcome={diag.get('outcome', '?')}  "
                f"tools={diag.get('num_tool_calls', '?')}"
            )

    return cards, n_rejected, n_skipped


def _write_to_seed_bank(
    cards: List[ProcessMemoryCard],
    *,
    domain: str,
    store: MemoryStore,
) -> int:
    """Write cards directly to the seed (permanent) bank.

    The seed bank is the canonical permanent experience store — not the runtime
    bank. Cards here survive across all future runs.
    """
    if not cards:
        return 0

    seed_path = store.path_for(store.SEED_KIND, domain)
    seed_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing signatures to deduplicate
    existing_sigs: set[str] = set()
    if seed_path.exists():
        with seed_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if isinstance(data, dict):
                        card = ProcessMemoryCard.from_dict(data)
                        existing_sigs.add(card.signature(store.config.dedup_signature_chars))
                except Exception:
                    continue

    written = 0
    with seed_path.open("a", encoding="utf-8") as f:
        for card in cards:
            sig = card.signature(store.config.dedup_signature_chars)
            if sig in existing_sigs:
                continue
            existing_sigs.add(sig)
            f.write(json.dumps(card.to_dict(), ensure_ascii=False, sort_keys=True, default=str) + "\n")
            written += 1

    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ns = _parse_args()
    inputs_path = Path(ns.inputs)
    output_bank = Path(ns.output_bank)

    # Build config — override min_confidence if requested
    cfg = default_config()
    if ns.min_confidence is not None:
        cfg = cfg.with_overrides(promotion_min_confidence=ns.min_confidence)

    # The output bank is the SEED (permanent) store.  Runtime dir is a
    # scratch directory that is never scanned by this script.
    store = MemoryStore(
        seed_dir=output_bank,
        runtime_dir=output_bank / "_runtime_scratch",
        config=cfg,
    )

    # Discover trajectory files
    file_domain_pairs = _collect_inputs(inputs_path)
    if not file_domain_pairs:
        print("No trajectories.jsonl files found.  Nothing to do.")
        return 0

    # If --domain is provided, override all inferred domains
    if ns.domain:
        file_domain_pairs = [(p, ns.domain) for p, _ in file_domain_pairs]

    print(f"build_experience_memory — offline distillation pipeline")
    print(f"  inputs      : {inputs_path}")
    print(f"  output-bank : {output_bank}")
    print(f"  dry-run     : {ns.dry_run}")
    print(f"  files found : {len(file_domain_pairs)}")
    print()

    t0 = time.monotonic()
    grand_total_written = 0
    grand_total_rejected = 0
    grand_total_skipped = 0
    grand_total_records = 0

    # Group by domain so we make one write pass per domain
    by_domain: Dict[str, List[Path]] = {}
    for path, domain in file_domain_pairs:
        by_domain.setdefault(domain, []).append(path)

    for domain in sorted(by_domain.keys()):
        paths = by_domain[domain]
        print(f"[domain={domain}]  {len(paths)} file(s)")

        # Load all records for this domain
        all_records: List[Dict[str, Any]] = []
        for p in paths:
            recs = _load_records(p)
            print(f"  loaded {len(recs):>5} records  ← {p}")
            all_records.extend(recs)

        grand_total_records += len(all_records)
        if not all_records:
            print(f"  (no records for domain '{domain}', skipping)")
            continue

        # Harvest forbidden tokens from raw records to prevent ID leakage
        forbidden = _harvest_forbidden_tokens(all_records)
        if forbidden:
            print(f"  forbidden tokens harvested: {len(forbidden)}")

        # Distill
        cards, n_rejected, n_skipped = _distill_and_collect(
            all_records,
            config=cfg,
            verbose=ns.verbose,
            forbidden_tokens=forbidden,
        )
        grand_total_rejected += n_rejected
        grand_total_skipped += n_skipped

        print(
            f"  distilled  : {len(cards)} cards  "
            f"(skipped={n_skipped}, rejected={n_rejected})"
        )

        if ns.dry_run:
            print(f"  [dry-run] would write {len(cards)} cards to {output_bank / domain}.jsonl")
            grand_total_written += len(cards)
            continue

        # Write to permanent seed bank
        written = _write_to_seed_bank(cards, domain=domain, store=store)
        grand_total_written += written
        print(f"  wrote      : {written} new cards to {output_bank}/{domain}.jsonl")
        print()

    elapsed = time.monotonic() - t0
    print("=" * 60)
    print("Build complete.")
    print(f"  total records processed : {grand_total_records}")
    print(f"  total cards written     : {grand_total_written}")
    print(f"  total skipped           : {grand_total_skipped}")
    print(f"  total rejected          : {grand_total_rejected}")
    print(f"  elapsed                 : {elapsed:.1f}s")
    if ns.dry_run:
        print("  [dry-run mode — nothing was written to disk]")
    else:
        print(f"  permanent memory bank   : {output_bank}")
        print()
        print("Next step: re-run run_project.sh — REx will retrieve from this")
        print("permanent experience bank during all future benchmark runs.")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
