"""Download BFCL V4 dataset files from HuggingFace to a local directory.

Usage:
    python -m src.scripts.download_bfcl --output-dir external/bfcl

Downloads the BFCL V4 evaluation files from the gorilla-llm/Berkeley-Function-Calling-Leaderboard
HuggingFace dataset. Falls back to BFCL V3 if V4 files are not yet available.

Categories downloaded by default:
  simple, multiple, parallel, parallel_multiple, multi_turn_base

These categories have sequential tool-use patterns (parallel calls, multi-turn
interactions) that the REx-RPE ProcessMemoryCard distiller captures as
tool_ordering_hints — unlike isolated single-call benchmarks.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# HuggingFace dataset sources
# ---------------------------------------------------------------------------

HF_DATASET = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
HF_RAW_BASE = "https://huggingface.co/datasets/{dataset}/resolve/main"

# Ordered list of (version_tag, filenames_to_try) for each category.
# We try V4 first; fall back to V3.
CATEGORIES: List[str] = [
    "simple",
    "multiple",
    "parallel",
    "parallel_multiple",
    "multi_turn_base",
]

VERSION_PREFIXES: List[str] = ["BFCL_v4_", "BFCL_v3_", ""]


def _make_url(dataset: str, filename: str) -> str:
    base = HF_RAW_BASE.format(dataset=dataset)
    return f"{base}/{filename}"


def _try_download(url: str, dest: Path, token: Optional[str] = None) -> bool:
    """Attempt to download ``url`` to ``dest``. Returns True on success."""
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            if resp.status != 200:
                return False
            data = resp.read()
        dest.write_bytes(data)
        return True
    except (urllib.error.HTTPError, urllib.error.URLError, OSError):
        return False


def download_category(
    cat: str,
    out_dir: Path,
    dataset: str,
    token: Optional[str] = None,
    force: bool = False,
) -> bool:
    """Download data for one category; returns True if a file was saved."""
    for prefix in VERSION_PREFIXES:
        for ext in (".json", ".jsonl"):
            fname = f"{prefix}{cat}{ext}"
            dest = out_dir / fname
            if dest.exists() and not force:
                print(f"  [skip] already exists: {dest.name}")
                return True
            url = _make_url(dataset, fname)
            print(f"  Trying {url} ...", end=" ", flush=True)
            if _try_download(url, dest, token=token):
                # Quick sanity check: must be valid JSON/JSONL with ≥1 record
                try:
                    text = dest.read_text(encoding="utf-8").strip()
                    if not text:
                        dest.unlink(missing_ok=True)
                        print("empty")
                        continue
                    obj = json.loads(text)
                    n = len(obj) if isinstance(obj, list) else 1
                    print(f"OK ({n} records)")
                    return True
                except (json.JSONDecodeError, OSError):
                    # Try JSONL
                    try:
                        lines = [l for l in text.splitlines() if l.strip()]
                        _ = json.loads(lines[0])
                        print(f"OK ({len(lines)} lines)")
                        return True
                    except Exception:
                        dest.unlink(missing_ok=True)
                        print("invalid JSON")
                        continue
            else:
                print("404/error")
    return False


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "download_bfcl",
        description="Download BFCL V4 dataset to a local directory.",
    )
    p.add_argument(
        "--output-dir", dest="output_dir",
        default=str(Path(__file__).parents[2] / "external" / "bfcl"),
        help="Destination directory (default: external/bfcl)",
    )
    p.add_argument(
        "--dataset", default=HF_DATASET,
        help=f"HuggingFace dataset repo (default: {HF_DATASET})",
    )
    p.add_argument(
        "--categories", default="",
        help=(
            f"Comma-separated list of categories to download. "
            f"Default: {','.join(CATEGORIES)}"
        ),
    )
    p.add_argument(
        "--hf-token", dest="hf_token", default="",
        help="HuggingFace access token (or set HF_TOKEN env var)",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-download even if file already exists",
    )
    return p.parse_args()


def main() -> int:
    import os
    args = _parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    token: Optional[str] = args.hf_token or os.environ.get("HF_TOKEN") or None
    cats = (
        [c.strip() for c in args.categories.split(",") if c.strip()]
        if args.categories
        else CATEGORIES
    )

    print(f"Downloading BFCL V4 dataset")
    print(f"  source:  {args.dataset}")
    print(f"  dest:    {out_dir}")
    print(f"  cats:    {cats}")
    print()

    ok_count = 0
    for cat in cats:
        print(f"Category: {cat}")
        if download_category(cat, out_dir, args.dataset, token=token, force=args.force):
            ok_count += 1
        else:
            print(f"  WARNING: could not download '{cat}' — no file found at any URL")
        print()

    print(f"Done: {ok_count}/{len(cats)} categories downloaded to {out_dir}")
    if ok_count < len(cats):
        print(
            "\nPartial download. If the HuggingFace dataset requires authentication,\n"
            "set HF_TOKEN or pass --hf-token.\n"
            "\nAlternatively, download manually from:\n"
            f"  https://huggingface.co/datasets/{args.dataset}/tree/main\n"
            "and place the files in:\n"
            f"  {out_dir}"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
