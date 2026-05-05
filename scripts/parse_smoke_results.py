#!/usr/bin/env python3
"""Summarize CARGO smoke outputs without claiming unavailable live metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        return {"status": "unreadable", "path": str(path), "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-summary", default="outputs/smoke/smoke_summary.json")
    parser.add_argument("--json-out", default="")
    ns = parser.parse_args()

    summary = load(Path(ns.smoke_summary))
    checks = summary.get("checks", {}) if isinstance(summary, dict) else {}
    compact = {
        name: {
            "status": result.get("status"),
            "returncode": result.get("returncode"),
            "reason": result.get("reason", ""),
        }
        for name, result in checks.items()
        if isinstance(result, dict)
    }
    text = json.dumps(compact, indent=2)
    if ns.json_out:
        out = Path(ns.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
