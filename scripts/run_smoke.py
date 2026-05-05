#!/usr/bin/env python3
"""Run the smallest honest CARGO smoke checks available in this environment."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: List[str], *, timeout: int = 120) -> Dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "output": proc.stdout[-5000:],
    }


def has_module(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def has_model_endpoint() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_BASE_URL"))


def synthetic_smoke() -> Dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "unittest",
        "tests.test_cargo.TestCargoV2Adapters",
        "tests.test_cargo.TestRunCargoIntegration",
        "-v",
    ]
    out = run(cmd)
    out["status"] = "ok" if out["returncode"] == 0 else "failed"
    return out


def tau_smoke() -> Dict[str, Any]:
    if not has_module("tau_bench"):
        return {"status": "blocked", "reason": "tau_bench is not importable; run python3 scripts/benchmark_setup.py --bench tau --install"}
    if not has_model_endpoint():
        return {
            "status": "blocked",
            "reason": "no OPENAI_API_KEY or OPENAI_BASE_URL; live tau-bench needs a model endpoint and user simulator provider",
            "rerun": [
                "OPENAI_BASE_URL=http://localhost:8001/v1 OPENAI_API_KEY=dummy python3 -m src.runners.tau_runner --env retail --agent cargo --model qwen-agent --user-model gpt-4o --output-dir outputs/smoke/tau_retail --end-index 1 --num-trials 1 --max-concurrency 1 --max-num-steps 5",
                "OPENAI_BASE_URL=http://localhost:8001/v1 OPENAI_API_KEY=dummy python3 -m src.runners.tau_runner --env airline --agent cargo --model qwen-agent --user-model gpt-4o --output-dir outputs/smoke/tau_airline --end-index 1 --num-trials 1 --max-concurrency 1 --max-num-steps 5",
            ],
        }
    results: Dict[str, Any] = {}
    for env in ("retail", "airline"):
        cmd = [
            sys.executable, "-m", "src.runners.tau_runner",
            "--env", env,
            "--agent", "cargo",
            "--model", os.environ.get("SMOKE_MODEL", "qwen-agent"),
            "--user-model", os.environ.get("SMOKE_USER_MODEL", "gpt-4o"),
            "--model-provider", os.environ.get("SMOKE_MODEL_PROVIDER", "openai"),
            "--user-model-provider", os.environ.get("SMOKE_USER_MODEL_PROVIDER", "openai"),
            "--output-dir", f"outputs/smoke/tau_{env}",
            "--end-index", "1",
            "--num-trials", "1",
            "--max-concurrency", "1",
            "--max-num-steps", "5",
        ]
        out = run(cmd, timeout=600)
        out["status"] = "ok" if out["returncode"] == 0 else "failed"
        results[env] = out
    return {
        "status": "ok" if all(r.get("status") == "ok" for r in results.values()) else "failed",
        "results": results,
    }


def ace_smoke() -> Dict[str, Any]:
    data_candidates = [
        ROOT / "external" / "ACEBench" / "data_all" / "data_en" / "data_agent_en.json",
        ROOT / "external" / "ACEBench" / "data_all" / "data_en" / "data_agent_multi_turn.json",
        ROOT / "external" / "ACEBench" / "data_all" / "data_en" / "data_agent_multi_step.json",
    ]
    if not any(p.exists() for p in data_candidates):
        return {
            "status": "blocked",
            "reason": "ACEBench agent data missing; expected data_agent_en, data_agent_multi_turn, or data_agent_multi_step under external/ACEBench/data_all/data_en",
        }
    if not has_model_endpoint():
        return {
            "status": "blocked",
            "reason": "no OPENAI_API_KEY or OPENAI_BASE_URL; ACEBench inference needs a model endpoint",
            "rerun": "OPENAI_BASE_URL=http://localhost:8001/v1 OPENAI_API_KEY=dummy python3 -m src.runners.ace_runner --agent cargo --model qwen-agent --limit 1 --max-concurrency 1 --max-num-steps 5 --output-dir outputs/smoke/ace_agent",
        }
    cmd = [
        sys.executable, "-m", "src.runners.ace_runner",
        "--agent", "cargo",
        "--model", os.environ.get("SMOKE_MODEL", "qwen-agent"),
        "--limit", "1",
        "--max-concurrency", "1",
        "--max-num-steps", "5",
        "--output-dir", "outputs/smoke/ace_agent",
    ]
    out = run(cmd, timeout=600)
    out["status"] = "ok" if out["returncode"] == 0 else "failed"
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["all", "synthetic", "tau", "ace"], default="all")
    parser.add_argument("--json-out", default="outputs/smoke/smoke_summary.json")
    ns = parser.parse_args()

    summary: Dict[str, Any] = {
        "python": sys.executable,
        "repo_root": str(ROOT),
        "checks": {},
    }
    if ns.target in ("all", "synthetic"):
        summary["checks"]["synthetic"] = synthetic_smoke()
    if ns.target in ("all", "tau"):
        summary["checks"]["tau"] = tau_smoke()
    if ns.target in ("all", "ace"):
        summary["checks"]["ace"] = ace_smoke()

    out = Path(ns.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if all(c.get("status") in {"ok", "blocked"} for c in summary["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
