#!/usr/bin/env python3
"""Clone and lightly install benchmark dependencies for CARGO smoke tests.

This is a Python helper rather than another shell script because the project
intentionally keeps exactly two shell scripts: ``setup_env.sh`` and
``run_project.sh``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "external"
TAU = EXTERNAL / "tau-bench"
ACE = EXTERNAL / "ACEBench"


def run(cmd: List[str], cwd: Path | None = None, *, check: bool = False) -> Dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd or ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    out = {
        "cmd": cmd,
        "cwd": str(cwd or ROOT),
        "returncode": proc.returncode,
        "output": proc.stdout[-5000:],
    }
    if check and proc.returncode != 0:
        raise RuntimeError(json.dumps(out, indent=2))
    return out


def clone_repo(url: str, dest: Path) -> Dict[str, Any]:
    if (dest / ".git").exists():
        return {"status": "present", "path": str(dest)}
    dest.parent.mkdir(parents=True, exist_ok=True)
    return run(["git", "clone", "--depth", "1", url, str(dest)])


def install_tau() -> Dict[str, Any]:
    if not TAU.exists():
        return {"status": "blocked", "reason": f"missing {TAU}"}
    return run([sys.executable, "-m", "pip", "install", "-e", str(TAU)])


def install_ace(*, include_vllm: bool) -> Dict[str, Any]:
    req = ACE / "requirements.txt"
    if not req.exists():
        return {"status": "blocked", "reason": f"missing {req}"}
    if include_vllm:
        return run([sys.executable, "-m", "pip", "install", "-r", str(req)])
    safe = []
    skipped = []
    # ACEBench's historical requirements pin several shared packages below
    # what tau-bench and current OpenAI-compatible clients need.  For smoke
    # testing we only need import-level compatibility, so keep the environment
    # stable and skip pins that would downgrade the active benchmark runtime.
    conflict_prone = {
        "vllm",
        "openai",
        "litellm",
        "pydantic",
        "numpy",
        "pandas",
        "httpx",
        "python-dotenv",
        "typing-extensions",
        "tqdm",
    }
    for raw in req.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split("==", 1)[0].strip().lower().replace("_", "-")
        if name in conflict_prone:
            skipped.append(line)
        else:
            safe.append(line)
    if not safe:
        return {"status": "skipped", "reason": "no safe ACEBench requirements"}
    result = run([sys.executable, "-m", "pip", "install", *safe])
    result["skipped"] = skipped
    return result


def import_check() -> Dict[str, bool]:
    import importlib.util

    return {
        "tau_bench": importlib.util.find_spec("tau_bench") is not None,
        "pandas": importlib.util.find_spec("pandas") is not None,
        "openpyxl": importlib.util.find_spec("openpyxl") is not None,
        "vllm": importlib.util.find_spec("vllm") is not None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", choices=["all", "tau", "ace"], default="all")
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--include-ace-vllm", action="store_true")
    parser.add_argument("--json-out", default="")
    ns = parser.parse_args()

    summary: Dict[str, Any] = {
        "repo_root": str(ROOT),
        "python": sys.executable,
        "actions": [],
    }
    if ns.bench in ("all", "tau"):
        summary["actions"].append(clone_repo("https://github.com/sierra-research/tau-bench.git", TAU))
        if ns.install:
            summary["actions"].append(install_tau())
    if ns.bench in ("all", "ace"):
        summary["actions"].append(clone_repo("https://github.com/chenchen0103/ACEBench.git", ACE))
        if ns.install:
            summary["actions"].append(install_ace(include_vllm=ns.include_ace_vllm))
    summary["import_check"] = import_check()
    summary["git_available"] = shutil.which("git") is not None

    text = json.dumps(summary, indent=2)
    if ns.json_out:
        out = Path(ns.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
