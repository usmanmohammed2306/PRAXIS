"""ACEBench Agent runner — vanilla / Act / ReAct / REx.

ACEBench's Agent split is scored offline against the saved trajectory.
Tools cannot be executed live in this driver, so per-step tool results are
stubbed; what we measure here is the *sequence of tool calls* the agent
chooses to issue.

Outputs:

  * ``trajectories.jsonl`` — one JSON record per task (full conversation,
    expected vs actual tool names, controller-specific diagnostics).
  * ``metrics.json`` — completion rate, name-coverage (where ground-truth is
    available), avg tool calls and steps. These are diagnostic; the
    canonical ACEBench score should be re-computed by upstream
    ``score_agent.py`` against the saved trajectories.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..baselines.ace_loops import run_baseline_style
from ..common.io_utils import append_jsonl, ensure_dir, safe_mean, write_json
from ..common.openai_client import get_client
from ..rex.config import RexConfig
from ..rex.experience import promote_trajectories
from ..rex.memory_quality import consolidate
from ..rex.memory_store import MemoryStore
from ..rex.pipeline import promote_records as pipeline_promote_records
from ..rex.retrieval_logging import aggregate_logs


REPO_ROOT = Path(__file__).resolve().parents[2]
ACE_REPO = REPO_ROOT / "external" / "ACEBench"
AGENT_CHOICES = ["baseline", "act", "react", "rex"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("ace_runner")
    p.add_argument("--agent", required=True, choices=AGENT_CHOICES)
    p.add_argument("--model", required=True)
    p.add_argument("--language", default="en")
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--max-num-steps", type=int, default=20)
    p.add_argument("--max-concurrency", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--promote-runtime-memory",
        choices=["auto", "always", "never"],
        default="auto",
        help=(
            "After the run completes, distill emitted trajectories into the REx "
            "runtime memory bank under $REX_RUNTIME_DIR (auto = promote)."
        ),
    )
    p.add_argument(
        "--runtime-dir",
        default="",
        help="Override REX_RUNTIME_DIR for this run; empty falls back to the env var.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------
def _candidate_task_paths(language: str) -> List[Path]:
    lang = language.lower()
    names = [
        f"data_all/data_{lang}/data_agent_{lang}.json",
        f"data_all/data_{lang}/data_agent_multi_turn.json",
        f"data_all/data_{lang}/data_agent_multi_step.json",
        f"data_all/data_agent_{lang}.json",
        f"data/data_{lang}/data_agent_{lang}.json",
        f"data/data_{lang}/data_agent_multi_turn.json",
        f"data/data_{lang}/data_agent_multi_step.json",
        f"data/data_agent_{lang}.json",
        f"data_agent_{lang}.json",
    ]
    candidates = [ACE_REPO / n for n in names]
    if ACE_REPO.exists():
        for p in ACE_REPO.rglob(f"data_agent_{lang}.json"):
            candidates.append(p)
        for p in ACE_REPO.rglob("data_agent_multi_turn.json"):
            candidates.append(p)
        for p in ACE_REPO.rglob("data_agent_multi_step.json"):
            candidates.append(p)
    seen: List[Path] = []
    for p in candidates:
        if p not in seen:
            seen.append(p)
    return seen


def _load_tasks(language: str, limit: int) -> Tuple[List[Dict[str, Any]], Optional[Path]]:
    for path in _candidate_task_paths(language):
        if not path.exists():
            continue
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if not raw:
            continue
        tasks: List[Dict[str, Any]] = []
        if raw[0] == "[":
            try:
                data = json.loads(raw)
                if isinstance(data, list):
                    tasks = [t for t in data if isinstance(t, dict)]
            except Exception:
                tasks = []
        else:
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        tasks.append(obj)
                except Exception:
                    continue
        if tasks:
            return tasks[:limit], path
    return [], None


# ---------------------------------------------------------------------------
# Task field normalization
# ---------------------------------------------------------------------------
def _extract_user_turn(task: Dict[str, Any]) -> str:
    for key in ("question", "query", "instruction", "user", "prompt"):
        v = task.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    msgs = task.get("messages") or task.get("conversation")
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "user" and m.get("content"):
                return str(m["content"])
    return ""


def _normalize_param_schema(params: Any) -> Dict[str, Any]:
    """Convert ACEBench parameter schema to OpenAI-compatible format.

    ACEBench uses "type": "dict" for object types; OpenAI API requires
    "type": "object". Converts recursively.
    """
    if not isinstance(params, dict):
        return {"type": "object", "properties": {}}
    out = {}
    for k, v in params.items():
        if k == "type" and v == "dict":
            out[k] = "object"
        elif isinstance(v, dict):
            out[k] = _normalize_param_schema(v)
        elif isinstance(v, list):
            out[k] = [
                _normalize_param_schema(i) if isinstance(i, dict) else i
                for i in v
            ]
        else:
            out[k] = v
    # OpenAI requires "properties" key on object types
    if out.get("type") == "object" and "properties" not in out:
        out["properties"] = {}
    return out


def _extract_tool_specs(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    # ACEBench stores tools in "function" (singular list).
    # Other benchmarks use "tools", "functions", or "available_tools".
    raw = (
        task.get("function")          # ACEBench agent format
        or task.get("tools")
        or task.get("functions")
        or task.get("available_tools")
        or []
    )
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        # Already in OpenAI wrapped format: {"type": "function", "function": {...}}
        if item.get("type") == "function" and isinstance(item.get("function"), dict):
            fn = item["function"]
            params = _normalize_param_schema(
                fn.get("parameters") or fn.get("params") or {}
            )
            out.append({
                "type": "function",
                "function": {
                    "name": str(fn.get("name", "")),
                    "description": str(fn.get("description", "")),
                    "parameters": params,
                },
            })
            continue
        # Flat format: {"name": ..., "description": ..., "parameters": ...}
        fn = item.get("function") if isinstance(item.get("function"), dict) else item
        name = fn.get("name")
        if not name:
            continue
        params = _normalize_param_schema(
            fn.get("parameters") or fn.get("params") or {}
        )
        spec = {
            "type": "function",
            "function": {
                "name": str(name),
                "description": str(fn.get("description", "")),
                "parameters": params,
            },
        }
        out.append(spec)
    return out


def _extract_ground_truth_tools(task: Dict[str, Any]) -> List[str]:
    # "path" is the ACEBench field for expected tool call sequence.
    # It may be a list of dicts like [{"name": "func", ...}, ...] or a list of strings.
    path = task.get("path")
    if isinstance(path, list) and path:
        names = []
        for item in path:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, dict):
                n = item.get("name") or item.get("tool_name") or item.get("function")
                if isinstance(n, str) and n:
                    names.append(n)
        if names:
            return names

    # Fallback: check other common ground-truth field names
    for key in ("ground_truth", "gold", "expected", "answer", "target"):
        gt = task.get(key)
        if gt is None:
            continue
        names = _walk_for_tool_names(gt)
        if names:
            return names
    return []


def _walk_for_tool_names(obj: Any) -> List[str]:
    out: List[str] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            name = x.get("name") or x.get("tool_name") or x.get("function")
            if isinstance(name, str) and name and "args" not in name and len(name) < 80:
                parent_looks_toollike = any(k in x for k in ("arguments", "args", "parameters"))
                if parent_looks_toollike:
                    out.append(name)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(obj)
    return out


def _system_prompt_for_task(task: Dict[str, Any]) -> str:
    sys_msg = task.get("system") or task.get("system_prompt")
    if isinstance(sys_msg, str) and sys_msg.strip():
        return sys_msg.strip()
    return ""


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _coverage(expected: List[str], actual: List[str]) -> float:
    if not expected:
        return 0.0
    expected_set = set(expected)
    actual_set = set(actual)
    hit = sum(1 for n in expected_set if n in actual_set)
    return hit / len(expected_set)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _make_run_fn(agent_kind: str):
    if agent_kind in ("baseline", "act", "react"):
        def run_fn(*, client, model, task, max_num_steps, temperature):
            return run_baseline_style(
                style=agent_kind,
                client=client,
                model=model,
                task=task,
                tool_specs=_extract_tool_specs(task),
                user_turn=_extract_user_turn(task),
                max_num_steps=max_num_steps,
                temperature=temperature,
            )
        return run_fn

    if agent_kind == "rex":
        from ..rex.ace_loop import run_rex

        def run_fn(*, client, model, task, max_num_steps, temperature):
            return run_rex(
                client=client,
                model=model,
                task=task,
                tool_specs=_extract_tool_specs(task),
                user_turn=_extract_user_turn(task),
                max_num_steps=max_num_steps,
                temperature=temperature,
            )
        return run_fn

    raise ValueError(f"Unknown agent kind: {agent_kind}")


def main() -> int:
    ns = _parse_args()
    out_dir = ensure_dir(ns.output_dir)
    traj_path = Path(out_dir) / "trajectories.jsonl"
    if traj_path.exists():
        traj_path.unlink()

    tasks, source = _load_tasks(ns.language, ns.limit)
    summary: Dict[str, Any] = {
        "benchmark": "ACEBench",
        "category": "agent",
        "agent": ns.agent,
        "model": ns.model,
        "language": ns.language,
        "source_file": str(source) if source else None,
        "config": {
            "limit": ns.limit,
            "max_num_steps": ns.max_num_steps,
            "temperature": ns.temperature,
        },
    }

    if not tasks:
        checked = [str(p) for p in _candidate_task_paths(ns.language)]
        summary["status"] = "skipped"
        summary["note"] = (
            "No ACEBench Agent task file could be located. "
            "Expected one of the paths below to exist under the cloned ACEBench repo. "
            "Re-run setup_env.sh (to clone ACEBench) or set the correct path manually."
        )
        summary["searched_paths"] = checked
        summary["metrics"] = {"num_tasks": 0}
        write_json(os.path.join(ns.output_dir, "metrics.json"), summary)
        print(json.dumps(summary, indent=2))
        return 0

    client = get_client()
    run_fn = _make_run_fn(ns.agent)

    def _solve_task(i: int, task: Dict[str, Any]) -> Dict[str, Any]:
        try:
            res = run_fn(
                client=client,
                model=ns.model,
                task=task,
                max_num_steps=ns.max_num_steps,
                temperature=ns.temperature,
            )
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            res = {
                "status": "error",
                "error": f"{exc.__class__.__name__}: {exc}",
                "messages": [],
                "tool_calls_made": [],
            }
        expected = _extract_ground_truth_tools(task)
        actual = res.get("tool_calls_made", [])
        coverage = _coverage(expected, actual)
        return {
            "index": i,
            "task_id": task.get("id") or task.get("task_id") or i,
            "controller": ns.agent,
            "status": res.get("status"),
            "error": res.get("error"),
            "expected_tools": expected,
            "actual_tools": actual,
            "tool_coverage": coverage,
            "num_steps": sum(1 for m in res.get("messages", []) if m.get("role") == "assistant"),
            "messages": res.get("messages", []),
            "rex_stats": res.get("rex_stats"),
        }

    write_lock = threading.Lock()
    records: List[Dict[str, Any]] = []
    max_workers = max(1, int(ns.max_concurrency))

    if max_workers == 1:
        for i, task in enumerate(tasks):
            record = _solve_task(i, task)
            append_jsonl(traj_path, record)
            records.append(record)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_solve_task, i, t): i for i, t in enumerate(tasks)}
            for fut in as_completed(futures):
                record = fut.result()
                with write_lock:
                    append_jsonl(traj_path, record)
                    records.append(record)
    records.sort(key=lambda r: r.get("index", 0))

    completed = [r for r in records if r.get("status") == "ok"]
    coverages = [r["tool_coverage"] for r in records if r.get("expected_tools")]
    metrics = {
        "num_tasks": len(records),
        "completion_rate": safe_mean([1.0 if r.get("status") == "ok" else 0.0 for r in records]),
        "avg_tool_calls": safe_mean([float(len(r.get("actual_tools", []))) for r in records]),
        "avg_steps": safe_mean([float(r.get("num_steps", 0)) for r in records]),
        "tool_name_coverage": safe_mean(coverages) if coverages else None,
        "tasks_with_ground_truth": len(coverages),
        "error_tasks": len(records) - len(completed),
    }
    summary["status"] = "ok"
    summary["metrics"] = metrics

    # Promote trajectories into the REx runtime memory bank under the "ace"
    # domain. ACE has no live reward, so we synthesize one from tool coverage
    # (when ground-truth is available) or from "tools were called at all" so
    # genuinely no-op runs are excluded.
    promotion = _promote_ace_records(ns, records)
    summary["runtime_memory_promotion"] = promotion

    # Aggregate retrieval logs (best-effort).
    try:
        cfg = RexConfig.from_env()
        summary["retrieval_logs"] = aggregate_logs(cfg.retrieval_log_dir)
    except Exception as exc:  # noqa: BLE001
        summary["retrieval_logs"] = {"error": f"{exc.__class__.__name__}: {exc}"}

    write_json(os.path.join(ns.output_dir, "metrics.json"), summary)
    print(json.dumps(metrics, indent=2))
    return 0


def _promote_ace_records(
    ns: argparse.Namespace,
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Distill ACE trajectories into the runtime memory bank for the ace domain."""
    mode = (ns.promote_runtime_memory or "auto").lower()
    if mode == "never":
        print("[memory] ACE promotion skipped: promote_runtime_memory=never", file=sys.stderr)
        return {"status": "skipped", "reason": "promote_runtime_memory=never"}
    runtime_dir = Path(ns.runtime_dir) if ns.runtime_dir else None
    print(
        f"[memory] promoting {len(records)} ACE trajectories "
        f"(promote_mode={mode})",
        file=sys.stderr,
    )

    enriched: List[Dict[str, Any]] = []
    for r in records:
        if not isinstance(r, dict):
            continue
        coverage = r.get("tool_coverage")
        actual_tools = r.get("actual_tools") or []
        # Synthesize a reward signal for the distiller.
        # tool_coverage is 0.0 for all agent tasks whose expected path is empty
        # (the benchmark doesn't provide a gold call sequence). In that case we
        # fall back to "were tools called at all?" as a proxy for a useful run.
        if isinstance(coverage, (int, float)) and float(coverage) > 0.0:
            reward = float(coverage)
        elif actual_tools:
            # Tools were called even though we have no ground-truth to score
            # against — treat this as a partial success so the distiller keeps
            # the trajectory as a useful procedural example.
            reward = 0.5
        else:
            reward = 0.0
        enriched.append({
            "task_id": r.get("task_id") or r.get("index"),
            "trial": 0,
            "reward": reward,
            # Include domain/environment so trajectory_parser._environment_for()
            # correctly labels these cards as "ace" rather than "generic".
            "info": {
                "controller": r.get("controller", ns.agent),
                "domain": "ace",
                "environment": "ace",
            },
            "messages": r.get("messages") or [],
            "status": r.get("status", "ok"),
            "error": r.get("error") or "",
            # Preserve ACE-specific fields so parse_record can use them.
            "tool_coverage": coverage if isinstance(coverage, (int, float)) else 0.0,
            "expected_tools": r.get("expected_tools") or [],
            "actual_tools": actual_tools,
        })
    try:
        manifest = promote_trajectories(
            enriched,
            domain="ace",
            runtime_dir=runtime_dir,
            allow_test_split=mode == "always",
        )
        manifest["status"] = "ok"
        promoted = manifest.get("promoted", 0)
        total_after = manifest.get("total_runtime_cards_after", 0)
        print(
            f"[memory] v1 promoted: {promoted} cards → "
            f"total runtime={total_after}",
            file=sys.stderr,
        )
        # v2 pipeline (ProcessMemoryCard) — non-fatal on failure.
        try:
            v2_runtime = (runtime_dir / "v2") if runtime_dir else None
            v2_manifest = pipeline_promote_records(
                enriched,
                domain="ace",
                runtime_dir=v2_runtime,
                allow_test_split=mode == "always",
            )
            manifest["v2"] = v2_manifest
            v2_promoted = v2_manifest.get("promoted", 0)
            v2_total = v2_manifest.get("total_runtime_cards_after", 0)
            print(
                f"[memory] v2 promoted: {v2_promoted} cards → "
                f"total runtime={v2_total}",
                file=sys.stderr,
            )
        except Exception as exc:  # noqa: BLE001
            manifest["v2_error"] = f"{exc.__class__.__name__}: {exc}"
            print(
                f"[memory] v2 promotion failed: {exc.__class__.__name__}",
                file=sys.stderr,
            )
        # Memory consolidation pass over the v2 ACE store.
        try:
            cfg = RexConfig.from_env()
            v2_runtime_dir = (runtime_dir / "v2") if runtime_dir else cfg.runtime_dir
            store = MemoryStore(seed_dir=cfg.bank_dir, runtime_dir=v2_runtime_dir, config=cfg)
            cards = store.load_runtime("ace")
            if cards:
                report = consolidate(cards, config=cfg)
                manifest["consolidation"] = report.to_dict()
                print(
                    f"[memory] consolidation: {len(cards)} cards, "
                    f"duplicates={report.num_duplicates}, "
                    f"decay_applied={report.cards_with_decay}",
                    file=sys.stderr,
                )
        except Exception as exc:  # noqa: BLE001
            manifest["consolidation_error"] = f"{exc.__class__.__name__}: {exc}"
            print(
                f"[memory] consolidation failed: {exc.__class__.__name__}",
                file=sys.stderr,
            )
        print(f"[memory] promotion complete: status={manifest['status']}", file=sys.stderr)
        return manifest
    except Exception as exc:  # noqa: BLE001
        print(
            f"[memory] promotion error: {exc.__class__.__name__}: {exc}",
            file=sys.stderr,
        )
        return {"status": "error", "error": f"{exc.__class__.__name__}: {exc}"}


if __name__ == "__main__":
    sys.exit(main())
