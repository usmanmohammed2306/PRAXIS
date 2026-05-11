"""Unified tau-bench runner for the four-way comparison.

All four controllers share a single in-process loop. The only thing that
varies between them is the agent class instantiated per task:

  * ``baseline`` — :class:`baselines.ToolCallingAgent` (vanilla tool-calling
    with a minimal system prompt)
  * ``act``      — :class:`baselines.ActAgent` (no reasoning prose, action-only)
  * ``react``    — :class:`baselines.ReActAgent` (one-line Thought before each
    Action)
  * ``rex``      — :class:`rex.agent.RexAgent` (leakage-safe experience
    retrieval + native function calling + write-only SABER-style reflection).

Sharing the loop guarantees the gap between conditions is due to the
controller and not divergent control flow / tool-result formatting.

The litellm-truncation sitecustomize patch is loaded in-process for every
condition because tau-bench's user simulator calls ``litellm.completion``
under the hood and can otherwise blow the model's context window on long
multi-turn dialogues.
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
from typing import Any, Dict, List, Tuple

from ..common.io_utils import append_jsonl, ensure_dir, safe_mean, write_json
from ..rex.config import RexConfig
from ..rex.memory_quality import consolidate
from ..rex.memory_store import MemoryStore
from ..rex.pipeline import promote_records as pipeline_promote_records
from ..rex.retrieval_logging import aggregate_logs


AGENT_CHOICES = [
    "baseline", "act", "react",        # evaluation baselines
    "cot", "plan-solve", "reflexion-lite",  # donor-only controllers (Phase B)
    "rex", "praxis",
]


def _try_install_litellm_patch() -> None:
    """Best-effort: load the litellm-truncation sitecustomize hook in-process."""
    try:
        repo_root = Path(__file__).resolve().parents[2]
        patch_dir = repo_root / "src" / "_taubench_patches"
        if str(patch_dir) not in sys.path:
            sys.path.insert(0, str(patch_dir))
        import sitecustomize  # noqa: F401  (executes _install on import)
    except Exception as exc:  # noqa: BLE001
        print(f"[tau_runner] litellm patch not loaded: {exc}", flush=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("tau_runner")
    p.add_argument("--env", required=True, choices=["retail", "airline"])
    p.add_argument("--agent", required=True, choices=AGENT_CHOICES)
    p.add_argument("--model", required=True)
    p.add_argument("--user-model", required=True)
    p.add_argument("--model-provider", default="openai")
    p.add_argument("--user-model-provider", default="openai")
    p.add_argument("--user-strategy", default="llm")
    p.add_argument("--task-split", default="test")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--end-index", type=int, default=20)
    p.add_argument("--num-trials", type=int, default=1)
    p.add_argument("--max-concurrency", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-num-steps", type=int, default=30)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--promote-runtime-memory",
        choices=["auto", "always", "never"],
        default="auto",
        help=(
            "After the run completes, distill successful trajectories into the REx "
            "runtime memory bank under $REX_RUNTIME_DIR (auto = promote unless task "
            "split is 'test')."
        ),
    )
    p.add_argument(
        "--runtime-dir",
        default="",
        help="Override REX_RUNTIME_DIR for this run; empty falls back to the env var.",
    )
    p.add_argument(
        "--internal-train-end",
        type=int,
        default=0,
        help=(
            "Index boundary for an internal train/test split (used for airline, "
            "which ships only a 'test' split upstream). Task indices in [0, "
            "internal_train_end) are tagged info.task_split='train' (eligible "
            "for promotion); indices >= internal_train_end are tagged 'test' "
            "and blocked from promotion by the pipeline. 0 disables the override."
        ),
    )
    return p.parse_args()


def _resolve_agent_cls(kind: str):
    if kind == "baseline":
        from ..baselines.agents import ToolCallingAgent
        return ToolCallingAgent
    if kind == "act":
        from ..baselines.agents import ActAgent
        return ActAgent
    if kind == "react":
        from ..baselines.agents import ReActAgent
        return ReActAgent
    if kind == "cot":
        from ..baselines.agents import CoTAgent
        return CoTAgent
    if kind == "plan-solve":
        from ..baselines.agents import PlanSolveAgent
        return PlanSolveAgent
    if kind == "reflexion-lite":
        from ..baselines.agents import ReflexionLiteAgent
        return ReflexionLiteAgent
    if kind in ("rex", "praxis"):
        from ..rex.agent import RexAgent
        return RexAgent
    raise ValueError(f"Unknown agent kind: {kind}")


def _collect_records(output_dir: Path) -> List[Dict[str, Any]]:
    """Load every per-trial record under ``output_dir`` into a flat list."""
    records: List[Dict[str, Any]] = []
    for path in sorted(output_dir.rglob("results-*.json")) + sorted(output_dir.rglob("results.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(data, list):
            records.extend(x for x in data if isinstance(x, dict))
        elif isinstance(data, dict) and "results" in data:
            items = data.get("results")
            if isinstance(items, list):
                records.extend(x for x in items if isinstance(x, dict))
    seen = set()
    dedup: List[Dict[str, Any]] = []
    for r in records:
        key = (r.get("trial"), r.get("task_id"))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)
    return dedup


def _compute_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    rewards = [float(r.get("reward", 0.0) or 0.0) for r in records]
    successes = [
        1.0 if (r.get("reward") is not None and float(r.get("reward") or 0.0) >= 1.0) else 0.0
        for r in records
    ]
    step_counts: List[float] = []
    for r in records:
        msgs = r.get("messages") or r.get("traj") or []
        if isinstance(msgs, list):
            step_counts.append(float(len(msgs)))
    info_errors = sum(1 for r in records if isinstance(r.get("info"), dict) and r["info"].get("error"))

    # Aggregate REx diagnostics if present.
    rex_n = 0
    rex_reflections = 0
    rex_blocks = 0
    rex_tools = 0
    rex_mutations = 0
    rex_cards_loaded = 0
    rex_cards_used = 0
    for r in records:
        info = r.get("info") or {}
        rs = info.get("rex_stats") if isinstance(info, dict) else None
        if isinstance(rs, dict):
            rex_n += 1
            rex_reflections += int(rs.get("reflection_calls") or 0)
            rex_blocks += int(rs.get("reflection_blocks") or 0)
            rex_tools += int(rs.get("tool_calls_executed") or 0)
            rex_mutations += int(rs.get("mutating_tool_calls") or 0)
            rex_cards_loaded += int(rs.get("cards_loaded") or 0)
            rex_cards_used += len(rs.get("cards_used") or [])
    out: Dict[str, Any] = {
        "num_tasks": len(records),
        "success_rate": safe_mean(successes),
        "avg_reward": safe_mean(rewards),
        "avg_trajectory_messages": safe_mean(step_counts),
        "error_tasks": info_errors,
    }
    if rex_n:
        # Aggregate playbook & retrieval-backend telemetry from rex_stats.
        playbook_chars: List[float] = []
        backends: Dict[str, int] = {}
        retrieval_refreshes_sum = 0
        for r in records:
            info = r.get("info") or {}
            rs = info.get("rex_stats") if isinstance(info, dict) else None
            if not isinstance(rs, dict):
                continue
            pc = rs.get("playbook_chars")
            if isinstance(pc, (int, float)):
                playbook_chars.append(float(pc))
            backend = rs.get("retrieval_backend") or ""
            if backend:
                backends[backend] = backends.get(backend, 0) + 1
            retrieval_refreshes_sum += int(rs.get("retrieval_refreshes") or 0)
        out["rex_diagnostics"] = {
            "trajectories_with_stats": rex_n,
            "reflection_calls": rex_reflections,
            "reflection_blocks": rex_blocks,
            "tool_calls_executed": rex_tools,
            "mutating_tool_calls": rex_mutations,
            "avg_cards_loaded": rex_cards_loaded / max(1, rex_n),
            "avg_cards_used": rex_cards_used / max(1, rex_n),
            "avg_playbook_chars": (sum(playbook_chars) / len(playbook_chars)) if playbook_chars else 0.0,
            "retrieval_backends": backends,
            "total_retrieval_refreshes": retrieval_refreshes_sum,
        }
    return out


def _should_promote(ns: argparse.Namespace) -> bool:
    """Decide whether to distill this run's trajectories into runtime memory.

    Promotion writes to a memory bank that future runs read, so we never
    promote test-split runs by default — that would cause label leakage.
    The ``always`` option exists for ablation studies that explicitly want
    it; ``never`` opts out entirely.
    """
    mode = (ns.promote_runtime_memory or "auto").lower()
    if mode == "never":
        return False
    if mode == "always":
        return True
    # auto: promote unless this is a pure held-out test-split run.
    # When --internal-train-end is set, we have a mixed run (some train, some
    # test indices) and rely on the pipeline's per-record _is_test_split_record
    # gate to drop the test ones — so we DO want to enter the promotion path.
    if int(getattr(ns, "internal_train_end", 0) or 0) > 0:
        return True
    return (ns.task_split or "").strip().lower() != "test"


def _promote_records_to_runtime(
    ns: argparse.Namespace,
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Distill ``records`` into the runtime memory bank for ``ns.env``.

    Writes ``ProcessMemoryCard``-schema JSONL (v2) to ``${REX_RUNTIME_DIR}``
    via :func:`pipeline_promote_records`. The legacy v1 ``promote_trajectories``
    path has been removed: it wrote a different schema to the same file,
    producing a mixed-schema JSONL that caused ``RexAgent`` to silently drop
    all v2 cards when building the hybrid retriever corpus.
    """
    if not _should_promote(ns):
        reason = f"task_split={ns.task_split}"
        print(f"[memory] promotion skipped: {reason}", file=sys.stderr)
        return {"status": "skipped", "reason": reason}
    runtime_dir = Path(ns.runtime_dir) if ns.runtime_dir else None
    print(
        f"[memory] promoting {len(records)} trajectories from {ns.env} "
        f"(promote_mode={ns.promote_runtime_memory or 'auto'})",
        file=sys.stderr,
    )
    try:
        # v2 pipeline only — writes ProcessMemoryCard-schema JSONL to REX_RUNTIME_DIR.
        # The legacy promote_trajectories (v1 ExperienceCard) is no longer called here
        # because it wrote a different schema to the same file, producing a mixed-schema
        # JSONL that confused MemoryStore readers and caused RexAgent to silently drop
        # all v2 cards when building the hybrid retriever.
        manifest = pipeline_promote_records(
            records,
            domain=ns.env,
            runtime_dir=runtime_dir,
            allow_test_split=ns.promote_runtime_memory == "always",
        )
        manifest["status"] = "ok"
        promoted = manifest.get("promoted", 0)
        total_after = manifest.get("total_runtime_cards_after", 0)
        print(
            f"[memory] promoted: {promoted} cards → "
            f"total runtime={total_after}",
            file=sys.stderr,
        )
        # Memory consolidation pass — runs detect_duplicates, contradictions,
        # decay over the runtime store (same dir that pipeline_promote_records writes to).
        try:
            cfg = RexConfig.from_env()
            store_runtime_dir = runtime_dir if runtime_dir else cfg.runtime_dir
            store = MemoryStore(seed_dir=cfg.bank_dir, runtime_dir=store_runtime_dir, config=cfg)
            cards = store.load_runtime(ns.env)
            if cards:
                report = consolidate(cards, config=cfg)
                manifest["consolidation"] = report.to_dict()
                print(
                    f"[memory] consolidation: {len(cards)} cards, "
                    f"duplicates={len(report.duplicates)}, "
                    f"decay_applied={report.decayed}",
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


def _make_env(ns: argparse.Namespace, task_index: int):
    """Construct a fresh tau-bench env for ``task_index``."""
    from tau_bench.envs import get_env  # type: ignore

    kwargs_common = dict(
        env_name=ns.env,
        user_strategy=ns.user_strategy,
        user_model=ns.user_model,
        task_split=ns.task_split,
        task_index=task_index,
    )
    try:
        return get_env(user_provider=ns.user_model_provider, **kwargs_common)
    except TypeError:
        return get_env(user_model_provider=ns.user_model_provider, **kwargs_common)


def _solve_one(
    ns: argparse.Namespace,
    AgentCls,
    task_index: int,
    trial: int,
) -> Dict[str, Any]:
    """Run a single (task_index, trial) and return the record dict."""
    env = _make_env(ns, task_index)
    agent_kwargs: Dict[str, Any] = dict(
        tools_info=getattr(env, "tools_info", []) or [],
        wiki=getattr(env, "wiki", "") or "",
        model=ns.model,
        provider=ns.model_provider,
        temperature=float(ns.temperature),
    )
    if ns.agent == "rex":
        agent_kwargs["env_hint"] = ns.env
    agent = AgentCls(**agent_kwargs)
    try:
        result = agent.solve(env, task_index=task_index, max_num_steps=ns.max_num_steps)
        reward = float(getattr(result, "reward", 0.0) or 0.0)
        info = getattr(result, "info", {}) or {}
        messages = getattr(result, "messages", []) or []
        total_cost = float(getattr(result, "total_cost", 0.0) or 0.0)
        status = "ok"
        err = ""
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        reward, info, messages, total_cost = 0.0, {"error": str(exc)}, [], 0.0
        status = "error"
        err = f"{exc.__class__.__name__}: {exc}"
    # Stamp domain/environment into info so trajectory_parser._environment_for()
    # labels cards as "retail"/"airline" rather than "generic", which activates
    # the correct domain-specific distillation heuristics.
    if isinstance(info, dict):
        info.setdefault("domain", ns.env)
        info.setdefault("environment", ns.env)
    else:
        info = {"domain": ns.env, "environment": ns.env}
    # Stamp task_split into info so the pipeline's _is_test_split_record() gate
    # can block test-split records from being distilled into the memory bank.
    # This is defence-in-depth on top of _should_promote() at the run level.
    #   • If --internal-train-end > 0 (airline-style internal split), label by
    #     task_index: indices in [0, N) are "train", rest are "test".
    #   • Otherwise, fall back to ns.task_split (e.g. retail's upstream split).
    internal_end = int(getattr(ns, "internal_train_end", 0) or 0)
    if internal_end > 0:
        info["task_split"] = "train" if task_index < internal_end else "test"
    else:
        info.setdefault("task_split", ns.task_split)
    return {
        "task_id": task_index,
        "trial": trial,
        "reward": reward,
        "info": info,
        "messages": messages,
        "total_cost": total_cost,
        "status": status,
        "error": err,
    }


def _run_inprocess(ns: argparse.Namespace) -> None:
    _try_install_litellm_patch()
    AgentCls = _resolve_agent_cls(ns.agent)

    out_dir = Path(ns.output_dir)
    ensure_dir(out_dir)
    traj_path = out_dir / f"results-{ns.agent}.json"
    jsonl_path = out_dir / "trajectories.jsonl"
    if traj_path.exists():
        traj_path.unlink()
    if jsonl_path.exists():
        jsonl_path.unlink()

    work: List[Tuple[int, int]] = [
        (ti, tr)
        for ti in range(ns.start_index, ns.end_index)
        for tr in range(ns.num_trials)
    ]

    write_lock = threading.Lock()
    records: List[Dict[str, Any]] = []
    max_workers = max(1, int(ns.max_concurrency))

    if max_workers == 1:
        for (ti, tr) in work:
            rec = _solve_one(ns, AgentCls, ti, tr)
            append_jsonl(jsonl_path, rec)
            records.append(rec)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_solve_one, ns, AgentCls, ti, tr): (ti, tr) for (ti, tr) in work}
            for fut in as_completed(futures):
                rec = fut.result()
                with write_lock:
                    append_jsonl(jsonl_path, rec)
                    records.append(rec)

    records.sort(key=lambda r: (r.get("task_id", 0), r.get("trial", 0)))
    write_json(traj_path, records)


def main() -> int:
    ns = _parse_args()
    ensure_dir(ns.output_dir)

    status = "ok"
    error: str = ""
    try:
        _run_inprocess(ns)
    except Exception as exc:  # noqa: BLE001 — record any failure
        status = "error"
        error = f"{exc.__class__.__name__}: {exc}"
        traceback.print_exc()

    records = _collect_records(Path(ns.output_dir))
    metrics = _compute_metrics(records)

    # Promote successful + recoverable-failure trajectories into the runtime
    # memory bank so that the next run can retrieve lessons learned in this
    # run. Skipped automatically for the test split unless --promote-runtime-memory
    # always is set.
    promotion = _promote_records_to_runtime(ns, records)

    # Aggregate retrieval logs (best-effort).
    retrieval_logs_summary: Dict[str, Any] = {}
    try:
        cfg = RexConfig.from_env()
        retrieval_logs_summary = aggregate_logs(cfg.retrieval_log_dir)
    except Exception as exc:  # noqa: BLE001
        retrieval_logs_summary = {"error": f"{exc.__class__.__name__}: {exc}"}

    summary = {
        "benchmark": "tau-bench",
        "env": ns.env,
        "agent": ns.agent,
        "model": ns.model,
        "status": status,
        "error": error,
        "config": {
            "task_split": ns.task_split,
            "start_index": ns.start_index,
            "end_index": ns.end_index,
            "num_trials": ns.num_trials,
            "max_num_steps": ns.max_num_steps,
            "temperature": ns.temperature,
            "promote_runtime_memory": ns.promote_runtime_memory,
            "internal_train_end": int(getattr(ns, "internal_train_end", 0) or 0),
        },
        "metrics": metrics,
        "runtime_memory_promotion": promotion,
        "retrieval_logs": retrieval_logs_summary,
    }
    write_json(os.path.join(ns.output_dir, "metrics.json"), summary)
    print(json.dumps(summary["metrics"], indent=2))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
