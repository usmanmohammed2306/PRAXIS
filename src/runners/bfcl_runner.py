"""BFCL V4 (Berkeley Function Calling Leaderboard v4) runner.

Data expected at:  external/bfcl/  (or $BFCL_DATA_DIR)

Accepted filenames (any of):
  BFCL_v4_<category>.json / .jsonl    — official BFCL v4 JSON arrays
  BFCL_v3_<category>.json / .jsonl    — fallback v3 format
  <category>.json / .jsonl            — bare category name

Download dataset:
  python -m src.scripts.download_bfcl --output-dir external/bfcl

Scoring metric:
  tool_name_coverage   — fraction of expected function names the agent called
  success_rate         — fraction of tasks where coverage == 1.0 (exact match)

Experience distillation:
  Ground truth provides a proper reward signal. Parallel/multi-turn categories
  have sequential tool patterns that the ProcessMemoryCard distiller captures
  as tool_ordering_hints — unlike ACEBench's isolated single-call tasks.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..common.io_utils import append_jsonl, ensure_dir, safe_mean, write_json
from ..rex.config import RexConfig
from ..rex.memory_quality import consolidate
from ..rex.memory_store import MemoryStore
from ..rex.pipeline import promote_records as pipeline_promote_records
from ..rex.retrieval_logging import aggregate_logs

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESPOND_TOOL_NAME = "respond"

DEFAULT_CATEGORIES: List[str] = [
    "simple",
    "multiple",
    "parallel",
    "parallel_multiple",
    "multi_turn_base",
]

SMOKE_CATEGORIES: List[str] = ["simple", "multiple"]


# ---------------------------------------------------------------------------
# Minimal tau-bench–compatible environment for BFCL tasks
# ---------------------------------------------------------------------------

class _BFCLResult:
    """Duck-typed tau-bench SolveResult / step result shim."""

    def __init__(
        self,
        observation: str = "",
        reward: float = 0.0,
        done: bool = False,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info or {}


class BFCLEnv:
    """Thin tau-bench–compatible environment for a single BFCL task.

    reset() → presents the user query.
    step()  → records tool calls; returns a generic JSON success observation.
              A ``respond`` action (or reaching max_steps) marks done=True
              and computes final tool-name-coverage reward.
    """

    def __init__(self, task: Dict[str, Any], max_steps: int = 15) -> None:
        self.task = task
        self.max_steps = max_steps
        self._tool_calls: List[Dict[str, Any]] = []
        self._step_count = 0

    # ---- tau-bench interface ------------------------------------------------

    def reset(self, task_index: Optional[int] = None) -> _BFCLResult:
        self._tool_calls = []
        self._step_count = 0
        return _BFCLResult(observation=_extract_user_query(self.task))

    def step(self, action: Any) -> _BFCLResult:
        name = str(getattr(action, "name", "") or "")
        kwargs = dict(getattr(action, "kwargs", {}) or {})

        # Record before incrementing so the last tool in a max-step scenario
        # is captured.
        if name and name != RESPOND_TOOL_NAME:
            self._tool_calls.append({"name": name, "kwargs": kwargs})

        self._step_count += 1
        terminal = (name == RESPOND_TOOL_NAME) or (self._step_count >= self.max_steps)

        if terminal:
            reward = _compute_reward(self._tool_calls, self.task)
            return _BFCLResult(
                observation="",
                reward=reward,
                done=True,
                info={
                    "domain": "bfcl",
                    "environment": "bfcl",
                    "controller": "bfcl",
                    "tool_coverage": reward,
                    "steps": self._step_count,
                },
            )

        obs = json.dumps(
            {"status": "ok", "result": f"{name} executed successfully", "name": name},
            ensure_ascii=False,
        )
        return _BFCLResult(observation=obs, reward=0.0, done=False)

    # ---- properties used by _solve_one -------------------------------------

    @property
    def tools_info(self) -> List[Dict[str, Any]]:
        """Available tools in OpenAI function-calling format."""
        funcs = self.task.get("function") or []
        if isinstance(funcs, dict):
            funcs = [funcs]
        return [
            _to_openai_tool(f)
            for f in funcs
            if isinstance(f, dict) and f.get("name")
        ]

    @property
    def wiki(self) -> str:
        return ""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_bfcl_tasks(
    data_dir: Path,
    categories: List[str],
    limit: int,
) -> List[Dict[str, Any]]:
    """Load BFCL tasks from local files; supports v3/v4 naming and JSON/JSONL."""
    tasks: List[Dict[str, Any]] = []
    for cat in categories:
        candidates = [
            f"BFCL_v4_{cat}.json",
            f"BFCL_v4_{cat}.jsonl",
            f"BFCL_v3_{cat}.json",
            f"BFCL_v3_{cat}.jsonl",
            f"{cat}.json",
            f"{cat}.jsonl",
        ]
        for fname in candidates:
            path = data_dir / fname
            if not path.exists():
                continue
            try:
                loaded = _read_jsonl_or_json(path)
                # Tag each task with its category so the train/test splitter
                # can partition per-category deterministically.
                for t in loaded:
                    t.setdefault("_category", cat)
                tasks.extend(loaded)
                print(f"[bfcl] loaded {len(loaded)} tasks from {fname}", file=sys.stderr)
                break
            except Exception as exc:
                print(f"[bfcl] warning: failed to read {fname}: {exc}", file=sys.stderr)
        else:
            print(f"[bfcl] warning: no data file found for category '{cat}' in {data_dir}", file=sys.stderr)

    if limit > 0 and len(tasks) > limit:
        tasks = tasks[:limit]
    return tasks


def _read_jsonl_or_json(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    # Try JSON array first
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass
    # Fall back to JSONL
    rows: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
        except json.JSONDecodeError:
            continue
    return rows


# ---------------------------------------------------------------------------
# Tool format conversion
# ---------------------------------------------------------------------------

def _to_openai_tool(func_def: Dict[str, Any]) -> Dict[str, Any]:
    """Convert BFCL function definition to OpenAI tool schema."""
    params = func_def.get("parameters", {"type": "object", "properties": {}})
    if isinstance(params, dict):
        params = _normalize_param_types(params)
    return {
        "type": "function",
        "function": {
            "name": str(func_def["name"]),
            "description": str(func_def.get("description", "")),
            "parameters": params,
        },
    }


_TYPE_MAP = {"dict": "object", "str": "string", "int": "integer", "bool": "boolean", "float": "number"}


def _normalize_param_types(obj: Any) -> Any:
    """Recursively map Python-style type names to JSON Schema names."""
    if not isinstance(obj, dict):
        return obj
    out: Dict[str, Any] = {}
    for k, v in obj.items():
        if k == "type" and isinstance(v, str):
            out[k] = _TYPE_MAP.get(v, v)
        elif isinstance(v, dict):
            out[k] = _normalize_param_types(v)
        elif isinstance(v, list):
            out[k] = [_normalize_param_types(i) for i in v]
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Query / ground-truth helpers
# ---------------------------------------------------------------------------

def _extract_user_query(task: Dict[str, Any]) -> str:
    """Return the user-facing text from a BFCL task."""
    q = task.get("question") or task.get("prompt") or ""
    if isinstance(q, str):
        return q
    if isinstance(q, list):
        # [[{"role": "user", "content": "..."}], ...] — take first turn
        if q and isinstance(q[0], list):
            q = q[0]
        if q and isinstance(q[0], dict):
            return next((m.get("content", "") for m in q if m.get("role") == "user"), str(q))
    return str(q)


def _extract_expected_names(task: Dict[str, Any]) -> Set[str]:
    """Return the set of function names the agent is expected to call."""
    gt = task.get("ground_truth") or []
    if isinstance(gt, str):
        gt = [gt]
    names: Set[str] = set()
    for entry in gt:
        if isinstance(entry, str):
            # "func_name(arg=val)" or "func_name"
            m = re.match(r"([A-Za-z0-9_.]+)\s*[\(\[]?", entry.strip())
            if m:
                names.add(m.group(1))
        elif isinstance(entry, dict):
            names.update(entry.keys())
    return names


def _extract_actual_names_from_messages(messages: List[Dict[str, Any]]) -> Set[str]:
    names: Set[str] = set()
    for msg in messages:
        if msg.get("role") == "tool" and msg.get("name"):
            names.add(str(msg["name"]))
        elif msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                fn = (tc.get("function") or {}).get("name")
                if fn:
                    names.add(str(fn))
    return names


def _compute_reward(tool_calls: List[Dict[str, Any]], task: Dict[str, Any]) -> float:
    expected = _extract_expected_names(task)
    if not expected:
        return 0.5 if tool_calls else 0.0
    actual = {tc["name"] for tc in tool_calls if tc.get("name")}
    return len(actual & expected) / len(expected)


# ---------------------------------------------------------------------------
# Agent resolution
# ---------------------------------------------------------------------------

def _resolve_agent_cls(kind: str):
    if kind in ("baseline", "tool-calling"):
        from ..baselines.agents import ToolCallingAgent
        return ToolCallingAgent
    if kind == "act":
        from ..baselines.agents import ActAgent
        return ActAgent
    if kind == "react":
        from ..baselines.agents import ReActAgent
        return ReActAgent
    if kind in ("rex", "praxis"):
        from ..rex.agent import RexAgent
        return RexAgent
    raise ValueError(f"Unknown agent kind: {kind!r}")


# ---------------------------------------------------------------------------
# Single-task solver
# ---------------------------------------------------------------------------

def _solve_one(
    ns: argparse.Namespace,
    AgentCls,
    task_idx: int,
    task: Dict[str, Any],
) -> Dict[str, Any]:
    env = BFCLEnv(task, max_steps=ns.max_num_steps)
    agent_kwargs: Dict[str, Any] = dict(
        tools_info=env.tools_info,
        wiki=env.wiki,
        model=ns.model,
        provider=ns.model_provider,
        temperature=float(ns.temperature),
    )
    if ns.agent in ("rex", "praxis"):
        agent_kwargs["env_hint"] = "bfcl"

    expected = _extract_expected_names(task)
    try:
        agent = AgentCls(**agent_kwargs)
        result = agent.solve(env, task_index=task_idx, max_num_steps=ns.max_num_steps)
        reward = float(getattr(result, "reward", 0.0) or 0.0)
        info = dict(getattr(result, "info", {}) or {})
        messages = list(getattr(result, "messages", []) or [])
        # Cross-check coverage from message history
        actual_names = _extract_actual_names_from_messages(messages)
        if expected and actual_names & expected:
            coverage = len(actual_names & expected) / len(expected)
            reward = max(reward, coverage)
        status = "ok"
        err = ""
    except Exception as exc:
        traceback.print_exc()
        reward, info, messages, actual_names = 0.0, {"error": str(exc)}, [], set()
        status = "error"
        err = f"{exc.__class__.__name__}: {exc}"

    info.setdefault("domain", "bfcl")
    info.setdefault("environment", "bfcl")
    info.setdefault("controller", ns.agent)
    # Stamp task_split so the promotion pipeline can drop test-split records
    # via _is_test_split_record (defence-in-depth on top of run-level gating).
    split_label = task.get("_split_label")
    if split_label:
        info["task_split"] = split_label
        info.setdefault("category", task.get("_category"))

    return {
        "task_id": task.get("id") or task_idx,
        "reward": reward,
        "tool_coverage": reward,
        "expected_tools": sorted(expected),
        "actual_tools": sorted(actual_names),
        "messages": messages,
        "status": status,
        "error": err,
        "info": info,
    }


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def _apply_train_test_split(
    tasks: List[Dict[str, Any]],
    train_fraction: float,
    keep_split: str,
) -> List[Dict[str, Any]]:
    """Assign each task a deterministic train/test label per category and
    optionally filter the list down to one side of the split.

    - Per-category: tasks are sorted by their stable id and the first
      ``ceil(N * train_fraction)`` become "train"; the rest are "test".
    - When ``keep_split`` is "train" or "test", only that subset is returned;
      "all" (default) returns every task with its label stamped in place.
    - When ``train_fraction <= 0`` no labels are added and all tasks are kept;
      this preserves legacy behaviour for callers that opt out of the split.
    """
    import math
    if train_fraction <= 0.0 or not tasks:
        return tasks
    train_fraction = min(max(train_fraction, 0.0), 1.0)

    # Group by category, sort each group deterministically, label, then
    # rebuild the original list order so trajectories.jsonl ordering is stable.
    by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for t in tasks:
        by_cat.setdefault(str(t.get("_category", "default")), []).append(t)

    train_ids: set = set()
    for cat, group in by_cat.items():
        group_sorted = sorted(group, key=lambda x: str(x.get("id", "")))
        n_train = math.ceil(len(group_sorted) * train_fraction)
        for t in group_sorted[:n_train]:
            train_ids.add(id(t))  # use object identity — ids may collide

    for t in tasks:
        t["_split_label"] = "train" if id(t) in train_ids else "test"

    keep = (keep_split or "all").strip().lower()
    if keep in ("train", "test"):
        return [t for t in tasks if t.get("_split_label") == keep]
    return tasks


def _run_inprocess(ns: argparse.Namespace) -> None:
    data_dir = Path(ns.data_dir)
    if not data_dir.exists():
        print(
            f"[bfcl] ERROR: data directory not found: {data_dir}\n"
            f"       Download with:  python -m src.scripts.download_bfcl --output-dir {data_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    cats = (
        [c.strip() for c in ns.categories.split(",") if c.strip()]
        if ns.categories
        else DEFAULT_CATEGORIES
    )
    tasks = _load_bfcl_tasks(data_dir, cats, ns.limit)
    if not tasks:
        print(f"[bfcl] ERROR: no tasks loaded from {data_dir} (categories={cats})", file=sys.stderr)
        sys.exit(1)

    # Apply the deterministic per-category train/test split if requested.
    # BFCL has no upstream split distinction, so we manufacture one to keep
    # PRAXIS evaluation honest: the first `train_fraction` of each category
    # is used for PRAXIS pre-warm; the remainder is the held-out test set on
    # which all four controllers are evaluated. Records keep their split label
    # in `info.task_split` so the pipeline can defence-in-depth block test
    # records from being distilled into the memory bank.
    train_fraction = float(getattr(ns, "train_fraction", 0.0) or 0.0)
    keep_split = getattr(ns, "task_split", "all") or "all"
    tasks = _apply_train_test_split(tasks, train_fraction, keep_split)
    if not tasks:
        print(
            f"[bfcl] ERROR: split filter '{keep_split}' left zero tasks "
            f"(train_fraction={train_fraction})",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"[bfcl] running {len(tasks)} tasks  agent={ns.agent}  "
        f"train_fraction={train_fraction}  keep_split={keep_split}",
        file=sys.stderr,
    )

    AgentCls = _resolve_agent_cls(ns.agent)
    out_dir = Path(ns.output_dir)
    ensure_dir(out_dir)
    jsonl_path = out_dir / "trajectories.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    records: List[Dict[str, Any]] = []
    write_lock = threading.Lock()
    max_workers = max(1, int(ns.max_concurrency))

    if max_workers == 1:
        for idx, task in enumerate(tasks):
            rec = _solve_one(ns, AgentCls, idx, task)
            append_jsonl(jsonl_path, rec)
            records.append(rec)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(_solve_one, ns, AgentCls, idx, task): idx
                for idx, task in enumerate(tasks)
            }
            for fut in as_completed(futures):
                rec = fut.result()
                with write_lock:
                    append_jsonl(jsonl_path, rec)
                    records.append(rec)

    records.sort(key=lambda r: r.get("task_id", 0))
    write_json(out_dir / f"results-{ns.agent}.json", records)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {"num_tasks": 0, "success_rate": 0.0, "avg_tool_coverage": 0.0}
    coverages = [float(r.get("tool_coverage", 0.0)) for r in records]
    successes = [1.0 if c >= 1.0 else 0.0 for c in coverages]
    avg_tool_calls = safe_mean([len(r.get("actual_tools") or []) for r in records])
    avg_steps = safe_mean([len(r.get("messages") or []) // 2 for r in records])
    return {
        "num_tasks": len(records),
        "success_rate": safe_mean(successes),
        "avg_tool_coverage": safe_mean(coverages),
        "avg_tool_calls": avg_tool_calls,
        "avg_steps": avg_steps,
        "tasks_with_ground_truth": sum(1 for r in records if r.get("expected_tools")),
        "error_tasks": sum(1 for r in records if r.get("status") == "error"),
    }


# ---------------------------------------------------------------------------
# Experience promotion
# ---------------------------------------------------------------------------

def _promote_bfcl_records(
    ns: argparse.Namespace,
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    mode = (getattr(ns, "promote_runtime_memory", None) or "auto").lower()
    if mode == "never":
        print("[memory] promotion skipped: promote_mode=never", file=sys.stderr)
        return {"status": "skipped", "reason": "promote_mode=never"}

    runtime_dir = Path(ns.runtime_dir) if getattr(ns, "runtime_dir", None) else None
    print(
        f"[memory] promoting {len(records)} BFCL trajectories (promote_mode={mode})",
        file=sys.stderr,
    )
    try:
        # allow_test_split is True only when the user explicitly opts in via
        # --promote-runtime-memory always (e.g. for ablation runs). In auto
        # mode the per-record `_is_test_split_record` gate drops test-labeled
        # records — this is what keeps PRAXIS from learning on its eval set.
        manifest = pipeline_promote_records(
            records,
            domain="bfcl",
            runtime_dir=runtime_dir,
            allow_test_split=mode == "always",
        )
        manifest["status"] = "ok"
        promoted = manifest.get("promoted", 0)
        total_after = manifest.get("total_runtime_cards_after", 0)
        print(
            f"[memory] promoted: {promoted} BFCL cards → total runtime={total_after}",
            file=sys.stderr,
        )

        # Consolidation pass
        try:
            cfg = RexConfig.from_env()
            store_dir = runtime_dir if runtime_dir else cfg.runtime_dir
            store = MemoryStore(seed_dir=cfg.bank_dir, runtime_dir=store_dir, config=cfg)
            cards = store.load_runtime("bfcl")
            if cards:
                report = consolidate(cards, config=cfg)
                manifest["consolidation"] = report.to_dict()
                print(
                    f"[memory] consolidation: {len(cards)} BFCL cards, "
                    f"duplicates={len(report.duplicates)}, "
                    f"decay_applied={report.decayed}",
                    file=sys.stderr,
                )
        except Exception as exc:  # noqa: BLE001
            manifest["consolidation_error"] = f"{exc.__class__.__name__}: {exc}"
            print(f"[memory] consolidation failed: {exc.__class__.__name__}", file=sys.stderr)

        print(f"[memory] promotion complete: status={manifest['status']}", file=sys.stderr)
        return manifest
    except Exception as exc:  # noqa: BLE001
        print(f"[memory] promotion error: {exc.__class__.__name__}: {exc}", file=sys.stderr)
        return {"status": "error", "error": f"{exc.__class__.__name__}: {exc}"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("bfcl_runner", description="BFCL V4 evaluation runner")
    p.add_argument("--agent", default="baseline",
                   choices=["baseline", "act", "react", "rex", "praxis"])
    p.add_argument("--model", default="qwen-agent")
    p.add_argument("--model-provider", dest="model_provider", default="openai")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--data-dir", dest="data_dir",
                   default=str(Path(__file__).parents[2] / "external" / "bfcl"))
    p.add_argument("--categories", default="",
                   help=f"Comma-separated BFCL categories. Default: {','.join(DEFAULT_CATEGORIES)}")
    p.add_argument("--limit", type=int, default=0,
                   help="Max tasks to run per category combined (0 = no limit)")
    p.add_argument("--max-concurrency", dest="max_concurrency", type=int, default=2)
    p.add_argument("--max-num-steps", dest="max_num_steps", type=int, default=15)
    p.add_argument("--output-dir", dest="output_dir", required=True)
    p.add_argument("--runtime-dir", dest="runtime_dir", default="")
    p.add_argument("--promote-runtime-memory", dest="promote_runtime_memory", default="auto",
                   choices=["auto", "always", "never"])
    p.add_argument(
        "--train-fraction",
        dest="train_fraction",
        type=float,
        default=0.0,
        help=(
            "Per-category fraction of BFCL tasks assigned to the internal "
            "'train' split (used for PRAXIS pre-warm). The remainder is the "
            "held-out 'test' split on which all four controllers are evaluated. "
            "0.0 disables the split (legacy behaviour)."
        ),
    )
    p.add_argument(
        "--task-split",
        dest="task_split",
        default="all",
        choices=["all", "train", "test"],
        help=(
            "Filter to one side of the internal train/test split. Requires "
            "--train-fraction > 0 to take effect."
        ),
    )
    return p.parse_args()


def main() -> int:
    ns = _parse_args()
    ensure_dir(ns.output_dir)
    out_dir = Path(ns.output_dir)

    status = "ok"
    error = ""
    try:
        _run_inprocess(ns)
    except SystemExit:
        raise
    except Exception as exc:
        status = "error"
        error = f"{exc.__class__.__name__}: {exc}"
        traceback.print_exc()

    # Load records for metrics
    records: List[Dict[str, Any]] = []
    result_path = out_dir / f"results-{ns.agent}.json"
    try:
        if result_path.exists():
            records = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            traj = out_dir / "trajectories.jsonl"
            if traj.exists():
                for line in traj.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
    except Exception:
        pass

    metrics = _compute_metrics(records)
    metrics["status"] = status
    if error:
        metrics["error"] = error

    write_json(out_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))

    promotion = _promote_bfcl_records(ns, records)

    try:
        cfg = RexConfig.from_env()
        retrieval_logs: Dict[str, Any] = aggregate_logs(cfg.retrieval_log_dir)
    except Exception:
        retrieval_logs = {}

    write_json(out_dir / "run_summary.json", {
        "metrics": metrics,
        "promotion": promotion,
        "retrieval_logs": retrieval_logs,
    })

    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
