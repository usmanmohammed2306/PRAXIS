"""Aggregate per-run metrics.json files into outputs/summary/{summary.json,summary.md}.

Renders a four-way comparison across the same fixed base model on three benchmarks:
  - tau-bench retail   (task-completion / success_rate)
  - tau-bench airline  (task-completion / success_rate)
  - BFCL V4            (tool-name coverage / success_rate)

Four controllers:
  1. baseline — vanilla tool-calling (minimal system prompt)
  2. act      — Act (Yao et al. 2022): action-only, no reasoning prose
  3. react    — ReAct (Yao et al. 2022): one-line Thought before each Action
  4. rex      — REx-RPE (this project's contribution): leakage-safe
                experience retrieval, native tool calling, and write-only
                SABER-style mutation reflection.

Each section renders a row per metric for all four conditions. Missing runs
are reported as ``status=missing`` so the table never collapses on partial
data.

The summary computes a ``deltas`` block per benchmark — REx-RPE minus
the strongest non-REx controller on success rate — to make the wins
immediately visible.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..common.io_utils import read_json, write_json

try:  # The new architecture provides retrieval-log aggregation.
    from ..rex.retrieval_logging import aggregate_logs as _aggregate_retrieval_logs
    from ..rex.config import RexConfig as _RexConfig
    from ..rex.memory_store import MemoryStore as _MemoryStore
except Exception:  # pragma: no cover - defensive: summary always renders
    _aggregate_retrieval_logs = None  # type: ignore[assignment]
    _RexConfig = None  # type: ignore[assignment]
    _MemoryStore = None  # type: ignore[assignment]


SECTIONS: List[Tuple[str, Dict[str, str]]] = [
    ("tau-bench retail", {
        "baseline": "tau_retail_baseline",
        "act": "tau_retail_act",
        "react": "tau_retail_react",
        "rex": "tau_retail_rex",
    }),
    ("tau-bench airline", {
        "baseline": "tau_airline_baseline",
        "act": "tau_airline_act",
        "react": "tau_airline_react",
        "rex": "tau_airline_rex",
    }),
    ("BFCL V4", {
        "baseline": "bfcl_agent_baseline",
        "act": "bfcl_agent_act",
        "react": "bfcl_agent_react",
        "rex": "bfcl_agent_rex",
    }),
]

CONDITIONS: List[str] = ["baseline", "act", "react", "rex"]
CONDITION_LABELS: Dict[str, str] = {
    "baseline": "Vanilla TC",
    "act": "Act",
    "react": "ReAct",
    "rex": "REx-RPE (ours)",
}

OURS = "rex"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("build_summary")
    p.add_argument("--outputs-dir", required=True)
    p.add_argument("--active-model", default="")
    p.add_argument("--served-name", default="")
    return p.parse_args()


def _load(outputs_dir: Path, subdir: str) -> Dict[str, Any]:
    path = outputs_dir / subdir / "metrics.json"
    data = read_json(path, default=None)
    if data is None:
        return {
            "status": "missing",
            "note": f"No metrics.json found at {path}",
            "metrics": {},
        }
    return data


def _pct(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{100.0 * float(x):.1f}%"
    except Exception:
        return "n/a"


def _num(x: Any, digits: int = 2) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "n/a"


def _rows_with_keys(label: str, by_cond: Dict[str, Dict[str, Any]],
                    rows_spec: List[Tuple[str, str, Any]]) -> List[str]:
    def cell(cond: str, key: str, fmt) -> str:
        metrics = (by_cond.get(cond, {}) or {}).get("metrics", {}) or {}
        return fmt(metrics.get(key))
    header = f"| {label} | Metric | " + " | ".join(CONDITION_LABELS[c] for c in CONDITIONS) + " |"
    sep = "|" + "|".join(["---"] * (2 + len(CONDITIONS))) + "|"
    rows: List[str] = []
    for key, human, fmt in rows_spec:
        row = f"| {label} | {human} | " + " | ".join(cell(c, key, fmt) for c in CONDITIONS) + " |"
        rows.append(row)
    status_row = f"| {label} | status | " + " | ".join(
        str((by_cond.get(c) or {}).get("status", "n/a")) for c in CONDITIONS
    ) + " |"
    rows.append(status_row)
    return [header, sep] + rows


def _tau_rows(label: str, by_cond: Dict[str, Dict[str, Any]]) -> List[str]:
    return _rows_with_keys(label, by_cond, [
        ("success_rate", "success rate", _pct),
        ("avg_reward", "avg reward", _num),
        ("num_tasks", "tasks", lambda x: str(x) if x is not None else "n/a"),
        ("error_tasks", "error tasks", lambda x: str(x) if x is not None else "n/a"),
        ("avg_trajectory_messages", "avg traj msgs", _num),
    ])


def _ace_rows(label: str, by_cond: Dict[str, Dict[str, Any]]) -> List[str]:
    return _rows_with_keys(label, by_cond, [
        ("completion_rate", "completion rate", _pct),
        ("tool_name_coverage", "tool-name coverage", _pct),
        ("avg_tool_calls", "avg tool calls", _num),
        ("avg_steps", "avg steps", _num),
        ("num_tasks", "tasks", lambda x: str(x) if x is not None else "n/a"),
    ])


def _bfcl_rows(label: str, by_cond: Dict[str, Dict[str, Any]]) -> List[str]:
    return _rows_with_keys(label, by_cond, [
        ("success_rate", "success rate (exact)", _pct),
        ("avg_tool_coverage", "avg tool coverage", _pct),
        ("avg_tool_calls", "avg tool calls", _num),
        ("avg_steps", "avg steps", _num),
        ("num_tasks", "tasks", lambda x: str(x) if x is not None else "n/a"),
        ("error_tasks", "error tasks", lambda x: str(x) if x is not None else "n/a"),
    ])


def _headline_metric_key(label: str) -> str:
    return "success_rate"


def _strongest_baseline(by_cond: Dict[str, Dict[str, Any]], key: str) -> Tuple[Optional[str], Optional[float]]:
    """Return (condition_name, value) for the best non-REx controller on ``key``."""
    best_cond: Optional[str] = None
    best_val: Optional[float] = None
    for c in ("baseline", "act", "react"):
        m = (by_cond.get(c, {}) or {}).get("metrics", {}) or {}
        v = m.get(key)
        if v is None:
            continue
        try:
            v_f = float(v)
        except Exception:
            continue
        if best_val is None or v_f > best_val:
            best_val = v_f
            best_cond = c
    return best_cond, best_val


def build(outputs_dir: Path, active_model: str, served_name: str) -> Dict[str, Any]:
    if not active_model:
        am_file = outputs_dir / "active_model.txt"
        if am_file.exists():
            try:
                active_model = am_file.read_text(encoding="utf-8").strip()
            except Exception:
                active_model = ""
    summary: Dict[str, Any] = {
        "active_model": active_model,
        "served_name": served_name,
        "sections": [],
        "deltas": [],
        "rex_memory": _build_rex_memory_block(outputs_dir),
        "retrieval_logs": _build_retrieval_logs_block(outputs_dir),
    }
    for label, subdirs in SECTIONS:
        by_cond = {cond: _load(outputs_dir, subdirs[cond]) for cond in CONDITIONS if cond in subdirs}
        summary["sections"].append({"label": label, "by_condition": by_cond})

        key = _headline_metric_key(label)
        best_cond, best_val = _strongest_baseline(by_cond, key)
        ours_metrics = (by_cond.get(OURS, {}) or {}).get("metrics", {}) or {}
        baseline_metrics = (by_cond.get("baseline", {}) or {}).get("metrics", {}) or {}
        ours_val = ours_metrics.get(key)
        baseline_val = baseline_metrics.get(key)
        try:
            ours_val_f = float(ours_val) if ours_val is not None else None
        except Exception:
            ours_val_f = None
        try:
            baseline_val_f = float(baseline_val) if baseline_val is not None else None
        except Exception:
            baseline_val_f = None
        delta: Dict[str, Any] = {
            "section": label,
            "metric": key,
            "best_baseline": best_cond,
            "best_baseline_value": best_val,
            "rex_value": ours_val_f,
            "baseline_value": baseline_val_f,
            "delta_vs_best_baseline": (
                ours_val_f - best_val
                if ours_val_f is not None and best_val is not None
                else None
            ),
            "delta_vs_baseline": (
                ours_val_f - baseline_val_f
                if ours_val_f is not None and baseline_val_f is not None
                else None
            ),
        }
        summary["deltas"].append(delta)
    return summary


def _build_rex_memory_block(outputs_dir: Path) -> Dict[str, Any]:
    """Snapshot the seed + v2 runtime memory bank sizes."""
    if _MemoryStore is None or _RexConfig is None:
        return {}
    block: Dict[str, Any] = {}
    try:
        cfg = _RexConfig.from_env()
        # Legacy v1 runtime
        v1_dir = Path(cfg.runtime_dir)
        block["seed_dir"] = str(cfg.bank_dir)
        block["runtime_dir"] = str(v1_dir)
        block["v1_card_counts"] = _count_jsonl_files(v1_dir)
        block["seed_card_counts"] = _count_jsonl_files(Path(cfg.bank_dir))
        # v2 ProcessMemoryCard runtime nests under runtime/v2 in the runners.
        v2_root = v1_dir / "v2"
        if v2_root.exists():
            store = _MemoryStore(seed_dir=cfg.bank_dir, runtime_dir=v2_root, config=cfg)
            block["v2_memory_statistics"] = store.memory_statistics()
            block["v2_retrieval_statistics"] = store.retrieval_statistics()
    except Exception as exc:  # noqa: BLE001
        block["error"] = f"{exc.__class__.__name__}: {exc}"
    return block


def _count_jsonl_files(path: Path) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not path.exists():
        return out
    for child in sorted(path.iterdir()):
        if child.is_file() and child.suffix == ".jsonl":
            n = 0
            try:
                with child.open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            n += 1
            except OSError:
                continue
            out[child.stem] = n
    return out


def _build_retrieval_logs_block(outputs_dir: Path) -> Dict[str, Any]:
    """Aggregate retrieval logs under outputs/retrieval_logs."""
    if _aggregate_retrieval_logs is None:
        return {}
    candidate = outputs_dir / "retrieval_logs"
    if not candidate.exists() and _RexConfig is not None:
        try:
            candidate = Path(_RexConfig.from_env().retrieval_log_dir)
        except Exception:
            return {}
    try:
        return _aggregate_retrieval_logs(candidate)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{exc.__class__.__name__}: {exc}"}


def render_markdown(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Vanilla / Act / ReAct / REx-RPE — Comparison Summary")
    lines.append("")
    lines.append(f"- Active model: `{summary.get('active_model') or '(unknown)'}`")
    lines.append(f"- Served name:  `{summary.get('served_name') or '(unknown)'}`")
    lines.append("")
    lines.append(
        "Same fixed base model and same in-process loop across all four conditions. "
        "Vanilla TC = minimal system prompt with native function-calling. "
        "Act = action-only, no reasoning prose (Yao et al. 2022). "
        "ReAct = one-line Thought before each Action (Yao et al. 2022). "
        "REx-RPE = leakage-safe experience retrieval from allowed support "
        "data, native tool calling, brief startup analysis, and write-only "
        "SABER-style mutation reflection."
    )
    lines.append("")
    lines.append("## Headline deltas (REx-RPE vs. best of vanilla / Act / ReAct, and vs. baseline)")
    lines.append("")
    lines.append("| Benchmark | Metric | Best baseline | Best baseline val | REx-RPE | Δ vs best | Δ vs baseline |")
    lines.append("|---|---|---|---|---|---|---|")
    for d in summary.get("deltas", []):
        bb = d.get("best_baseline") or "n/a"
        bb_label = CONDITION_LABELS.get(bb, bb)
        bbv = d.get("best_baseline_value")
        cgv = d.get("rex_value")
        dv = d.get("delta_vs_best_baseline")
        dvb = d.get("delta_vs_baseline")
        is_pct_metric = d.get("metric") in ("success_rate", "completion_rate", "tool_name_coverage")
        bbv_s = _pct(bbv) if is_pct_metric else _num(bbv)
        cgv_s = _pct(cgv) if is_pct_metric else _num(cgv)

        def _fmt(x):
            if x is None:
                return "n/a"
            return f"{100.0 * x:+.1f} pp" if is_pct_metric else f"{x:+.2f}"

        lines.append(f"| {d.get('section')} | {d.get('metric')} | {bb_label} | "
                     f"{bbv_s} | {cgv_s} | {_fmt(dv)} | {_fmt(dvb)} |")
    lines.append("")
    lines.append("## Per-benchmark detail")
    lines.append("")
    for section in summary["sections"]:
        label = section["label"]
        by_cond = section["by_condition"]
        lines.append(f"### {label}")
        if label.startswith("BFCL"):
            lines.extend(_bfcl_rows(label, by_cond))
        else:
            lines.extend(_tau_rows(label, by_cond))
        lines.append("")
    # REx-specific diagnostics block.
    lines.append("## REx diagnostics")
    lines.append("")
    lines.append("| Benchmark | Trajectories w/ stats | Reflection calls | Reflection blocks | Tool calls executed | Mutating calls | Avg cards loaded | Avg cards used |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for section in summary["sections"]:
        label = section["label"]
        by_cond = section.get("by_condition") or {}
        rex_data = (by_cond.get(OURS) or {}).get("metrics") or {}
        rd = rex_data.get("rex_diagnostics") or {}
        lines.append(
            f"| {label} | {rd.get('trajectories_with_stats', 0)} | "
            f"{rd.get('reflection_calls', 0)} | {rd.get('reflection_blocks', 0)} | "
            f"{rd.get('tool_calls_executed', 0)} | {rd.get('mutating_tool_calls', 0)} | "
            f"{_num(rd.get('avg_cards_loaded'), 1)} | {_num(rd.get('avg_cards_used'), 1)} |"
        )
    # ----- REx procedural-memory diagnostics ---------------------------
    rex_memory = summary.get("rex_memory") or {}
    if rex_memory:
        lines.append("")
        lines.append("## REx procedural-memory bank")
        lines.append("")
        seed = rex_memory.get("seed_card_counts") or {}
        v1 = rex_memory.get("v1_card_counts") or {}
        if seed:
            lines.append("**Seed bank** (allowed-only support data):")
            for d, n in sorted(seed.items()):
                lines.append(f"- `{d}`: {n} cards")
        if v1:
            lines.append("")
            lines.append("**Runtime bank v1** (legacy ExperienceCard JSONL):")
            for d, n in sorted(v1.items()):
                lines.append(f"- `{d}`: {n} cards")
        v2_mem = rex_memory.get("v2_memory_statistics") or {}
        if v2_mem and v2_mem.get("by_domain"):
            lines.append("")
            lines.append("**Runtime bank v2** (ProcessMemoryCard, deduped seed+runtime):")
            for d, stats in sorted(v2_mem["by_domain"].items()):
                lines.append(
                    f"- `{d}`: seed={stats.get('seed', 0)}, "
                    f"runtime={stats.get('runtime', 0)}, "
                    f"combined_dedup={stats.get('combined_dedup', 0)}"
                )
        v2_use = rex_memory.get("v2_retrieval_statistics") or {}
        if v2_use and v2_use.get("by_domain"):
            lines.append("")
            lines.append("**v2 usage statistics:**")
            for d, stats in sorted(v2_use["by_domain"].items()):
                lines.append(
                    f"- `{d}`: total_uses={stats.get('total_uses', 0)}, "
                    f"avg_confidence={stats.get('avg_confidence', 0.0):.2f}"
                )

    retrieval_logs = summary.get("retrieval_logs") or {}
    totals = retrieval_logs.get("totals") if isinstance(retrieval_logs, dict) else None
    if totals:
        lines.append("")
        lines.append("## Retrieval diagnostics")
        lines.append("")
        lines.append(f"- total events logged: {totals.get('events', 0)}")
        lines.append(f"- avg corpus size at retrieval time: {totals.get('avg_corpus', 0.0):.1f}")
        lines.append(f"- avg playbook size: {totals.get('avg_playbook_chars', 0.0):.1f} chars")
        lines.append(f"- unique cards used across runs: {totals.get('unique_card_ids', 0)}")
        by_file = retrieval_logs.get("by_file") or {}
        if by_file:
            lines.append("")
            lines.append("Per-(benchmark, env, controller) breakdown:")
            for fname, stats in sorted(by_file.items()):
                lines.append(
                    f"- `{fname}`: events={stats.get('events', 0)}, "
                    f"avg_corpus={stats.get('avg_corpus', 0.0):.1f}, "
                    f"avg_playbook={stats.get('avg_playbook_chars', 0.0):.1f}"
                )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for section in summary["sections"]:
        for cond in CONDITIONS:
            data = (section.get("by_condition") or {}).get(cond) or {}
            note = data.get("note")
            if note:
                lines.append(f"- **{section['label']} / {CONDITION_LABELS[cond]}**: {note}")
    lines.append("")
    lines.append("## Method notes")
    lines.append("- Same fixed base model, same temperature, same max-steps, same "
                 "truncation budget across all four controllers. Baselines call the "
                 "raw tool surface; REx-RPE keeps native function-calling but adds "
                 "leakage-safe procedural retrieval and write-only reflection.")
    lines.append("- Vanilla TC: minimal role + policy in the system prompt; native "
                 "function-calling does the rest.")
    lines.append("- Act: prompt instructs the model to emit tool calls only, no "
                 "reasoning prose.")
    lines.append("- ReAct: prompt requires one short `Thought:` line before each tool "
                 "call (capped to ~20 words to keep prompt growth bounded).")
    lines.append("- REx-RPE: retrieves top procedural experience cards from an "
                 "audited bank built from allowed support data, renders them as "
                 "process guidance with explicit no-copy-ID rules, runs one brief "
                 "no-tools startup analysis, then uses native tool calls. Only "
                 "mutating tau-bench tools trigger same-model SABER-style reflection; "
                 "reads are never blocked by the REx layer.")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    outputs_dir = Path(args.outputs_dir)
    summary = build(outputs_dir, args.active_model, args.served_name)

    out_dir = outputs_dir / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary.json", summary)
    (out_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(f"Wrote {out_dir / 'summary.json'}")
    print(f"Wrote {out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
