"""Persistent retrieval diagnostics.

A small JSONL writer that the agent loops use to log:

  * per-step retrieval queries
  * retrieved card ids + scores
  * playbook char_count + section names
  * working state snapshot

Logs land under ``$REX_RETRIEVAL_LOG_DIR`` (default
``outputs/retrieval_logs/``). They are useful both for debugging and for
the post-run summary (``build_summary.py``) which aggregates them.

The logger is *non-fatal*: any I/O error is swallowed so that retrieval
diagnostics never crash a benchmark run.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .config import RexConfig, default_config
from .memory_types import RetrievalQuery, RetrievalResult, TacticalPlaybook


_GLOBAL_LOCK = threading.Lock()


@dataclass
class RetrievalEvent:
    """One step's retrieval activity."""

    ts_ms: int
    benchmark: str
    environment: str
    controller: str
    task_id: str
    trial: int
    step_index: int
    query_text: str
    last_tool: str
    last_observation_excerpt: str
    selected_card_ids: List[str]
    selected_intents: List[str]
    selected_scores: List[float]
    num_corpus: int
    playbook_char_count: int
    playbook_sections: List[str]
    diagnostic: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts_ms": int(self.ts_ms),
            "benchmark": self.benchmark,
            "environment": self.environment,
            "controller": self.controller,
            "task_id": str(self.task_id),
            "trial": int(self.trial),
            "step_index": int(self.step_index),
            "query_text": self.query_text,
            "last_tool": self.last_tool,
            "last_observation_excerpt": self.last_observation_excerpt,
            "selected_card_ids": list(self.selected_card_ids),
            "selected_intents": list(self.selected_intents),
            "selected_scores": [float(s) for s in self.selected_scores],
            "num_corpus": int(self.num_corpus),
            "playbook_char_count": int(self.playbook_char_count),
            "playbook_sections": list(self.playbook_sections),
            "diagnostic": dict(self.diagnostic),
        }


class RetrievalLogger:
    """Append-only JSONL retrieval logger, one file per (benchmark, environment, controller)."""

    def __init__(
        self,
        *,
        config: Optional[RexConfig] = None,
        log_dir: Optional[Path] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        self.config = config or default_config()
        self.log_dir = Path(log_dir) if log_dir else Path(self.config.retrieval_log_dir)
        self.enabled = bool(self.config.retrieval_log_enabled if enabled is None else enabled)
        self._max_records = max(0, int(self.config.retrieval_log_max_records))
        self._counts: Dict[str, int] = {}

    def _path(self, benchmark: str, environment: str, controller: str) -> Path:
        safe = lambda s: (s or "unknown").lower().replace("/", "_").replace(" ", "_")  # noqa: E731
        fn = f"{safe(benchmark)}_{safe(environment)}_{safe(controller)}.jsonl"
        return self.log_dir / fn

    def log(
        self,
        *,
        benchmark: str,
        environment: str,
        controller: str,
        task_id: str,
        trial: int,
        step_index: int,
        query: RetrievalQuery,
        result: RetrievalResult,
        playbook: Optional[TacticalPlaybook] = None,
    ) -> None:
        if not self.enabled:
            return
        try:
            event = RetrievalEvent(
                ts_ms=int(time.time() * 1000),
                benchmark=benchmark or "unknown",
                environment=environment or "generic",
                controller=controller or "rex",
                task_id=str(task_id),
                trial=int(trial or 0),
                step_index=int(step_index or 0),
                query_text=str(query.text or "")[:500],
                last_tool=str(query.last_tool or ""),
                last_observation_excerpt=str(query.last_observation or "")[:240],
                selected_card_ids=result.card_ids(),
                selected_intents=[c.task_category for c in result.cards],
                selected_scores=list(result.scores),
                num_corpus=int((result.diagnostic or {}).get("num_corpus", 0) or 0),
                playbook_char_count=playbook.char_count if playbook else 0,
                playbook_sections=list((playbook.sections or {}).keys()) if playbook else [],
                diagnostic=dict(result.diagnostic or {}),
            )
            self._append(self._path(benchmark, environment, controller), event)
        except Exception:
            # Never crash the agent loop on logging errors.
            return

    def _append(self, path: Path, event: RetrievalEvent) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        key = str(path)
        with _GLOBAL_LOCK:
            count = self._counts.get(key)
            if count is None:
                count = 0
                if path.exists():
                    try:
                        with path.open("r", encoding="utf-8") as f:
                            for _ in f:
                                count += 1
                    except OSError:
                        count = 0
            if self._max_records and count >= self._max_records:
                return
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict(), ensure_ascii=False, default=str) + "\n")
            self._counts[key] = count + 1


def aggregate_logs(log_dir: os.PathLike) -> Dict[str, Any]:
    """Read every JSONL log under ``log_dir`` and return aggregate stats.

    Used by ``build_summary.py`` to report retrieval diagnostics. The
    result is intentionally small (no per-event detail) so it serializes
    cleanly into ``summary.json``.
    """
    p = Path(log_dir)
    out: Dict[str, Any] = {"by_file": {}, "totals": {
        "events": 0,
        "avg_corpus": 0.0,
        "avg_playbook_chars": 0.0,
        "unique_card_ids": 0,
    }}
    if not p.exists():
        return out
    seen_ids: set[str] = set()
    total_corpus = 0
    total_playbook = 0
    total_events = 0
    for child in sorted(p.iterdir()):
        if not child.is_file() or child.suffix != ".jsonl":
            continue
        events = 0
        corpus_sum = 0
        playbook_sum = 0
        try:
            with child.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    events += 1
                    corpus_sum += int(rec.get("num_corpus") or 0)
                    playbook_sum += int(rec.get("playbook_char_count") or 0)
                    for cid in rec.get("selected_card_ids") or []:
                        seen_ids.add(str(cid))
        except OSError:
            continue
        out["by_file"][child.name] = {
            "events": events,
            "avg_corpus": (corpus_sum / events) if events else 0.0,
            "avg_playbook_chars": (playbook_sum / events) if events else 0.0,
        }
        total_events += events
        total_corpus += corpus_sum
        total_playbook += playbook_sum
    if total_events:
        out["totals"]["events"] = total_events
        out["totals"]["avg_corpus"] = total_corpus / total_events
        out["totals"]["avg_playbook_chars"] = total_playbook / total_events
    out["totals"]["unique_card_ids"] = len(seen_ids)
    return out


__all__ = [
    "RetrievalEvent",
    "RetrievalLogger",
    "aggregate_logs",
]
