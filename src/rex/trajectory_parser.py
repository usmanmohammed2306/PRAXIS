"""Streaming, schema-tolerant parser for benchmark trajectory records.

Trajectories live as one-JSON-per-line files under ``outputs/<env>_<agent>/``.
Each line is a single record produced by either ``tau_runner`` or
``ace_runner``. The schema varies in small ways across:

  * tau-bench retail / airline records (rich ``info``, OpenAI-style messages)
  * ACEBench records (simpler, may omit ``reward``, has ``actual_tools``)
  * partial / interrupted trajectories (status="error", missing ``reward``)

This parser normalizes everything into ``TrajectorySummary``, which is what
the distiller consumes. It is:

  * **streaming** — never reads the whole file into memory at once
  * **fault tolerant** — bad lines are skipped, not raised
  * **deterministic** — same input → same summaries
  * **safe** — every textual field is sanitized at parse time so the
    distiller never sees raw PII
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from .memory_types import TrajectorySummary
from .sanitize import sanitize_text


# ---------------------------------------------------------------------------
# Low-level streaming
# ---------------------------------------------------------------------------
def iter_jsonl_records(path: os.PathLike) -> Iterator[Dict[str, Any]]:
    """Yield JSON objects from a JSONL file, skipping malformed lines.

    The file is read line-by-line; we never load the whole file into RAM.
    """
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


# ---------------------------------------------------------------------------
# Field extraction helpers
# ---------------------------------------------------------------------------
def _benchmark_for(record: Dict[str, Any]) -> str:
    """Best-effort label of which benchmark produced ``record``."""
    info = record.get("info") or {}
    if isinstance(info, dict):
        bench = info.get("benchmark")
        if isinstance(bench, str) and bench.strip():
            return bench.strip()
    if "tool_coverage" in record or "expected_tools" in record:
        return "ACEBench"
    if "reward" in record or "task_id" in record:
        return "tau-bench"
    return "unknown"


def _environment_for(record: Dict[str, Any]) -> str:
    info = record.get("info") or {}
    if isinstance(info, dict):
        for key in ("env", "environment", "domain"):
            v = info.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip().lower()
    # BFCL records carry expected_tools / tool_coverage at the top level but
    # info["environment"] = "bfcl" is already checked above, so this fallback
    # only fires for genuinely legacy ACE records that lack an info dict.
    if "tool_coverage" in record or "expected_tools" in record:
        # Prefer a top-level "domain" or "environment" key before assuming ACE.
        for key in ("domain", "environment", "env"):
            v = record.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip().lower()
        return "ace"
    return "generic"


def _controller_for(record: Dict[str, Any]) -> str:
    info = record.get("info") or {}
    if isinstance(info, dict):
        for key in ("controller", "agent", "agent_kind", "style"):
            v = info.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip().lower()
    for key in ("controller", "agent"):
        v = record.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip().lower()
    return "unknown"


def _status_for(record: Dict[str, Any]) -> str:
    s = record.get("status")
    if isinstance(s, str) and s.strip():
        return s.strip().lower()
    if record.get("error"):
        return "error"
    return "ok"


def _error_for(record: Dict[str, Any]) -> str:
    err = record.get("error")
    if isinstance(err, str) and err.strip():
        return sanitize_text(err, limit=180)
    info = record.get("info") or {}
    if isinstance(info, dict):
        ie = info.get("error")
        if isinstance(ie, str) and ie.strip():
            return sanitize_text(ie, limit=180)
    return ""


def _reward_for(record: Dict[str, Any]) -> float:
    r = record.get("reward")
    if r is None:
        # ACE records expose tool_coverage instead.
        cov = record.get("tool_coverage")
        if isinstance(cov, (int, float)):
            return float(cov)
        return 0.0
    try:
        return float(r)
    except (TypeError, ValueError):
        return 0.0


def _messages_for(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    msgs = record.get("messages") or record.get("traj") or []
    if not isinstance(msgs, list):
        return []
    return [m for m in msgs if isinstance(m, dict)]


def _user_first(messages: Sequence[Dict[str, Any]]) -> str:
    for m in messages:
        if m.get("role") == "user" and m.get("content"):
            return str(m.get("content"))
    return ""


def _user_last(messages: Sequence[Dict[str, Any]]) -> str:
    last = ""
    for m in messages:
        if m.get("role") == "user" and m.get("content"):
            last = str(m.get("content"))
    return last


def _tool_calls(messages: Sequence[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") if isinstance(tc, dict) else None
            if isinstance(fn, dict) and fn.get("name"):
                out.append(str(fn["name"]))
    return out


def _tool_args_redacted(messages: Sequence[Dict[str, Any]], limit: int = 80) -> List[str]:
    """Return a list of *redacted* JSON strings for each tool call's arguments.

    Used to detect "we passed the same argument twice / kept retrying"
    patterns during distillation. The strings are sanitized so they never
    expose raw IDs.
    """
    out: List[str] = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") if isinstance(tc, dict) else None
            if not isinstance(fn, dict):
                continue
            args_raw = fn.get("arguments") or "{}"
            try:
                if isinstance(args_raw, str):
                    parsed = json.loads(args_raw)
                else:
                    parsed = args_raw
            except json.JSONDecodeError:
                parsed = {"_raw": str(args_raw)[:200]}
            if not isinstance(parsed, dict):
                parsed = {"_value": parsed}
            try:
                rendered = json.dumps(parsed, sort_keys=True, default=str)
            except (TypeError, ValueError):
                rendered = str(parsed)
            out.append(sanitize_text(rendered, limit=limit))
    return out


def _tool_observations_redacted(messages: Sequence[Dict[str, Any]], limit: int = 200) -> List[str]:
    out: List[str] = []
    for m in messages:
        if m.get("role") != "tool":
            continue
        content = m.get("content") or ""
        out.append(sanitize_text(content, limit=limit))
    return out


def _count_role(messages: Sequence[Dict[str, Any]], role: str) -> int:
    return sum(1 for m in messages if m.get("role") == role)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def parse_record(record: Dict[str, Any]) -> Optional[TrajectorySummary]:
    """Convert a single trajectory dict into a ``TrajectorySummary``.

    Returns ``None`` if ``record`` is so malformed that no useful summary
    can be produced (no messages and no tool calls).
    """
    if not isinstance(record, dict):
        return None

    messages = _messages_for(record)
    tool_calls = _tool_calls(messages)
    if not messages and not tool_calls and not record.get("actual_tools"):
        # ACE records sometimes carry only ``actual_tools`` — treat that as
        # at least one parseable signal so we don't drop them.
        return None

    # ACE-style records may have actual_tools without proper assistant messages.
    if not tool_calls and isinstance(record.get("actual_tools"), list):
        tool_calls = [str(t) for t in record["actual_tools"] if isinstance(t, (str, int))]

    initial = _user_first(messages)
    if not initial:
        # ACE tasks store the initial user request in the record itself.
        for k in ("initial_user", "user_turn", "question", "instruction"):
            v = record.get(k)
            if isinstance(v, str) and v.strip():
                initial = v
                break

    last_user = _user_last(messages)

    summary = TrajectorySummary(
        task_id=str(record.get("task_id") if record.get("task_id") is not None else record.get("index", "0")),
        trial=int(record.get("trial") or 0),
        benchmark=_benchmark_for(record),
        environment=_environment_for(record),
        controller=_controller_for(record),
        reward=_reward_for(record),
        status=_status_for(record),
        error=_error_for(record),
        initial_user=sanitize_text(initial, limit=360),
        last_user=sanitize_text(last_user, limit=360),
        tool_calls=tool_calls,
        tool_arguments_redacted=_tool_args_redacted(messages),
        tool_observations_redacted=_tool_observations_redacted(messages),
        assistant_messages=_count_role(messages, "assistant"),
        user_messages=_count_role(messages, "user"),
        tool_messages=_count_role(messages, "tool"),
        info=record.get("info") if isinstance(record.get("info"), dict) else {},
    )
    return summary


def iter_summaries(records: Iterable[Dict[str, Any]]) -> Iterator[TrajectorySummary]:
    """Map an iterable of raw records to ``TrajectorySummary`` objects.

    Errors during individual records are swallowed (the bad records are
    simply not yielded). This keeps streaming pipelines robust against
    upstream corruption.
    """
    for r in records:
        try:
            s = parse_record(r)
        except Exception:  # noqa: BLE001 — defensive: never fail the stream
            continue
        if s is not None:
            yield s


def parse_jsonl_path(path: os.PathLike) -> Iterator[TrajectorySummary]:
    """Convenience: stream-parse a ``trajectories.jsonl`` file."""
    yield from iter_summaries(iter_jsonl_records(path))


__all__ = [
    "iter_jsonl_records",
    "parse_record",
    "iter_summaries",
    "parse_jsonl_path",
]
