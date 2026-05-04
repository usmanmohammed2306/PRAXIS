"""Argument-grounding gate: ID-typed argument values must be present in
the agent's evidence (user_facts ∪ db_facts ∪ str(last_obs)).

This single deterministic check kills a large fraction of τ-bench
failures where the model invents an order_id / user_id / item_id and
mutates against it. Cost: zero LLM calls.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, List, Tuple

from ..schemas import GateResult, ProposedAction, ToolEffectSchema
from ..working_memory import WorkingMemory


# Heuristic: a value "looks like an opaque ID" if it matches one of these.
_ID_PATTERNS = [
    re.compile(r"^[A-Z]{1,3}\d{2,}$"),               # O1234, R567, USR12
    re.compile(r"^\d{4,}$"),                         # 123456
    re.compile(r"^[A-Z0-9][A-Z0-9_\-]{3,}$"),         # ABC-1234
    re.compile(r"^[a-z]+_[a-z]+_\d{1,8}$"),           # alex_smith_42
    re.compile(r"^[\w.+\-]+@[\w\-.]+\.[A-Za-z]{2,}$"),  # email
]


def _looks_like_id(value: str) -> bool:
    if not isinstance(value, str):
        return False
    v = value.strip()
    if len(v) < 3 or len(v) > 80:
        return False
    return any(p.match(v) for p in _ID_PATTERNS)


def _grounded_in_evidence(value: str, evidence: str) -> bool:
    if not value:
        return False
    return value in evidence


def _field_key(path: str) -> str:
    key = str(path or "").split(".")[-1]
    key = key.split("[")[0]
    return key


def _iter_arg_values(name: str, value: Any) -> Iterable[Tuple[str, str]]:
    """Yield scalar argument values with their logical argument path."""
    if value is None:
        return
    if isinstance(value, list):
        for i, item in enumerate(value):
            yield from _iter_arg_values(f"{name}[{i}]", item)
        return
    if isinstance(value, dict):
        for k, item in value.items():
            yield from _iter_arg_values(f"{name}.{k}", item)
        return
    yield name, value if isinstance(value, str) else str(value)


def check_arg_grounding(
    action: ProposedAction,
    schema: ToolEffectSchema,
    wm: WorkingMemory,
) -> GateResult:
    evidence = wm.all_evidence()
    last_obs_text = ""
    if wm.last_obs not in (None, ""):
        try:
            import json as _json
            last_obs_text = _json.dumps(wm.last_obs, default=str)
        except Exception:
            last_obs_text = str(wm.last_obs)
    full_evidence = evidence + "\n" + last_obs_text

    ungrounded: List[str] = []

    # Per-schema explicit ID fields are checked unconditionally.
    id_fields = list(schema.arg_id_fields or [])

    for k, v in action.args.items():
        for path, v_str in _iter_arg_values(k, v):
            base = _field_key(path)
            forced_id = k in id_fields or base in id_fields
            typed_values = wm.typed_evidence_for(base) if forced_id else []
            if typed_values:
                if v_str not in typed_values:
                    ungrounded.append(f"{path}={v_str}")
                continue
            if not (forced_id or _looks_like_id(v_str)):
                continue
            if not _grounded_in_evidence(v_str, full_evidence):
                ungrounded.append(f"{path}={v_str}")

    if ungrounded:
        return GateResult.failing(
            "arg_grounding",
            f"ungrounded_id_values:{ungrounded[:3]}",
            ungrounded=ungrounded[:3],
        )
    return GateResult.passing("arg_grounding")


__all__ = ["check_arg_grounding"]
