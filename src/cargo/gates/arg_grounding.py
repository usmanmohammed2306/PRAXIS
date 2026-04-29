"""Argument-grounding gate: ID-typed argument values must be present in
the agent's evidence (user_facts ∪ db_facts ∪ str(last_obs)).

This single deterministic check kills a large fraction of τ-bench
failures where the model invents an order_id / user_id / item_id and
mutates against it. Cost: zero LLM calls.
"""
from __future__ import annotations

import re
from typing import Iterable, List

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
        if v is None:
            continue
        v_str = v if isinstance(v, str) else str(v)
        forced_id = k in id_fields
        if forced_id or _looks_like_id(v_str):
            if not _grounded_in_evidence(v_str, full_evidence):
                ungrounded.append(f"{k}={v_str}")

    if ungrounded:
        return GateResult.failing(
            "arg_grounding",
            f"ungrounded_id_values:{ungrounded[:3]}",
            ungrounded=ungrounded[:3],
        )
    return GateResult.passing("arg_grounding")


__all__ = ["check_arg_grounding"]
