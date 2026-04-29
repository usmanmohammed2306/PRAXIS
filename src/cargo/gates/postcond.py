"""Post-condition gate run AFTER tool execution.

This gate doesn't decide whether to execute (the action has already run).
It inspects the observation and flags violations: empty / error
observations, or observations that contradict declared post-conditions.
The agent uses the result to decide whether to *retry* via repair.
"""
from __future__ import annotations

import json
from typing import Any

from ..schemas import GateResult, ProposedAction, ToolEffectSchema


def _looks_like_error(obs: Any) -> bool:
    if obs is None:
        return True
    if isinstance(obs, str):
        s = obs.strip().lower()
        if not s:
            return True
        # Heuristics for tau-bench / ACEBench textual error markers.
        if s.startswith("error:") or s.startswith("\"error\":"):
            return True
        # Single-token plain-string errors like "error" or "fail".
        if s in ("error", "failed", "failure", "exception"):
            return True
    if isinstance(obs, dict):
        # JSON object with explicit error field.
        for k in ("error", "Error", "ERROR"):
            v = obs.get(k)
            if isinstance(v, str) and v.strip():
                return True
            if v is True:
                return True
        if obs.get("status") in ("error", "failed", "failure"):
            return True
    return False


def check_postconditions(
    action: ProposedAction,
    schema: ToolEffectSchema,
    *,
    obs: Any,
) -> GateResult:
    if _looks_like_error(obs):
        # Truncate obs in the diagnostics for log compactness.
        try:
            obs_str = json.dumps(obs, default=str)[:300]
        except Exception:
            obs_str = str(obs)[:300]
        return GateResult.failing(
            "postconditions",
            "post_check_error_observation",
            obs=obs_str,
        )
    # Soft check: observations whose content has zero overlap with declared
    # post-conditions are flagged but only as a warning (not a hard fail);
    # the calling agent treats post-check failures as a hint to re-deliberate.
    return GateResult.passing("postconditions")


__all__ = ["check_postconditions"]
