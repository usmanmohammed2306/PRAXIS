"""Repeat-loop gate: reject the same (name, canonicalized args) appearing
within a short rolling window. This is the cheapest defence against a
common ReAct failure mode where the model gets stuck retrying the same
no-op tool call.
"""
from __future__ import annotations

from ..schemas import GateResult, ProposedAction, ToolEffectSchema
from ..working_memory import WorkingMemory


def check_repeat_loop(
    action: ProposedAction,
    schema: ToolEffectSchema,
    wm: WorkingMemory,
) -> GateResult:
    sig = action.signature()
    if not sig:
        return GateResult.passing("repeat_loop")
    if sig in wm.recent_signatures:
        # Multiple READ calls with identical args are usually pointless;
        # multiple identical mutations are dangerous. Either way reject.
        return GateResult.failing(
            "repeat_loop",
            "repeated_action_signature",
            signature=sig,
        )
    return GateResult.passing("repeat_loop", signature=sig)


__all__ = ["check_repeat_loop"]
