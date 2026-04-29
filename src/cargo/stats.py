"""Per-trajectory diagnostics aggregator for CARGO.

Lives in its own module so the ACEBench loop and the tau-bench agent
can both reuse it without dragging the tau-bench / OpenAI imports into
each other's import paths.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .schemas import GateResult


class CargoStats:
    """Per-trajectory aggregate diagnostics surfaced through ``info.cargo_stats``."""

    def __init__(self) -> None:
        self.steps_total = 0
        self.steps_fast_path = 0
        self.steps_gated = 0
        self.gate_runs: Dict[str, int] = {}
        self.gate_fails: Dict[str, int] = {}
        self.abstain_total = 0
        self.repair_retry = 0
        self.repair_ask_user = 0
        self.repair_finalize = 0
        self.json_parse_failures = 0
        self.actions_executed = 0
        self.executed_by_class: Dict[str, int] = {}
        self.per_step: List[Dict[str, Any]] = []

    def record_gate(self, result: GateResult) -> None:
        self.gate_runs[result.gate] = self.gate_runs.get(result.gate, 0) + 1
        if not result.ok:
            self.gate_fails[result.gate] = self.gate_fails.get(result.gate, 0) + 1

    def record_step(self, info: Dict[str, Any]) -> None:
        # Cap per_step length so info doesn't grow unbounded.
        if len(self.per_step) < 200:
            self.per_step.append(info)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "steps_total": self.steps_total,
            "steps_fast_path": self.steps_fast_path,
            "steps_gated": self.steps_gated,
            "gate_runs": dict(self.gate_runs),
            "gate_fails": dict(self.gate_fails),
            "abstain_total": self.abstain_total,
            "repair_retry": self.repair_retry,
            "repair_ask_user": self.repair_ask_user,
            "repair_finalize": self.repair_finalize,
            "json_parse_failures": self.json_parse_failures,
            "actions_executed": self.actions_executed,
            "executed_by_class": dict(self.executed_by_class),
            "per_step": list(self.per_step),
        }


__all__ = ["CargoStats"]
