"""ACEBench adapter.

The adapter models ACEBench/AgentCE-style tasks as hidden slots with local
and global constraints.  It rejects candidates that satisfy shallow local
syntax but violate global constraints, which is the decoy pattern these
benchmarks are designed to expose.
"""
from __future__ import annotations

from typing import Optional

from ..core import CandidateObject
from ..schemas import GateResult, ProposedAction, ToolEffectSchema
from .synthetic_generic import SyntheticGenericAdapter


class ACEBenchAdapter(SyntheticGenericAdapter):
    name = "acebench"
    benchmark_name = "ACEBench"
    domain_name = "agent"
    id_fields = {
        "slot_id", "candidate_id", "api_id", "tool_id", "item_id",
    }
    semantic_fields = {
        "category", "constraint", "attribute", "value", "language",
    }

    def validate_action(self, action: ProposedAction, schema: ToolEffectSchema, wm) -> GateResult:
        if action.name in {"set_slot", "fill_slot", "assign_slot"}:
            cid = str(action.args.get("candidate_id") or action.args.get("item_id") or "").strip()
            candidate = wm.task_state.candidate_objects.get(cid)
            if candidate is None:
                return GateResult.failing("semantic_validation", "unknown_candidate", candidate_id=cid)
            base = self.validate_candidate(candidate, wm.task_state)
            if not base.ok:
                return base
            if candidate.attributes.get("global_valid") is False:
                return GateResult.failing(
                    "semantic_validation",
                    "candidate_violates_global_constraints",
                    candidate_id=cid,
                )
        return GateResult.passing("semantic_validation", adapter=self.name)

    def validate_write_completeness(self, action: ProposedAction, wm) -> Optional[GateResult]:
        if action.name in {"done", "finish", "finalize"} and wm.task_state.unresolved_obligations:
            return GateResult.failing(
                "completeness",
                "ace_hidden_slots_unresolved",
                obligations=sorted(wm.task_state.unresolved_obligations),
            )
        return super().validate_write_completeness(action, wm)


__all__ = ["ACEBenchAdapter", "CandidateObject"]
