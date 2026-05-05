"""Synthetic generic adapter used to prove the CARGO-v2 core is reusable."""
from __future__ import annotations

from typing import Optional

from ..core import BaseCargoAdapter, CandidateObject
from ..schemas import GateResult, ProposedAction, ToolEffectSchema


class SyntheticGenericAdapter(BaseCargoAdapter):
    name = "synthetic_generic"
    benchmark_name = "synthetic"
    domain_name = "generic"
    id_fields = {"entity_id", "slot_id", "candidate_id"}
    semantic_fields = {"date", "location", "quantity", "color", "size", "preference"}

    def validate_action(self, action: ProposedAction, schema: ToolEffectSchema, wm) -> GateResult:
        # Generic hidden-slot shape used by tests and ACE-style adapters:
        # set_slot(slot_id=..., candidate_id=...).
        if action.name in {"set_slot", "fill_slot", "assign_slot"}:
            cid = str(action.args.get("candidate_id") or action.args.get("item_id") or "").strip()
            if not cid:
                return GateResult.failing("semantic_validation", "missing_candidate_id")
            candidate = wm.task_state.candidate_objects.get(cid)
            if candidate is None:
                return GateResult.failing(
                    "semantic_validation",
                    "unknown_candidate",
                    candidate_id=cid,
                )
            return self.validate_candidate(candidate, wm.task_state)
        return GateResult.passing("semantic_validation", adapter=self.name)

    def validate_write_completeness(self, action: ProposedAction, wm) -> Optional[GateResult]:
        generic = super().validate_write_completeness(action, wm)
        if generic is not None:
            return generic
        if action.name in {"done", "finish_task"} and wm.task_state.unresolved_obligations:
            return GateResult.failing(
                "completeness",
                "synthetic_hidden_slots_unresolved",
                obligations=sorted(wm.task_state.unresolved_obligations),
            )
        return None


__all__ = ["SyntheticGenericAdapter", "CandidateObject"]
