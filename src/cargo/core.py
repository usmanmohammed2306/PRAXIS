"""Generic CARGO-v4 state, adapter, validation, and decision kernel.

This module is intentionally domain-neutral.  It knows about facts, slots,
constraints, preferences, obligations, candidates, actions, and gates; all
domain and benchmark semantics live in ``src.cargo.adapters``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Set, Tuple, TYPE_CHECKING

from .risk_class import RiskClass
from .schemas import GateResult, ProposedAction, ToolEffectSchema

if TYPE_CHECKING:  # pragma: no cover - type-only import avoids circularity.
    from .working_memory import WorkingMemory


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")


def normalize_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    synonyms = {
        "no backlight": "none",
        "without backlight": "none",
        "no lights": "none",
        "no light": "none",
        "full-size": "full size",
        "fullsize": "full size",
        "google home": "google",
        "google assistant": "google",
    }
    return synonyms.get(text, text)


@dataclass
class StateFact:
    key: str
    value: Any
    source: str = "user"  # user | tool | inferred | assumption | policy
    confidence: float = 1.0
    confirmed: bool = False
    turn: int = 0


@dataclass
class Constraint:
    slot: str
    op: str
    value: Any
    source: str = "user"
    hard: bool = True
    scope: str = "local"

    def satisfied_by(self, attrs: Mapping[str, Any]) -> bool:
        actual = _lookup_attr(attrs, self.slot)
        if actual is None:
            return False
        return _compare(actual, self.op, self.value)


@dataclass
class Preference:
    slot: str
    value: Any
    rank: int = 0
    source: str = "user"


@dataclass
class FallbackRule:
    slot: str
    from_value: Any
    to_value: Any
    condition: str = "strict_unavailable"
    source: str = "user"


@dataclass
class CandidateObject:
    candidate_id: str
    object_type: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    available: bool = True


@dataclass
class CandidateSet:
    """Stored result set from an information-gathering action."""

    set_id: str
    source_tool: str
    query_args: Dict[str, Any] = field(default_factory=dict)
    candidates: List[CandidateObject] = field(default_factory=list)
    empty: bool = False
    exhausted: bool = False
    selected_candidate_id: str = ""
    rejected_candidates: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CandidateSelection:
    """Result of deterministic candidate selection."""

    ok: bool
    candidate: Optional[CandidateObject] = None
    reason: str = ""
    rejected: List[Dict[str, Any]] = field(default_factory=list)
    fallback_used: Optional[FallbackRule] = None


@dataclass
class GoalHypothesis:
    """A compact live branch in the current task frame.

    This is intentionally small.  It is not a plan tree; it is the tiny set of
    plausible task interpretations the router can keep warm while evidence is
    still arriving.
    """

    hypothesis_id: str
    label: str
    confidence: float = 0.5
    anchors: Dict[str, Any] = field(default_factory=dict)
    friction: float = 0.0
    last_evidence_turn: int = 0
    status: str = "live"  # live | downweighted | quarantined

    def score(self) -> float:
        return max(0.0, float(self.confidence) - float(self.friction))


@dataclass
class GoalField:
    """Soft task-continuity state used by the goal-field router."""

    active_goal: str = ""
    active_stage: str = "start"
    confirmed_facts: Dict[str, Any] = field(default_factory=dict)
    unresolved_slots: Set[str] = field(default_factory=set)
    candidate_actions: Dict[str, float] = field(default_factory=dict)
    hard_constraints: List[str] = field(default_factory=list)
    soft_preferences: List[str] = field(default_factory=list)
    recent_failures: Dict[str, int] = field(default_factory=dict)
    friction: Dict[str, float] = field(default_factory=dict)
    momentum: float = 0.0
    progress_events: List[str] = field(default_factory=list)
    hypotheses: List[GoalHypothesis] = field(default_factory=list)
    active_task_frame: Dict[str, Any] = field(default_factory=dict)
    recent_recenter_reason: str = ""
    last_selected_signature: str = ""
    last_decision: Dict[str, Any] = field(default_factory=dict)
    turn: int = 0

    def sync_from_memory(self, wm: "WorkingMemory") -> None:
        if not self.active_goal:
            self.active_goal = str(getattr(wm, "goal", "") or "")[:220]
        state = getattr(wm, "task_state", None)
        if state is not None:
            self.unresolved_slots = set(getattr(state, "open_slots", set()) or set())
            self.unresolved_slots.update(getattr(state, "unresolved_obligations", set()) or set())
            facts: Dict[str, Any] = {}
            for key, fact in list((getattr(state, "db_confirmed_facts", {}) or {}).items())[-10:]:
                facts[str(key)] = getattr(fact, "value", fact)
            self.confirmed_facts = facts
            self.hard_constraints = [
                f"{c.slot}{c.op}{normalize_value(c.value)}"
                for c in list(getattr(state, "constraints", []) or [])
                if getattr(c, "hard", False)
            ][:8]
            self.soft_preferences = [
                f"{p.slot}={normalize_value(p.value)}"
                for p in list(getattr(state, "preferences", []) or [])
            ][:8]
        frame: Dict[str, Any] = {}
        for key, value in (getattr(wm, "semantic_slots", {}) or {}).items():
            if key in (getattr(wm, "user_bound_slots", {}) or {}) or key in {
                "intents", "intent", "origin", "destination", "date", "cabin",
                "baggage_count", "travel_insurance", "payment_preferences",
            }:
                frame[str(key)] = value
        self.active_task_frame = frame
        self._ensure_hypotheses(wm)
        self._trim()

    def record_progress(self, label: str, amount: float = 0.4) -> None:
        self.momentum = min(12.0, max(0.0, self.momentum + float(amount)))
        clean = str(label or "progress")[:80]
        self.progress_events.append(clean)
        if len(self.progress_events) > 8:
            self.progress_events = self.progress_events[-8:]

    def record_friction(self, signature: str, amount: float = 1.0, reason: str = "") -> None:
        sig = str(signature or reason or "unknown")[:180]
        if not sig:
            return
        self.friction[sig] = min(12.0, self.friction.get(sig, 0.0) + float(amount))
        self.recent_failures[sig] = self.recent_failures.get(sig, 0) + 1
        self.momentum = max(0.0, self.momentum - (0.2 * float(amount)))
        if reason:
            self.recent_recenter_reason = str(reason)[:120]
        self._trim()

    def recenter(self, reason: str, *, preserve_goal: bool = True) -> None:
        self.recent_recenter_reason = str(reason or "recentered")[:120]
        for hyp in self.hypotheses:
            if hyp.status == "live":
                hyp.status = "downweighted"
                hyp.friction += 0.5
        if preserve_goal:
            self.record_progress("recenter_preserved_goal", 0.2)
        self._trim()

    def add_hypothesis(self, hypothesis: GoalHypothesis) -> None:
        if not hypothesis.hypothesis_id:
            hypothesis.hypothesis_id = normalize_key(hypothesis.label) or f"hyp_{len(self.hypotheses)}"
        for cur in self.hypotheses:
            if cur.hypothesis_id == hypothesis.hypothesis_id:
                cur.confidence = max(cur.confidence, hypothesis.confidence)
                cur.anchors.update(hypothesis.anchors)
                cur.last_evidence_turn = max(cur.last_evidence_turn, hypothesis.last_evidence_turn)
                if cur.status == "quarantined" and hypothesis.status == "live":
                    cur.status = "downweighted"
                self._trim()
                return
        self.hypotheses.append(hypothesis)
        self._trim()

    def render_compact(self, max_chars: int = 360) -> str:
        parts: List[str] = []
        parts.append(f"stage={self.active_stage}")
        parts.append(f"momentum={self.momentum:.1f}")
        if self.unresolved_slots:
            parts.append("open=" + ",".join(sorted(self.unresolved_slots))[:90])
        live = sorted(self.hypotheses, key=lambda h: h.score(), reverse=True)[:3]
        if live:
            parts.append(
                "hyp="
                + ";".join(
                    f"{h.label[:28]}:{h.score():.1f}/{h.status[0]}"
                    for h in live
                )
            )
        top_friction = sorted(self.friction.items(), key=lambda kv: kv[1], reverse=True)[:3]
        if top_friction:
            parts.append(
                "friction="
                + ";".join(f"{k.split('(')[0][:28]}:{v:.1f}" for k, v in top_friction)
            )
        if self.recent_recenter_reason:
            parts.append(f"recenter={self.recent_recenter_reason[:80]}")
        return " | ".join(p for p in parts if p)[:max_chars]

    def _ensure_hypotheses(self, wm: "WorkingMemory") -> None:
        intents = (getattr(wm, "semantic_slots", {}) or {}).get("intents") or []
        if not isinstance(intents, list):
            intents = [intents]
        for intent in intents[-3:]:
            label = str(intent or "").strip()
            if not label:
                continue
            self.add_hypothesis(GoalHypothesis(
                hypothesis_id=normalize_key(label),
                label=label,
                confidence=0.7,
                anchors={"source": "semantic_slots"},
                last_evidence_turn=self.turn,
            ))

    def _trim(self) -> None:
        self.hypotheses = sorted(
            self.hypotheses,
            key=lambda h: (h.status == "live", h.score(), h.last_evidence_turn),
            reverse=True,
        )[:3]
        if len(self.friction) > 24:
            keep = dict(sorted(self.friction.items(), key=lambda kv: kv[1], reverse=True)[:24])
            self.friction = keep
        if len(self.recent_failures) > 24:
            keep_keys = set(self.friction.keys())
            self.recent_failures = {
                k: v for k, v in self.recent_failures.items()
                if k in keep_keys
            }


@dataclass
class GoalActionCandidate:
    """A proposed next action plus soft progress metadata."""

    action: ProposedAction
    source: str = "model"
    progress: float = 0.0
    uncertainty_reduction: float = 0.0
    completion: float = 0.0
    frame_match: float = 0.0
    rationale: str = ""
    score: float = 0.0

    def signature(self) -> str:
        return self.action.signature()


@dataclass
class GoalFieldDecision:
    selected: GoalActionCandidate
    alternatives: List[GoalActionCandidate] = field(default_factory=list)
    reason: str = ""


@dataclass
class PhaseDecision:
    """Compact controller phase decision.

    Adapters name domain stages; the generic core maps them into broad control
    phases so READs stay permissive while WRITEs have to pass a commitment
    boundary.
    """

    phase: str = "DISCOVER"
    stage: str = ""
    reason: str = ""


@dataclass
class PreCommitVerdict:
    """Result of the cheap deterministic verifier for mutating actions."""

    ok: bool
    reason: str = ""
    missing: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_gate(self) -> GateResult:
        if self.ok:
            return GateResult.passing(
                "precommit_verifier",
                missing=list(self.missing),
                **self.diagnostics,
            )
        return GateResult.failing(
            "precommit_verifier",
            self.reason or "precommit_verifier_failed",
            missing=list(self.missing),
            **self.diagnostics,
        )


@dataclass
class ProofObligation:
    """Machine-checkable proof item for a proposed task transition."""

    name: str
    ok: bool
    reason: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "reason": self.reason,
            "evidence": dict(self.evidence),
        }


@dataclass
class CommitCertificate:
    """Proof-carrying certificate for WRITE/IRREVERSIBLE/FINAL actions.

    The core owns the neutral container and verifier.  Adapters own the
    domain-specific obligations that prove selected candidates and task-frame
    transitions are correct.
    """

    action_name: str
    action_signature: str = ""
    certificate_type: str = "generic"
    obligations: List[ProofObligation] = field(default_factory=list)
    selected_candidate_ids: List[str] = field(default_factory=list)
    rejected_candidates: List[Dict[str, Any]] = field(default_factory=list)
    ok: bool = True
    reason: str = ""

    def require(self, name: str, ok: bool, reason: str = "", **evidence: Any) -> bool:
        obligation = ProofObligation(
            name=str(name),
            ok=bool(ok),
            reason="ok" if ok else str(reason or "failed"),
            evidence=dict(evidence),
        )
        self.obligations.append(obligation)
        self.finalize()
        return obligation.ok

    def finalize(self) -> "CommitCertificate":
        failed = [o for o in self.obligations if not o.ok]
        self.ok = not failed
        self.reason = failed[0].reason if failed else "ok"
        return self

    def to_dict(self) -> Dict[str, Any]:
        self.finalize()
        return {
            "action_name": self.action_name,
            "action_signature": self.action_signature,
            "certificate_type": self.certificate_type,
            "ok": self.ok,
            "reason": self.reason,
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "rejected_candidates": list(self.rejected_candidates),
            "obligations": [o.to_dict() for o in self.obligations],
        }


@dataclass
class Obligation:
    """Domain-neutral unit of task closure.

    Obligations describe what must become true before a task can commit.  The
    core owns the shape; adapters decide which obligations exist for a domain.
    """

    obligation_id: str
    operation_type: str
    required_slots: Set[str] = field(default_factory=set)
    required_constraints: Set[str] = field(default_factory=set)
    candidate_retrieval_needs: Set[str] = field(default_factory=set)
    selected_candidate_needs: Set[str] = field(default_factory=set)
    confirmation_required: bool = False
    execution_required: bool = True
    status: str = "open"  # open | blocked | complete
    blockers: List[str] = field(default_factory=list)

    def open_slots(self, state: "TaskState") -> Set[str]:
        return {
            slot for slot in self.required_slots
            if state.fact_value(slot) in (None, "") and slot not in state.confirmations
        }

    @property
    def complete(self) -> bool:
        return self.status == "complete"


@dataclass
class TaskState:
    benchmark_name: str = ""
    domain_name: str = ""
    conversation_id: str = ""
    user_identity: Dict[str, Any] = field(default_factory=dict)
    authenticated_status: str = "unknown"
    intent: List[str] = field(default_factory=list)
    requested_operations: List[str] = field(default_factory=list)
    required_slots: Set[str] = field(default_factory=set)
    optional_slots: Set[str] = field(default_factory=set)
    user_provided_facts: Dict[str, StateFact] = field(default_factory=dict)
    db_confirmed_facts: Dict[str, StateFact] = field(default_factory=dict)
    inferred_facts: Dict[str, StateFact] = field(default_factory=dict)
    assumptions: Dict[str, StateFact] = field(default_factory=dict)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    open_slots: Set[str] = field(default_factory=set)
    constraints: List[Constraint] = field(default_factory=list)
    preferences: List[Preference] = field(default_factory=list)
    fallback_rules: List[FallbackRule] = field(default_factory=list)
    candidate_sets: Dict[str, CandidateSet] = field(default_factory=dict)
    candidate_objects: Dict[str, CandidateObject] = field(default_factory=dict)
    selected_objects: Dict[str, str] = field(default_factory=dict)
    obligations: Dict[str, Obligation] = field(default_factory=dict)
    unresolved_obligations: Set[str] = field(default_factory=set)
    confirmations: Set[str] = field(default_factory=set)
    executed_writes: Set[str] = field(default_factory=set)
    completed_operations: Set[str] = field(default_factory=set)
    failed_action_signatures: Dict[str, int] = field(default_factory=dict)
    last_error: str = ""
    last_meaningful_state_change: int = 0
    terminal_status: str = ""

    def bind_fact(
        self,
        key: str,
        value: Any,
        *,
        source: str,
        confirmed: bool = False,
        confidence: float = 1.0,
    ) -> bool:
        nkey = normalize_key(key)
        if not nkey or value in (None, ""):
            return False
        fact = StateFact(
            key=nkey,
            value=value,
            source=source,
            confirmed=bool(confirmed),
            confidence=float(confidence),
            turn=self.last_meaningful_state_change + 1,
        )
        target = self._store_for(source, confirmed)
        existing = self.get_fact(nkey)
        if existing is not None:
            if normalize_value(existing.value) == normalize_value(value):
                if confirmed and not existing.confirmed:
                    self.db_confirmed_facts[nkey] = fact
                    self.last_meaningful_state_change += 1
                    return True
                return False
            if existing.confirmed and not confirmed:
                self.conflicts.append({
                    "key": nkey,
                    "kept": existing.value,
                    "rejected": value,
                    "reason": "confirmed_fact_outranks_weaker_claim",
                })
                return False
            self.conflicts.append({
                "key": nkey,
                "old": existing.value,
                "new": value,
                "reason": "stronger_or_later_fact_replaced_prior",
            })
        target[nkey] = fact
        if nkey == "intent" and str(value) not in self.intent:
            self.intent.append(str(value))
        if nkey == "confirmation":
            self.confirmations.add(normalize_value(value))
        self.last_meaningful_state_change += 1
        return True

    def _store_for(self, source: str, confirmed: bool) -> Dict[str, StateFact]:
        if confirmed or source == "tool":
            return self.db_confirmed_facts
        if source == "inferred":
            return self.inferred_facts
        if source == "assumption":
            return self.assumptions
        return self.user_provided_facts

    def get_fact(self, key: str) -> Optional[StateFact]:
        nkey = normalize_key(key)
        return (
            self.db_confirmed_facts.get(nkey)
            or self.user_provided_facts.get(nkey)
            or self.inferred_facts.get(nkey)
            or self.assumptions.get(nkey)
        )

    def fact_value(self, key: str, default: Any = None) -> Any:
        fact = self.get_fact(key)
        return fact.value if fact else default

    def add_constraint(self, constraint: Constraint) -> bool:
        sig = (
            normalize_key(constraint.slot),
            constraint.op,
            normalize_value(constraint.value),
            constraint.hard,
            constraint.scope,
        )
        for cur in self.constraints:
            cur_sig = (
                normalize_key(cur.slot),
                cur.op,
                normalize_value(cur.value),
                cur.hard,
                cur.scope,
            )
            if cur_sig == sig:
                return False
        constraint.slot = normalize_key(constraint.slot)
        self.constraints.append(constraint)
        self.last_meaningful_state_change += 1
        return True

    def add_preference(self, preference: Preference) -> bool:
        sig = (normalize_key(preference.slot), normalize_value(preference.value))
        if any((normalize_key(p.slot), normalize_value(p.value)) == sig for p in self.preferences):
            return False
        preference.slot = normalize_key(preference.slot)
        self.preferences.append(preference)
        self.last_meaningful_state_change += 1
        return True

    def add_fallback(self, rule: FallbackRule) -> bool:
        sig = (
            normalize_key(rule.slot),
            normalize_value(rule.from_value),
            normalize_value(rule.to_value),
            rule.condition,
        )
        for cur in self.fallback_rules:
            cur_sig = (
                normalize_key(cur.slot),
                normalize_value(cur.from_value),
                normalize_value(cur.to_value),
                cur.condition,
            )
            if cur_sig == sig:
                return False
        rule.slot = normalize_key(rule.slot)
        self.fallback_rules.append(rule)
        self.last_meaningful_state_change += 1
        return True

    def add_candidate(self, candidate: CandidateObject) -> bool:
        cid = str(candidate.candidate_id or "").strip()
        if not cid:
            return False
        cur = self.candidate_objects.get(cid)
        if cur and cur.attributes == candidate.attributes and cur.available == candidate.available:
            return False
        self.candidate_objects[cid] = candidate
        self.last_meaningful_state_change += 1
        return True

    def record_candidate_set(
        self,
        source_tool: str,
        query_args: Mapping[str, Any],
        candidates: List[CandidateObject],
    ) -> CandidateSet:
        set_id = action_signature_key(source_tool, query_args)
        cur = self.candidate_sets.get(set_id)
        empty = not candidates
        if cur is None:
            cur = CandidateSet(
                set_id=set_id,
                source_tool=source_tool,
                query_args=dict(query_args),
                empty=empty,
                exhausted=empty,
            )
            self.candidate_sets[set_id] = cur
        cur.candidates = list(candidates)
        cur.empty = empty
        cur.exhausted = empty
        for candidate in candidates:
            self.add_candidate(candidate)
        self.last_meaningful_state_change += 1
        return cur

    def candidate_set_for(self, source_tool: str, query_args: Mapping[str, Any]) -> Optional[CandidateSet]:
        return self.candidate_sets.get(action_signature_key(source_tool, query_args))

    def upsert_obligation(self, obligation: Obligation) -> bool:
        oid = normalize_key(obligation.obligation_id)
        if not oid:
            return False
        obligation.obligation_id = oid
        existing = self.obligations.get(oid)
        if existing == obligation:
            return False
        self.obligations[oid] = obligation
        if not obligation.complete:
            self.unresolved_obligations.add(oid)
        else:
            self.unresolved_obligations.discard(oid)
            self.completed_operations.add(obligation.operation_type)
        self.refresh_open_slots()
        self.last_meaningful_state_change += 1
        return True

    def mark_obligation_complete(self, obligation_id: str) -> bool:
        oid = normalize_key(obligation_id)
        ob = self.obligations.get(oid)
        if not ob or ob.status == "complete":
            return False
        ob.status = "complete"
        ob.blockers = []
        self.unresolved_obligations.discard(oid)
        self.completed_operations.add(ob.operation_type)
        self.refresh_open_slots()
        self.last_meaningful_state_change += 1
        return True

    def refresh_open_slots(self) -> Set[str]:
        open_slots: Set[str] = set()
        for obligation in self.obligations.values():
            if obligation.complete:
                continue
            open_slots.update(obligation.open_slots(self))
        self.open_slots = open_slots
        return open_slots

    def hard_constraints_for(self, scope: Optional[str] = None) -> List[Constraint]:
        out = [c for c in self.constraints if c.hard]
        if scope:
            out = [c for c in out if c.scope == scope]
        return out


class CargoDomainAdapter(Protocol):
    name: str
    benchmark_name: str
    domain_name: str
    id_fields: Set[str]
    semantic_fields: Set[str]

    def enrich_schema(self, schema: ToolEffectSchema) -> ToolEffectSchema:
        ...

    def bind_user_message(self, text: str, state: TaskState) -> List[Tuple[str, Any, bool]]:
        ...

    def absorb_observation(self, obs: Any, state: TaskState, action_name: str = "") -> List[Tuple[str, Any, bool]]:
        ...

    def update_obligations(self, state: TaskState, wm: "WorkingMemory") -> None:
        ...

    def validate_read_action(self, action: ProposedAction, schema: ToolEffectSchema, wm: "WorkingMemory") -> GateResult:
        ...

    def validate_ask_user(self, action: ProposedAction, wm: "WorkingMemory") -> GateResult:
        ...

    def validate_action(self, action: ProposedAction, schema: ToolEffectSchema, wm: "WorkingMemory") -> GateResult:
        ...

    def validate_write_completeness(self, action: ProposedAction, wm: "WorkingMemory") -> Optional[GateResult]:
        ...

    def validate_final_completeness(self, action: ProposedAction, wm: "WorkingMemory") -> Optional[GateResult]:
        ...

    def build_commit_certificate(
        self,
        action: ProposedAction,
        schema: ToolEffectSchema,
        wm: "WorkingMemory",
    ) -> CommitCertificate:
        ...

    def goal_stage(self, wm: "WorkingMemory") -> str:
        ...

    def update_goal_field(
        self,
        field: GoalField,
        wm: "WorkingMemory",
        *,
        event: str,
        action_name: str = "",
        obs: Any = None,
    ) -> None:
        ...

    def score_goal_action(self, candidate: GoalActionCandidate, wm: "WorkingMemory") -> float:
        ...


class BaseCargoAdapter:
    """Domain-neutral adapter default used by tests and unknown benchmarks."""

    name = "generic"
    benchmark_name = "generic"
    domain_name = "generic"
    id_fields: Set[str] = set()
    semantic_fields: Set[str] = set()

    def enrich_schema(self, schema: ToolEffectSchema) -> ToolEffectSchema:
        merged_ids = set(schema.arg_id_fields or [])
        merged_ids.update(self.id_fields.intersection(schema.required_params or []))
        for field in list(self.id_fields):
            if field in schema.param_properties:
                merged_ids.add(field)
        schema.arg_id_fields = sorted(merged_ids)
        semantic = set(getattr(schema, "arg_semantic_fields", []) or [])
        semantic.update(f for f in self.semantic_fields if f in schema.param_properties)
        schema.arg_semantic_fields = sorted(semantic)
        return schema

    def bind_user_message(self, text: str, state: TaskState) -> List[Tuple[str, Any, bool]]:
        updates: List[Tuple[str, Any, bool]] = []
        for key, value in extract_generic_facts(text):
            state.bind_fact(key, value, source="user")
            updates.append((key, value, False))
        return updates

    def absorb_observation(self, obs: Any, state: TaskState, action_name: str = "") -> List[Tuple[str, Any, bool]]:
        updates: List[Tuple[str, Any, bool]] = []
        struct = _coerce_struct(obs)
        for key, value in _iter_scalar_items(struct):
            state.bind_fact(key, value, source="tool", confirmed=True)
            updates.append((key, value, True))
        return updates

    def update_obligations(self, state: TaskState, wm: "WorkingMemory") -> None:
        state.refresh_open_slots()

    def validate_read_action(self, action: ProposedAction, schema: ToolEffectSchema, wm: "WorkingMemory") -> GateResult:
        return GateResult.passing("state_validity", adapter=self.name, validation_level="read_permissive")

    def validate_ask_user(self, action: ProposedAction, wm: "WorkingMemory") -> GateResult:
        text = normalize_value(action.user_text or action.raw_thought)
        if text and text == normalize_value(wm.last_final_text):
            return GateResult.failing("state_validity", "ask_user_repeats_same_question")
        return GateResult.passing("state_validity", adapter=self.name, validation_level="ask_user")

    def validate_action(self, action: ProposedAction, schema: ToolEffectSchema, wm: "WorkingMemory") -> GateResult:
        return GateResult.passing("semantic_validation", adapter=self.name)

    def validate_write_completeness(self, action: ProposedAction, wm: "WorkingMemory") -> Optional[GateResult]:
        state = wm.task_state
        missing = []
        for slot in sorted(state.required_slots):
            if slot in state.unresolved_obligations:
                missing.append(slot)
                continue
            if state.fact_value(slot) in (None, "") and wm.semantic_slots.get(slot) in (None, "", []):
                missing.append(slot)
        if missing:
            return GateResult.failing(
                "completeness",
                "generic_missing_required_slots",
                missing=missing,
            )
        return None

    def validate_final_completeness(self, action: ProposedAction, wm: "WorkingMemory") -> Optional[GateResult]:
        if wm.task_state.unresolved_obligations:
            return GateResult.failing(
                "final_completeness",
                "unresolved_obligations",
                obligations=sorted(wm.task_state.unresolved_obligations),
            )
        return None

    def build_commit_certificate(
        self,
        action: ProposedAction,
        schema: ToolEffectSchema,
        wm: "WorkingMemory",
    ) -> CommitCertificate:
        cert = CommitCertificate(
            action_name=action.name,
            action_signature=action.signature(),
            certificate_type=f"{self.name}.generic_commit",
        )
        cert.require(
            "commitment_action_class",
            action.declared_class in (RiskClass.WRITE, RiskClass.IRREVERSIBLE, RiskClass.FINAL),
            "action_class_does_not_require_commit_certificate",
            declared_class=action.declared_class.value,
        )
        missing = [
            name for name in (schema.required_params or [])
            if action.args.get(name) in (None, "", [])
        ]
        cert.require(
            "required_args_present",
            not missing,
            "certificate_missing_required_args",
            missing=missing,
        )
        if action.declared_class == RiskClass.FINAL:
            cert.require(
                "final_has_user_visible_resolution",
                bool(action.user_text or wm.task_state.terminal_status or action.name),
                "final_without_user_visible_resolution",
                terminal_status=wm.task_state.terminal_status,
            )
        else:
            cert.require(
                "write_signature_present",
                bool(action.name),
                "write_without_tool_name",
                arg_keys=sorted(action.args.keys()),
            )
        return cert.finalize()

    def goal_stage(self, wm: "WorkingMemory") -> str:
        state = getattr(wm, "task_state", None)
        if getattr(wm, "task_completed", False):
            return "complete"
        if state is not None and getattr(state, "terminal_status", ""):
            return "terminal"
        if state is not None and getattr(state, "unresolved_obligations", set()):
            return "obligations_open"
        if getattr(wm, "semantic_slots", None):
            return "grounding"
        return "start"

    def update_goal_field(
        self,
        field: GoalField,
        wm: "WorkingMemory",
        *,
        event: str,
        action_name: str = "",
        obs: Any = None,
    ) -> None:
        if event == "user":
            field.record_progress("user_goal_evidence", 0.2)
        elif event == "tool":
            if _obs_is_error_or_empty(obs):
                field.record_friction(action_signature_key(action_name, {}), 0.7, "tool_no_progress")
            else:
                field.record_progress(f"tool:{action_name}", 0.4)
        elif event == "candidate_set":
            field.record_progress(f"candidates:{action_name}", 0.5)

    def score_goal_action(self, candidate: GoalActionCandidate, wm: "WorkingMemory") -> float:
        return 0.0

    def validate_candidate(self, candidate: CandidateObject, state: TaskState) -> GateResult:
        if not candidate.available:
            return GateResult.failing(
                "semantic_validation",
                "candidate_not_available",
                candidate_id=candidate.candidate_id,
            )
        failed = []
        for constraint in state.hard_constraints_for():
            if not constraint.satisfied_by(candidate.attributes):
                failed.append({
                    "slot": constraint.slot,
                    "op": constraint.op,
                    "expected": constraint.value,
                    "actual": _lookup_attr(candidate.attributes, constraint.slot),
                })
        if failed:
            return GateResult.failing(
                "semantic_validation",
                "hard_constraints_not_satisfied",
                candidate_id=candidate.candidate_id,
                failed=failed,
            )
        return GateResult.passing("semantic_validation", candidate_id=candidate.candidate_id)


class ConstraintPriorityEngine:
    """Deterministic candidate selector used by adapters.

    Selection order is invariant:
    hard constraints -> global constraints -> availability -> fallback ->
    preferences. Preferences never rescue a hard-constraint violation.
    """

    def select(
        self,
        candidates: Iterable[CandidateObject],
        *,
        hard_constraints: Iterable[Constraint] = (),
        global_constraints: Iterable[Constraint] = (),
        preferences: Iterable[Preference] = (),
        fallback_rules: Iterable[FallbackRule] = (),
    ) -> CandidateSelection:
        all_candidates = list(candidates)
        rejected: List[Dict[str, Any]] = []

        hard_valid = self._filter_constraints(
            all_candidates,
            list(hard_constraints),
            rejected,
            stage="hard_constraints",
        )
        global_valid = self._filter_constraints(
            hard_valid,
            list(global_constraints),
            rejected,
            stage="global_constraints",
        )
        available = [
            c for c in global_valid
            if c.available and bool(c.attributes.get("available", True))
        ]
        for candidate in global_valid:
            if candidate not in available:
                rejected.append({
                    "candidate_id": candidate.candidate_id,
                    "stage": "availability",
                    "reason": "candidate_not_actionable",
                })
        if not available:
            return CandidateSelection(False, reason="no_available_candidate_after_hard_constraints", rejected=rejected)

        prefs = list(preferences)
        if prefs:
            strict_pref = [c for c in available if self._matches_preferences(c, prefs)]
            if strict_pref:
                return CandidateSelection(True, candidate=self._rank(strict_pref, prefs), rejected=rejected)

        for rule in fallback_rules:
            fallback_pref = Preference(slot=rule.slot, value=rule.to_value, rank=0, source=rule.source)
            fallback_matches = [
                c for c in available
                if _compare(_lookup_attr(c.attributes, rule.slot), "eq", rule.to_value)
            ]
            if fallback_matches:
                return CandidateSelection(
                    True,
                    candidate=self._rank(fallback_matches, [fallback_pref]),
                    rejected=rejected,
                    fallback_used=rule,
                )

        return CandidateSelection(True, candidate=self._rank(available, prefs), rejected=rejected)

    @staticmethod
    def _filter_constraints(
        candidates: List[CandidateObject],
        constraints: List[Constraint],
        rejected: List[Dict[str, Any]],
        *,
        stage: str,
    ) -> List[CandidateObject]:
        if not constraints:
            return list(candidates)
        out: List[CandidateObject] = []
        for candidate in candidates:
            failed = []
            for constraint in constraints:
                if not constraint.satisfied_by(candidate.attributes):
                    failed.append({
                        "slot": constraint.slot,
                        "op": constraint.op,
                        "expected": constraint.value,
                        "actual": _lookup_attr(candidate.attributes, constraint.slot),
                    })
            if failed:
                rejected.append({
                    "candidate_id": candidate.candidate_id,
                    "stage": stage,
                    "failed": failed,
                })
                continue
            out.append(candidate)
        return out

    @staticmethod
    def _matches_preferences(candidate: CandidateObject, preferences: List[Preference]) -> bool:
        return all(
            _compare(_lookup_attr(candidate.attributes, pref.slot), "eq", pref.value)
            for pref in preferences
        )

    @staticmethod
    def _rank(candidates: List[CandidateObject], preferences: List[Preference]) -> CandidateObject:
        def score(candidate: CandidateObject) -> Tuple[int, str]:
            total = 0
            for pref in preferences:
                if _compare(_lookup_attr(candidate.attributes, pref.slot), "eq", pref.value):
                    total += max(1, 100 - pref.rank)
            return (total, str(candidate.candidate_id))

        return sorted(candidates, key=score, reverse=True)[0]


class DecisionEngine:
    """Small domain-neutral facade for deterministic decisions."""

    def __init__(self) -> None:
        self.constraints = ConstraintPriorityEngine()
        self.goal_router = SoftGoalFieldRouter()
        self.phase_engine = PhaseEngine()
        self.precommit_verifier = PreCommitVerifier()


class PhaseEngine:
    """Map adapter stages into coarse CARGO v2 phases."""

    AUTH_STAGES = {"identity", "authenticate", "authentication", "identity_or_order_anchor", "profile"}
    WRAP_STAGES = {"terminal", "complete", "post_write"}

    def decide(self, wm: "WorkingMemory", adapter: CargoDomainAdapter) -> PhaseDecision:
        try:
            stage = str(adapter.goal_stage(wm) or "start")
        except Exception:
            stage = "start"
        norm = normalize_key(stage)
        if norm in self.AUTH_STAGES:
            phase = "AUTHENTICATE"
        elif norm in {"confirm", "confirmation"}:
            phase = "CONFIRM"
        elif norm in {"commit", "commit_ready"}:
            phase = "COMMIT"
        elif norm in self.WRAP_STAGES:
            phase = "WRAP"
        else:
            phase = "DISCOVER"
        return PhaseDecision(phase=phase, stage=stage, reason=f"adapter_stage:{stage}")


class PreCommitVerifier:
    """Cheap deterministic verifier for WRITE/IRREVERSIBLE actions."""

    PLACEHOLDER_RE = re.compile(
        r"\b("
        r"latest_[a-z0-9_]+|<[^>]+>|tbd|unknown|none|null|"
        r"total_cost|taxes?_and_fees|reservation_id|flight_id|item_id"
        r")\b",
        re.I,
    )
    PSEUDO_WRITE_TOOLS = {"calculate", "calculator", "lookup_policy", "reason", "plan"}

    def verify(
        self,
        action: ProposedAction,
        schema: ToolEffectSchema,
        wm: "WorkingMemory",
        adapter: CargoDomainAdapter,
    ) -> PreCommitVerdict:
        if action.declared_class not in (RiskClass.WRITE, RiskClass.IRREVERSIBLE):
            return PreCommitVerdict(ok=True, reason="not_a_mutation")
        if normalize_key(action.name) in self.PSEUDO_WRITE_TOOLS:
            return PreCommitVerdict(
                ok=False,
                reason="unsupported_pseudo_write_tool",
                diagnostics={"action_name": action.name},
            )
        placeholders = [
            f"{path}={value}"
            for path, value in self._iter_scalars(action.args)
            if self._looks_placeholder(value)
        ]
        if placeholders:
            return PreCommitVerdict(
                ok=False,
                reason="placeholder_argument",
                diagnostics={"placeholders": placeholders[:5]},
            )
        missing = [
            name for name in (getattr(schema, "required_params", []) or [])
            if action.args.get(name) in (None, "", [])
        ]
        if missing:
            return PreCommitVerdict(
                ok=False,
                reason="missing_required_args",
                missing=missing,
            )
        wrong_typed = self._wrong_typed_ids(action, wm, adapter)
        if wrong_typed:
            return PreCommitVerdict(
                ok=False,
                reason="wrong_typed_id",
                diagnostics={"wrong_typed_ids": wrong_typed[:5]},
            )
        return PreCommitVerdict(ok=True, reason="deterministic_precommit_ok")

    def _wrong_typed_ids(
        self,
        action: ProposedAction,
        wm: "WorkingMemory",
        adapter: CargoDomainAdapter,
    ) -> List[str]:
        id_fields = {str(v) for v in getattr(adapter, "id_fields", set()) or set()}
        bad: List[str] = []
        for path, value in self._iter_scalars(action.args):
            base = path.split(".")[-1].split("[")[0]
            root = path.split(".")[0].split("[")[0]
            if base not in id_fields and root not in id_fields:
                continue
            v = str(value or "").strip()
            if not v:
                continue
            typed = list(getattr(wm, "typed_evidence_for", lambda _k: [])(base))
            typed += list(getattr(wm, "typed_evidence_for", lambda _k: [])(root))
            if typed and v in typed:
                continue
            if self._looks_adapter_id(v):
                continue
            bad.append(f"{path}={v}")
        return bad

    @classmethod
    def _looks_placeholder(cls, value: Any) -> bool:
        if isinstance(value, (int, float, bool)):
            return False
        text = str(value or "").strip()
        if not text:
            return False
        return bool(cls.PLACEHOLDER_RE.fullmatch(text) or cls.PLACEHOLDER_RE.search(text))

    @staticmethod
    def _looks_adapter_id(value: str) -> bool:
        v = str(value or "").strip()
        return bool(
            re.fullmatch(r"\d{4,}", v)
            or re.fullmatch(r"[A-Z0-9][A-Z0-9_-]{3,}", v)
            or re.fullmatch(r"[a-z]+_[a-z]+_\d{1,8}", v)
            or re.fullmatch(r"(?:credit_card|gift_card|certificate|paypal)_\d+", v)
            or re.fullmatch(r"#[A-Za-z]?\d{4,}", v)
        )

    @classmethod
    def _iter_scalars(cls, value: Any, prefix: str = "") -> Iterable[Tuple[str, Any]]:
        if isinstance(value, Mapping):
            for key, item in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                yield from cls._iter_scalars(item, path)
            return
        if isinstance(value, list):
            for idx, item in enumerate(value):
                yield from cls._iter_scalars(item, f"{prefix}[{idx}]")
            return
        yield prefix, value


class SoftGoalFieldRouter:
    """Compact soft router for progress-biased action selection.

    The router does not execute tools, prove commits, or search a tree.  It
    scores a small candidate set and lets the normal CARGO gates remain the
    hard safety boundary.
    """

    _GENERIC_ASK_RE = re.compile(
        r"\b(how can i assist|what would you like|anything else|provide more detail|"
        r"specific request|what else do you need)\b",
        re.I,
    )

    def observe_user_message(
        self,
        wm: "WorkingMemory",
        text: str,
        adapter: CargoDomainAdapter,
    ) -> None:
        field = self._field(wm)
        field.turn += 1
        if text and not field.active_goal:
            field.active_goal = str(text)[:220]
        field.sync_from_memory(wm)
        field.active_stage = self._adapter_stage(adapter, wm)
        self._adapter_update(adapter, field, wm, event="user", obs=text)

    def observe_tool_result(
        self,
        wm: "WorkingMemory",
        action_name: str,
        obs: Any,
        adapter: CargoDomainAdapter,
    ) -> None:
        field = self._field(wm)
        field.turn += 1
        if _obs_is_error_or_empty(obs):
            field.record_friction(
                action_signature_key(action_name, {}),
                0.9,
                f"{action_name}_returned_no_progress",
            )
        else:
            amount = 0.5
            name = str(action_name or "")
            if name.startswith(("get_", "find_", "list_", "search_")):
                amount = 0.7
            if name.startswith(("exchange_", "return_", "modify_", "book_", "update_", "cancel_")):
                amount = 1.5
            field.record_progress(f"tool:{name}", amount)
        field.sync_from_memory(wm)
        field.active_stage = self._adapter_stage(adapter, wm)
        self._adapter_update(adapter, field, wm, event="tool", action_name=action_name, obs=obs)

    def observe_candidate_set(
        self,
        wm: "WorkingMemory",
        candidate_set: CandidateSet,
        adapter: CargoDomainAdapter,
    ) -> None:
        field = self._field(wm)
        sig = candidate_set.set_id
        if candidate_set.empty or candidate_set.exhausted:
            field.record_friction(sig, 1.2, "empty_candidate_set")
        else:
            field.record_progress(f"candidate_set:{candidate_set.source_tool}", 0.9)
        field.sync_from_memory(wm)
        self._adapter_update(
            adapter,
            field,
            wm,
            event="candidate_set",
            action_name=candidate_set.source_tool,
            obs={"empty": candidate_set.empty, "count": len(candidate_set.candidates)},
        )

    def record_gate_failure(
        self,
        wm: "WorkingMemory",
        action: ProposedAction,
        gate: GateResult,
        adapter: CargoDomainAdapter,
    ) -> None:
        field = self._field(wm)
        sig = action.signature()
        reason = getattr(gate, "reason", "") or "gate_failure"
        field.record_friction(sig, 1.0, reason)
        if reason in {
            "repeated_action_signature",
            "user_profile_already_cached",
            "ask_user_repeats_same_question",
        }:
            field.recenter(reason)
        field.sync_from_memory(wm)
        self._adapter_update(adapter, field, wm, event="gate_failure", action_name=action.name, obs=reason)

    def record_executed_action(
        self,
        wm: "WorkingMemory",
        action: ProposedAction,
        adapter: CargoDomainAdapter,
    ) -> None:
        field = self._field(wm)
        sig = action.signature()
        field.last_selected_signature = sig
        if action.declared_class == RiskClass.READ:
            field.record_progress(f"execute_read:{action.name}", 0.2)
        elif action.declared_class in (RiskClass.WRITE, RiskClass.IRREVERSIBLE):
            field.record_progress(f"execute_write:{action.name}", 1.2)
        elif action.declared_class == RiskClass.FINAL:
            field.record_progress("execute_final", 0.6)
        elif action.declared_class == RiskClass.ASK_USER:
            if self._generic_ask(action):
                field.record_friction(sig, 0.5, "generic_ask_user")
            else:
                field.record_progress("ask_user_precise", 0.1)
        field.sync_from_memory(wm)
        self._adapter_update(adapter, field, wm, event="execute", action_name=action.name)

    def choose(
        self,
        wm: "WorkingMemory",
        candidates: Iterable[GoalActionCandidate],
        adapter: CargoDomainAdapter,
    ) -> GoalFieldDecision:
        field = self._field(wm)
        field.sync_from_memory(wm)
        field.active_stage = self._adapter_stage(adapter, wm)
        scored: List[GoalActionCandidate] = []
        for idx, candidate in enumerate(candidates):
            if candidate is None or candidate.action is None:
                continue
            candidate.score = self._score_candidate(candidate, wm, adapter, idx)
            scored.append(candidate)
        if not scored:
            raise ValueError("SoftGoalFieldRouter.choose requires at least one candidate")
        scored.sort(key=lambda c: (c.score, -len(c.signature()), c.source), reverse=True)
        selected = scored[0]
        field.candidate_actions = {c.signature(): round(c.score, 3) for c in scored[:6]}
        field.last_decision = {
            "selected": selected.signature(),
            "source": selected.source,
            "score": round(selected.score, 3),
            "alternatives": [
                {"signature": c.signature(), "source": c.source, "score": round(c.score, 3)}
                for c in scored[1:4]
            ],
        }
        return GoalFieldDecision(
            selected=selected,
            alternatives=scored[1:],
            reason=f"selected {selected.source} score={selected.score:.2f}",
        )

    def _score_candidate(
        self,
        candidate: GoalActionCandidate,
        wm: "WorkingMemory",
        adapter: CargoDomainAdapter,
        index: int,
    ) -> float:
        field = self._field(wm)
        action = candidate.action
        sig = action.signature()
        score = (
            float(candidate.progress)
            + float(candidate.uncertainty_reduction)
            + float(candidate.completion)
            + float(candidate.frame_match)
            + min(2.0, field.momentum * 0.08)
        )
        if action.declared_class == RiskClass.READ:
            score += 0.35
        elif action.declared_class == RiskClass.ASK_USER:
            score -= 0.15
            if self._generic_ask(action):
                score -= 2.4
        elif action.declared_class == RiskClass.FINAL:
            score -= 0.2
            if getattr(wm.task_state, "terminal_status", "") or action.bypass_gates:
                score += 0.6
        elif action.declared_class in (RiskClass.WRITE, RiskClass.IRREVERSIBLE):
            score += 0.55
        if sig in getattr(wm, "recent_signatures", []):
            score -= 2.0
        if getattr(wm, "failed_without_new_evidence", lambda _s: False)(sig):
            score -= 2.5
        score -= min(4.0, field.friction.get(sig, 0.0))
        if action.name == "get_user_details":
            uid = str(action.args.get("user_id") or "").strip()
            if uid and uid in getattr(wm, "user_profiles", {}):
                score -= 3.0
        if action.name.startswith("search_"):
            candidate_set = wm.task_state.candidate_set_for(action.name, action.args)
            if candidate_set and candidate_set.exhausted:
                score -= 2.0
        if action.declared_class == RiskClass.ASK_USER and getattr(wm, "semantic_slots", None):
            if getattr(wm.task_state, "open_slots", set()) or getattr(wm.task_state, "unresolved_obligations", set()):
                score -= 0.6
        try:
            score += float(adapter.score_goal_action(candidate, wm))
        except Exception:
            pass
        # Stable tiny tie-breaker: preserve source insertion order.
        score -= index * 0.001
        return score

    @classmethod
    def _generic_ask(cls, action: ProposedAction) -> bool:
        text = f"{action.user_text or ''} {action.raw_thought or ''}"
        return bool(cls._GENERIC_ASK_RE.search(text))

    @staticmethod
    def _field(wm: "WorkingMemory") -> GoalField:
        field = getattr(wm, "goal_field", None)
        if isinstance(field, GoalField):
            return field
        field = GoalField(active_goal=str(getattr(wm, "goal", "") or "")[:220])
        setattr(wm, "goal_field", field)
        return field

    @staticmethod
    def _adapter_stage(adapter: CargoDomainAdapter, wm: "WorkingMemory") -> str:
        try:
            return str(adapter.goal_stage(wm) or "start")
        except Exception:
            return "start"

    @staticmethod
    def _adapter_update(
        adapter: CargoDomainAdapter,
        field: GoalField,
        wm: "WorkingMemory",
        **kwargs: Any,
    ) -> None:
        try:
            adapter.update_goal_field(field, wm, **kwargs)
        except Exception:
            return


class GenericCargoKernel:
    """Small deterministic kernel that binds state and calls adapter validators."""

    def __init__(self, adapter: Optional[CargoDomainAdapter] = None) -> None:
        self.adapter: CargoDomainAdapter = adapter or BaseCargoAdapter()
        self.decision_engine = DecisionEngine()

    def enrich_schemas(self, schemas: Dict[str, ToolEffectSchema]) -> Dict[str, ToolEffectSchema]:
        return {name: self.adapter.enrich_schema(schema) for name, schema in schemas.items()}

    def observe_user_message(self, wm: "WorkingMemory", text: str) -> None:
        self._ensure_state_meta(wm)
        for key, value, confirmed in self.adapter.bind_user_message(text, wm.task_state):
            wm.bind_semantic_slot(key, value, confirmed=confirmed)
        self.adapter.update_obligations(wm.task_state, wm)
        self.decision_engine.goal_router.observe_user_message(wm, text, self.adapter)

    def observe_tool_result(self, wm: "WorkingMemory", action_name: str, obs: Any) -> None:
        self._ensure_state_meta(wm)
        semantic_fields = set(getattr(self.adapter, "semantic_fields", set()) or set())
        id_fields = set(getattr(self.adapter, "id_fields", set()) or set())
        for key, value, confirmed in self.adapter.absorb_observation(obs, wm.task_state, action_name):
            nkey = normalize_key(key.split(".")[-1] if isinstance(key, str) else str(key))
            if nkey in id_fields:
                continue
            if nkey in semantic_fields or nkey in {"date", "intent", "confirmation"}:
                wm.bind_semantic_slot(nkey, value, confirmed=confirmed)
        self.adapter.update_obligations(wm.task_state, wm)
        self.decision_engine.goal_router.observe_tool_result(wm, action_name, obs, self.adapter)

    def record_action_candidates(
        self,
        wm: "WorkingMemory",
        action_name: str,
        action_args: Mapping[str, Any],
        obs: Any,
    ) -> Optional[CandidateSet]:
        candidate_set = self._record_generic_candidate_set(wm.task_state, action_name, action_args, obs)
        if candidate_set is not None:
            self.decision_engine.goal_router.observe_candidate_set(wm, candidate_set, self.adapter)
        return candidate_set

    def validate_action(
        self,
        action: ProposedAction,
        schema: ToolEffectSchema,
        wm: "WorkingMemory",
    ) -> GateResult:
        self._ensure_state_meta(wm)
        if action.declared_class == RiskClass.READ:
            conflict = self._validate_read_against_bound_state(action, schema, wm)
            if not conflict.ok:
                return conflict
            return self.adapter.validate_read_action(action, schema, wm)
        if action.declared_class == RiskClass.ASK_USER:
            return self.adapter.validate_ask_user(action, wm)
        conflict = self._validate_commitment_against_bound_state(action, schema, wm)
        if not conflict.ok:
            return conflict
        return self.adapter.validate_action(action, schema, wm)

    def validate_write_completeness(self, action: ProposedAction, wm: "WorkingMemory") -> Optional[GateResult]:
        return self.adapter.validate_write_completeness(action, wm)

    def validate_final_completeness(self, action: ProposedAction, wm: "WorkingMemory") -> Optional[GateResult]:
        return self.adapter.validate_final_completeness(action, wm)

    def validate_commit_certificate(
        self,
        action: ProposedAction,
        schema: ToolEffectSchema,
        wm: "WorkingMemory",
    ) -> GateResult:
        self._ensure_state_meta(wm)
        cert = self.adapter.build_commit_certificate(action, schema, wm).finalize()
        try:
            wm.last_commit_certificate = cert.to_dict()
        except Exception:
            pass
        if not cert.ok:
            return GateResult.failing(
                "commit_certificate",
                cert.reason or "commit_certificate_failed",
                certificate=cert.to_dict(),
            )
        return GateResult.passing(
            "commit_certificate",
            certificate=cert.to_dict(),
        )

    def _ensure_state_meta(self, wm: "WorkingMemory") -> None:
        state = wm.task_state
        if not state.benchmark_name:
            state.benchmark_name = getattr(self.adapter, "benchmark_name", "")
        if not state.domain_name:
            state.domain_name = getattr(self.adapter, "domain_name", "")

    def _validate_read_against_bound_state(
        self,
        action: ProposedAction,
        schema: ToolEffectSchema,
        wm: "WorkingMemory",
    ) -> GateResult:
        """READs are retrieval-permissive.

        They may run while state is incomplete and may retrieve IDs that are
        not yet selected.  The only semantic-state conflict we block is a READ
        argument that contradicts a bound ordinary semantic field such as date,
        origin, destination, or cabin.  Opaque IDs are handled by the grounding
        gate, not by semantic state comparison.
        """
        semantic_fields = set(getattr(schema, "arg_semantic_fields", []) or [])
        semantic_fields.update(getattr(self.adapter, "semantic_fields", set()) or set())
        return self._validate_bound_state_fields(
            action,
            wm,
            semantic_fields=semantic_fields,
            skip_id_fields=True,
            validation_level="read_permissive",
        )

    def _record_generic_candidate_set(
        self,
        state: TaskState,
        action_name: str,
        action_args: Mapping[str, Any],
        obs: Any,
    ) -> Optional[CandidateSet]:
        if not str(action_name or "").startswith(("search", "list")):
            return None
        struct = _coerce_struct(obs)
        candidates: List[CandidateObject] = []
        if isinstance(struct, list):
            for idx, item in enumerate(struct):
                if isinstance(item, Mapping):
                    cid = (
                        item.get("id")
                        or item.get("flight_number")
                        or item.get("candidate_id")
                        or item.get("item_id")
                        or item.get("product_id")
                        or f"{action_name}_{idx}"
                    )
                    candidates.append(CandidateObject(
                        candidate_id=str(cid),
                        object_type=str(action_name),
                        attributes=dict(item),
                        available=bool(item.get("available", True)),
                    ))
                elif isinstance(item, list) and all(isinstance(x, Mapping) for x in item):
                    flight_numbers = [
                        str(x.get("flight_number") or x.get("id") or "").strip()
                        for x in item
                        if str(x.get("flight_number") or x.get("id") or "").strip()
                    ]
                    cid = "+".join(flight_numbers) or f"{action_name}_{idx}"
                    candidates.append(CandidateObject(
                        candidate_id=cid,
                        object_type=str(action_name),
                        attributes={"flights": [dict(x) for x in item], "value": [dict(x) for x in item]},
                        available=all(bool(x.get("available", True)) for x in item),
                    ))
                elif item not in (None, ""):
                    candidates.append(CandidateObject(
                        candidate_id=f"{action_name}_{idx}",
                        object_type=str(action_name),
                        attributes={"value": item},
                    ))
        return state.record_candidate_set(action_name, action_args, candidates)

    def _validate_commitment_against_bound_state(
        self,
        action: ProposedAction,
        schema: ToolEffectSchema,
        wm: "WorkingMemory",
    ) -> GateResult:
        semantic_fields = set(getattr(schema, "arg_semantic_fields", []) or [])
        semantic_fields.update(getattr(self.adapter, "semantic_fields", set()) or set())
        return self._validate_bound_state_fields(
            action,
            wm,
            semantic_fields=semantic_fields,
            skip_id_fields=True,
            validation_level="commitment_strict",
        )

    def _validate_bound_state_fields(
        self,
        action: ProposedAction,
        wm: "WorkingMemory",
        *,
        semantic_fields: Set[str],
        skip_id_fields: bool,
        validation_level: str,
    ) -> GateResult:
        id_fields = set(getattr(self.adapter, "id_fields", set()) or set())
        for key, expected in wm.semantic_slots.items():
            nkey = normalize_key(key)
            if skip_id_fields and nkey in id_fields:
                continue
            if semantic_fields and nkey not in semantic_fields and nkey not in {"date", "intent", "confirmation"}:
                continue
            if expected in (None, "", []):
                continue
            if key not in action.args and nkey not in action.args:
                continue
            proposed = action.args.get(key, action.args.get(nkey))
            if proposed in (None, "", []):
                continue
            adapter_match = getattr(self.adapter, "semantic_values_match", None)
            if callable(adapter_match) and adapter_match(nkey, proposed, expected):
                continue
            if not semantic_values_match(proposed, expected):
                return GateResult.failing(
                    "state_validity",
                    f"action_{nkey}_conflicts_with_state",
                    expected=expected,
                    proposed=proposed,
                    adapter=getattr(self.adapter, "name", "generic"),
                    validation_level=validation_level,
                )
        return GateResult.passing("state_validity", validation_level=validation_level)


def semantic_values_match(proposed: Any, expected: Any) -> bool:
    p = normalize_value(proposed)
    if isinstance(expected, list):
        vals = [normalize_value(v) for v in expected]
    else:
        vals = [normalize_value(expected)]
    return any(p == v or (p and v and (p in v or v in p)) for v in vals)


def action_signature_key(name: str, args: Mapping[str, Any]) -> str:
    try:
        arg_text = json.dumps(dict(args or {}), sort_keys=True, default=str)
    except Exception:
        arg_text = str(args)
    return f"{name}({arg_text})"


_MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


def extract_generic_facts(text: str) -> List[Tuple[str, Any]]:
    """Extract benchmark-neutral facts from user text."""
    s = str(text or "")
    low = s.lower()
    out: List[Tuple[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()

    def push(key: str, value: Any) -> None:
        if value in (None, ""):
            return
        sig = (normalize_key(key), normalize_value(value))
        if sig in seen:
            return
        seen.add(sig)
        out.append((key, value))

    for m in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", s):
        push("date", m.group(0))
    for m in re.finditer(
        r"\b(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\s+"
        r"(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(\d{4}))?\b",
        low,
    ):
        year = int(m.group(3) or 2024)
        day = int(m.group(2))
        if 1 <= day <= 31:
            push("date", f"{year:04d}-{_MONTHS[m.group(1)]:02d}-{day:02d}")
    for m in re.finditer(r"\b([a-zA-Z][\w -]{1,40})\s*(?:=|:)\s*([^,.;\n]{1,80})", s):
        push(m.group(1), m.group(2).strip())
    if re.search(r"\b(confirm|confirmed|yes|go ahead|proceed)\b", low):
        push("confirmation", "yes")
    if re.search(r"\b(cancel|return|exchange|modify|update|book|reserve|purchase)\b", low):
        for verb in ("cancel", "return", "exchange", "modify", "update", "book", "reserve", "purchase"):
            if re.search(rf"\b{verb}\b", low):
                push("intent", "book" if verb in ("reserve", "purchase") else verb)
    return out


def _coerce_struct(obs: Any) -> Any:
    if isinstance(obs, str):
        s = obs.strip()
        if s.startswith("{") or s.startswith("["):
            try:
                return json.loads(s)
            except Exception:
                return obs
    return obs


def _obs_is_error_or_empty(obs: Any) -> bool:
    struct = _coerce_struct(obs)
    if struct in (None, "", [], {}):
        return True
    if isinstance(struct, Mapping):
        text = " ".join(str(v).lower() for v in struct.values())
        return bool(struct.get("error") or struct.get("status") == "error" or "not found" in text)
    if isinstance(struct, list):
        return len(struct) == 0
    text = str(struct or "").strip().lower()
    return not text or text.startswith("error") or '"error"' in text or "not found" in text


def _iter_scalar_items(obj: Any, prefix: str = "") -> Iterable[Tuple[str, Any]]:
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            kpath = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_scalar_items(value, kpath)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from _iter_scalar_items(value, f"{prefix}[{i}]")
    elif isinstance(obj, (str, int, float, bool)) and obj not in (None, ""):
        yield prefix, obj


def _lookup_attr(attrs: Mapping[str, Any], slot: str) -> Any:
    nslot = normalize_key(slot)
    for key, value in attrs.items():
        if normalize_key(str(key)) == nslot:
            return value
    return None


def _compare(actual: Any, op: str, expected: Any) -> bool:
    a = normalize_value(actual)
    e = normalize_value(expected)
    op_l = str(op or "eq").lower()
    if op_l in ("eq", "=", "=="):
        return a == e
    if op_l in ("neq", "!=", "not"):
        return a != e
    if op_l in ("contains", "includes"):
        return e in a
    if op_l in ("in", "one_of"):
        if isinstance(expected, (list, tuple, set)):
            return any(a == normalize_value(v) for v in expected)
        return a in {p.strip() for p in e.split(",")}
    if op_l in ("<=", "lt_eq", "lte", ">=", "gt_eq", "gte", "<", ">"):
        try:
            fa = float(actual)
            fe = float(expected)
        except Exception:
            return False
        if op_l in ("<=", "lt_eq", "lte"):
            return fa <= fe
        if op_l in (">=", "gt_eq", "gte"):
            return fa >= fe
        if op_l == "<":
            return fa < fe
        if op_l == ">":
            return fa > fe
    return False


__all__ = [
    "BaseCargoAdapter",
    "CandidateSelection",
    "CandidateSet",
    "CandidateObject",
    "CargoDomainAdapter",
    "Constraint",
    "ConstraintPriorityEngine",
    "DecisionEngine",
    "FallbackRule",
    "GoalActionCandidate",
    "GoalField",
    "GoalFieldDecision",
    "GoalHypothesis",
    "GenericCargoKernel",
    "Obligation",
    "PhaseDecision",
    "PhaseEngine",
    "Preference",
    "PreCommitVerdict",
    "PreCommitVerifier",
    "StateFact",
    "SoftGoalFieldRouter",
    "TaskState",
    "extract_generic_facts",
    "normalize_key",
    "normalize_value",
    "semantic_values_match",
    "action_signature_key",
]
