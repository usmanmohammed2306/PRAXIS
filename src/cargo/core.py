"""Generic CARGO-v2 state, adapter, and validation kernel.

This module is intentionally domain-neutral.  It knows about facts, slots,
constraints, preferences, candidates, actions, and gates; all domain and
benchmark semantics live in ``src.cargo.adapters``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Set, Tuple, TYPE_CHECKING

from .schemas import GateResult, ProposedAction, ToolEffectSchema

if TYPE_CHECKING:  # pragma: no cover - type-only import avoids circularity.
    from .working_memory import WorkingMemory


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")


def normalize_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


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
    constraints: List[Constraint] = field(default_factory=list)
    preferences: List[Preference] = field(default_factory=list)
    fallback_rules: List[FallbackRule] = field(default_factory=list)
    candidate_objects: Dict[str, CandidateObject] = field(default_factory=dict)
    selected_objects: Dict[str, str] = field(default_factory=dict)
    unresolved_obligations: Set[str] = field(default_factory=set)
    confirmations: Set[str] = field(default_factory=set)
    executed_writes: Set[str] = field(default_factory=set)
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

    def validate_action(self, action: ProposedAction, schema: ToolEffectSchema, wm: "WorkingMemory") -> GateResult:
        ...

    def validate_write_completeness(self, action: ProposedAction, wm: "WorkingMemory") -> Optional[GateResult]:
        ...

    def validate_final_completeness(self, action: ProposedAction, wm: "WorkingMemory") -> Optional[GateResult]:
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


class GenericCargoKernel:
    """Small deterministic kernel that binds state and calls adapter validators."""

    def __init__(self, adapter: Optional[CargoDomainAdapter] = None) -> None:
        self.adapter: CargoDomainAdapter = adapter or BaseCargoAdapter()

    def enrich_schemas(self, schemas: Dict[str, ToolEffectSchema]) -> Dict[str, ToolEffectSchema]:
        return {name: self.adapter.enrich_schema(schema) for name, schema in schemas.items()}

    def observe_user_message(self, wm: "WorkingMemory", text: str) -> None:
        self._ensure_state_meta(wm)
        for key, value, confirmed in self.adapter.bind_user_message(text, wm.task_state):
            wm.bind_semantic_slot(key, value, confirmed=confirmed)

    def observe_tool_result(self, wm: "WorkingMemory", action_name: str, obs: Any) -> None:
        self._ensure_state_meta(wm)
        for key, value, confirmed in self.adapter.absorb_observation(obs, wm.task_state, action_name):
            wm.bind_semantic_slot(key, value, confirmed=confirmed)

    def validate_action(
        self,
        action: ProposedAction,
        schema: ToolEffectSchema,
        wm: "WorkingMemory",
    ) -> GateResult:
        self._ensure_state_meta(wm)
        conflict = self._validate_action_against_bound_state(action, wm)
        if not conflict.ok:
            return conflict
        return self.adapter.validate_action(action, schema, wm)

    def validate_write_completeness(self, action: ProposedAction, wm: "WorkingMemory") -> Optional[GateResult]:
        return self.adapter.validate_write_completeness(action, wm)

    def validate_final_completeness(self, action: ProposedAction, wm: "WorkingMemory") -> Optional[GateResult]:
        return self.adapter.validate_final_completeness(action, wm)

    def _ensure_state_meta(self, wm: "WorkingMemory") -> None:
        state = wm.task_state
        if not state.benchmark_name:
            state.benchmark_name = getattr(self.adapter, "benchmark_name", "")
        if not state.domain_name:
            state.domain_name = getattr(self.adapter, "domain_name", "")

    def _validate_action_against_bound_state(self, action: ProposedAction, wm: "WorkingMemory") -> GateResult:
        for key, expected in wm.semantic_slots.items():
            if expected in (None, "", []):
                continue
            if key not in action.args:
                continue
            proposed = action.args.get(key)
            if proposed in (None, "", []):
                continue
            if not semantic_values_match(proposed, expected):
                return GateResult.failing(
                    "state_validity",
                    f"action_{key}_conflicts_with_state",
                    expected=expected,
                    proposed=proposed,
                    adapter=getattr(self.adapter, "name", "generic"),
                )
        return GateResult.passing("state_validity")


def semantic_values_match(proposed: Any, expected: Any) -> bool:
    p = normalize_value(proposed)
    if isinstance(expected, list):
        vals = [normalize_value(v) for v in expected]
    else:
        vals = [normalize_value(expected)]
    return any(p == v or (p and v and (p in v or v in p)) for v in vals)


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
    "CandidateObject",
    "CargoDomainAdapter",
    "Constraint",
    "FallbackRule",
    "GenericCargoKernel",
    "Preference",
    "StateFact",
    "TaskState",
    "extract_generic_facts",
    "normalize_key",
    "normalize_value",
    "semantic_values_match",
]
