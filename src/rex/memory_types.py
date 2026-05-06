"""Typed memory data models for the REx-RPE procedural-memory architecture.

These dataclasses are the canonical schema for everything that flows
between the trajectory parser, distiller, store, retriever, and playbook
synthesizer. They are deliberately:

  * **frozen** where possible — process memory is immutable once written
  * **deterministic** under JSON serialization — sort keys, stable types
  * **typed** — enforced via ``__post_init__`` validation

Backwards compatibility note: ``ProcessMemoryCard`` is a strict superset
of the old ``ExperienceCard`` (it adds confidence, usage stats, retrieval
text, etc.). The legacy ``ExperienceCard`` lives in ``experience.py`` and
is kept for the seed-bank format. ``from_experience_card`` and
``to_experience_card`` bridge the two.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now_ms() -> int:
    """Current wall-clock time in milliseconds (int, deterministic granularity)."""
    return int(time.time() * 1000)


def _stable_id(prefix: str, *parts: str) -> str:
    """Deterministic short id from a list of parts; falls back to uuid4 hex."""
    if not parts:
        return f"{prefix}-{uuid.uuid4().hex[:12]}"
    joined = "|".join(p for p in parts if p)
    digest = abs(hash(joined)) & 0xFFFFFFFFFFFF
    return f"{prefix}-{digest:012x}"


def _truncate(text: str, limit: int) -> str:
    """Whitespace-collapse and clip ``text`` to ``limit`` chars."""
    s = " ".join(str(text or "").split())
    if len(s) <= max(0, limit):
        return s
    return s[: max(0, limit - 3)].rstrip() + "..."


def _coerce_str_list(value: Any) -> List[str]:
    """Best-effort conversion to a clean ``List[str]`` (drops empties + dupes)."""
    if value is None:
        return []
    if isinstance(value, str):
        seq: List[str] = [value]
    elif isinstance(value, (list, tuple)):
        seq = []
        for v in value:
            if v is None:
                continue
            seq.append(str(v))
    else:
        seq = [str(value)]
    seen: List[str] = []
    seen_set: set[str] = set()
    for s in seq:
        s = s.strip()
        if not s or s in seen_set:
            continue
        seen.append(s)
        seen_set.add(s)
    return seen


# ---------------------------------------------------------------------------
# Granular pattern types — extracted from trajectories during distillation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ToolTransition:
    """Adjacent tool calls observed in a successful trajectory."""

    from_tool: str
    to_tool: str
    count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {"from_tool": self.from_tool, "to_tool": self.to_tool, "count": int(self.count)}


@dataclass(frozen=True)
class FailurePattern:
    """A failure mode observed in a trajectory.

    ``signature`` is the redacted, lowercased, whitespace-collapsed text used
    for dedup; the human-readable form is in ``description``.
    """

    signature: str
    description: str
    triggering_tool: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signature": self.signature,
            "description": self.description,
            "triggering_tool": self.triggering_tool,
        }


@dataclass(frozen=True)
class RecoveryPattern:
    """A successful recovery sequence from a previous failure."""

    after_failure: str
    sequence: List[str]
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "after_failure": self.after_failure,
            "sequence": list(self.sequence),
            "description": self.description,
        }


@dataclass(frozen=True)
class VerificationPattern:
    """Tool calls used to verify state before mutation."""

    verifier_tool: str
    purpose: str

    def to_dict(self) -> Dict[str, Any]:
        return {"verifier_tool": self.verifier_tool, "purpose": self.purpose}


@dataclass(frozen=True)
class EscalationPattern:
    """Conditions under which the agent escalates to a human."""

    trigger: str
    response: str

    def to_dict(self) -> Dict[str, Any]:
        return {"trigger": self.trigger, "response": self.response}


# ---------------------------------------------------------------------------
# Trajectory parsing intermediate
# ---------------------------------------------------------------------------
@dataclass
class TrajectorySummary:
    """A normalized summary of a single trajectory.jsonl record.

    The summary is the input to ``distill_trajectory_to_card``. It is
    deliberately schema-tolerant of upstream variation (tau, ACE,
    interrupted runs, error-only records). Fields that could not be
    extracted are simply empty.
    """

    task_id: str
    trial: int
    benchmark: str  # "tau-bench" | "ACEBench" | "unknown"
    environment: str  # "retail" | "airline" | "ace" | ...
    controller: str  # "baseline" | "act" | "react" | "rex" | ...
    reward: float
    status: str  # "ok" | "error" | "interrupted"
    error: str
    initial_user: str  # *redacted*, truncated
    last_user: str  # *redacted*, truncated
    tool_calls: List[str]
    tool_arguments_redacted: List[str]
    tool_observations_redacted: List[str]
    assistant_messages: int
    user_messages: int
    tool_messages: int
    info: Dict[str, Any] = field(default_factory=dict)

    def is_eligible_for_distillation(self, *, min_messages: int = 2) -> bool:
        """Predicate matching ``_eligible_for_promotion`` in the legacy code."""
        if self.status not in ("", "ok"):
            return False
        total_msgs = self.assistant_messages + self.user_messages + self.tool_messages
        if total_msgs < min_messages:
            return False
        return bool(self.tool_calls)

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self.__class__)}


# ---------------------------------------------------------------------------
# The procedural memory card — the core schema
# ---------------------------------------------------------------------------
@dataclass
class ProcessMemoryCard:
    """A redacted, procedural lesson distilled from one or more trajectories.

    A card stores *process* — what tool to call when, what to verify, what
    to avoid. It must never store concrete arguments (ids, emails, payment
    methods). The audit pass enforces this.
    """

    # Identity
    card_id: str
    timestamp_ms: int

    # Provenance
    benchmark: str  # tau-bench | ACEBench | seed | unknown
    environment: str  # retail | airline | ace | generic
    controller_source: str  # baseline | act | react | rex | seed | policy
    task_category: str  # intent_label__outcome (e.g. exchange_items__success)
    source: str  # human-readable source label (e.g. "runtime_successful")

    # Core procedural content
    procedural_summary: str  # one-line abstract
    instruction_template: str  # representative redacted user request
    successful_patterns: List[str]
    failure_patterns: List[str]
    recovery_heuristics: List[str]
    recommended_next_tools: List[str]  # ordered next-tool hints
    forbidden_behaviors: List[str]
    required_evidence: List[str]
    tool_ordering_hints: List[str]
    confirmation: str
    common_trap: str

    # Retrieval / scoring
    retrieval_text: str  # concat of fields used by retrieval; precomputed
    confidence: float = 1.0
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0

    # Outcome flag — successful | partial | avoid | seed
    outcome: str = "seed"

    def __post_init__(self) -> None:
        # Normalize collections.
        self.successful_patterns = _coerce_str_list(self.successful_patterns)
        self.failure_patterns = _coerce_str_list(self.failure_patterns)
        self.recovery_heuristics = _coerce_str_list(self.recovery_heuristics)
        self.recommended_next_tools = _coerce_str_list(self.recommended_next_tools)
        self.forbidden_behaviors = _coerce_str_list(self.forbidden_behaviors)
        self.required_evidence = _coerce_str_list(self.required_evidence)
        self.tool_ordering_hints = _coerce_str_list(self.tool_ordering_hints)

        # Lower-bound confidence at 0.0; cap at 1.0.
        try:
            c = float(self.confidence)
        except (TypeError, ValueError):
            c = 0.5
        self.confidence = max(0.0, min(1.0, c))

        for attr in ("usage_count", "success_count", "failure_count"):
            try:
                setattr(self, attr, max(0, int(getattr(self, attr))))
            except (TypeError, ValueError):
                setattr(self, attr, 0)

        # Always have a deterministic retrieval_text so the retriever can
        # operate without re-derivation.
        if not self.retrieval_text:
            self.retrieval_text = self._derive_retrieval_text()

    # ---- Serialization ------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self.__class__)}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessMemoryCard":
        valid = {f.name for f in fields(cls)}
        clean = {k: v for k, v in (data or {}).items() if k in valid}
        clean.setdefault("card_id", _stable_id("card", json.dumps(data, sort_keys=True, default=str)))
        clean.setdefault("timestamp_ms", _now_ms())
        clean.setdefault("benchmark", "unknown")
        clean.setdefault("environment", "generic")
        clean.setdefault("controller_source", "unknown")
        clean.setdefault("task_category", "unknown")
        clean.setdefault("source", "unknown")
        clean.setdefault("procedural_summary", "")
        clean.setdefault("instruction_template", "")
        clean.setdefault("confirmation", "")
        clean.setdefault("common_trap", "")
        for key in (
            "successful_patterns",
            "failure_patterns",
            "recovery_heuristics",
            "recommended_next_tools",
            "forbidden_behaviors",
            "required_evidence",
            "tool_ordering_hints",
        ):
            clean.setdefault(key, [])
        clean.setdefault("retrieval_text", "")
        clean.setdefault("confidence", 1.0)
        clean.setdefault("usage_count", 0)
        clean.setdefault("success_count", 0)
        clean.setdefault("failure_count", 0)
        clean.setdefault("outcome", "seed")
        return cls(**clean)

    # ---- Derived ------------------------------------------------------
    def _derive_retrieval_text(self) -> str:
        parts = [
            self.environment,
            self.task_category,
            self.procedural_summary,
            self.instruction_template,
            " ".join(self.tool_ordering_hints),
            " ".join(self.recommended_next_tools),
            " ".join(self.required_evidence),
            self.common_trap,
            " ".join(self.failure_patterns),
            " ".join(self.successful_patterns),
        ]
        return _truncate(" ".join(p for p in parts if p), 1200)

    def signature(self, max_chars: int = 220) -> str:
        """Stable text used by the dedup engine."""
        return _truncate(
            "|".join([
                self.environment,
                self.task_category,
                "->".join(self.tool_ordering_hints),
                self.common_trap,
            ]),
            max_chars,
        )

    def update_usage(self, *, success: Optional[bool] = None) -> None:
        """Bump usage stats; ``success`` updates success/failure counters when set."""
        self.usage_count += 1
        if success is True:
            self.success_count += 1
        elif success is False:
            self.failure_count += 1

    def quality_score(self) -> float:
        """Combined score in [0, 1] used by the quality system."""
        n = max(1, self.usage_count)
        success_rate = self.success_count / n if n else 0.0
        return max(0.0, min(1.0, 0.7 * self.confidence + 0.3 * success_rate))


# ---------------------------------------------------------------------------
# Retrieval IO objects
# ---------------------------------------------------------------------------
@dataclass
class RetrievalQuery:
    """A single retrieval request capturing the current working state.

    Built fresh at every refresh point during execution.
    """

    text: str
    environment: str = "generic"
    benchmark: str = "unknown"
    last_tool: str = ""
    last_observation: str = ""
    failed_tools: List[str] = field(default_factory=list)
    attempted_tools: List[str] = field(default_factory=list)
    controller: str = ""
    step_index: int = 0
    top_k: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self.__class__)}


@dataclass
class RetrievalResult:
    """A retrieval ranking + scores for diagnostics and reconstruction."""

    cards: List[ProcessMemoryCard]
    scores: List[float]
    diagnostic: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.cards)

    def card_ids(self) -> List[str]:
        return [c.card_id for c in self.cards]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cards": [c.to_dict() for c in self.cards],
            "scores": [float(s) for s in self.scores],
            "diagnostic": self.diagnostic,
        }


# ---------------------------------------------------------------------------
# Tactical playbook
# ---------------------------------------------------------------------------
@dataclass
class TacticalPlaybook:
    """Compact synthesized prompt fragment derived from retrieved cards.

    Synthesis lives in ``rex.playbook``. The playbook is the only piece of
    procedural memory injected into the runtime prompt.
    """

    body: str  # rendered text ready for prompt insertion
    sections: Dict[str, List[str]]  # named buckets (verification, sequencing, …)
    card_ids: List[str]  # cards used for synthesis (for diagnostics)
    char_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "body": self.body,
            "sections": {k: list(v) for k, v in self.sections.items()},
            "card_ids": list(self.card_ids),
            "char_count": int(self.char_count),
        }


# ---------------------------------------------------------------------------
# Bridge to the legacy ExperienceCard schema
# ---------------------------------------------------------------------------
def from_experience_card(
    card: Any,
    *,
    benchmark: str = "seed",
    controller_source: str = "seed",
    outcome: str = "seed",
) -> ProcessMemoryCard:
    """Lift a legacy ``ExperienceCard`` (from ``experience.py``) to ``ProcessMemoryCard``."""
    intent = getattr(card, "intent", "") or ""
    domain = getattr(card, "domain", "generic") or "generic"
    sequence = list(getattr(card, "tool_sequence", []) or [])
    evidence = list(getattr(card, "needed_evidence", []) or [])
    confirmation = getattr(card, "confirmation", "") or ""
    common_trap = getattr(card, "common_trap", "") or ""
    instruction = getattr(card, "instruction_template", "") or ""
    source = getattr(card, "source", "seed") or "seed"
    cid = getattr(card, "card_id", None) or _stable_id(
        f"{domain}-seed", intent, "->".join(sequence)
    )

    procedural_summary = (
        f"{intent}: {' -> '.join(sequence) if sequence else 'respond'}"
    )
    forbidden = [
        "do not copy IDs, emails, payment methods, dates, or argument values from examples",
    ]
    if common_trap:
        forbidden.append(common_trap)

    pmc = ProcessMemoryCard(
        card_id=str(cid),
        timestamp_ms=_now_ms(),
        benchmark=benchmark,
        environment=domain,
        controller_source=controller_source,
        task_category=intent or "unknown",
        source=source,
        procedural_summary=procedural_summary,
        instruction_template=instruction,
        successful_patterns=[],
        failure_patterns=[common_trap] if common_trap else [],
        recovery_heuristics=[],
        recommended_next_tools=sequence[:3] if sequence else [],
        forbidden_behaviors=forbidden,
        required_evidence=evidence,
        tool_ordering_hints=sequence,
        confirmation=confirmation,
        common_trap=common_trap,
        retrieval_text="",
        confidence=0.85 if outcome == "seed" else 1.0,
        usage_count=0,
        outcome=outcome,
    )
    return pmc


def to_experience_card_kwargs(card: ProcessMemoryCard) -> Dict[str, Any]:
    """Project a ``ProcessMemoryCard`` back to legacy ``ExperienceCard`` kwargs."""
    return {
        "card_id": card.card_id,
        "domain": card.environment,
        "source": card.source,
        "intent": card.task_category,
        "instruction_template": card.instruction_template,
        "needed_evidence": list(card.required_evidence),
        "tool_sequence": list(card.tool_ordering_hints),
        "confirmation": card.confirmation,
        "common_trap": card.common_trap,
    }


__all__ = [
    "ProcessMemoryCard",
    "RetrievalQuery",
    "RetrievalResult",
    "TacticalPlaybook",
    "TrajectorySummary",
    "ToolTransition",
    "FailurePattern",
    "RecoveryPattern",
    "VerificationPattern",
    "EscalationPattern",
    "from_experience_card",
    "to_experience_card_kwargs",
]
