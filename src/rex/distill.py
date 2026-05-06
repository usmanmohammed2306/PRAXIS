"""Trajectory → ProcessMemoryCard distillation.

Distillation is the heart of the procedural-memory system: it converts a
completed trajectory into a *reusable abstraction*. The output is **not**:

  * a transcript replay
  * a copy of the gold actions
  * raw arguments or IDs
  * benchmark-specific identifiers

The output **is**:

  * tool ordering hints
  * verification / escalation patterns
  * recovery sequences after observed failures
  * forbidden behaviors (anti-patterns)
  * a one-line procedural summary

The module decomposes into:

  1. ``extract_*`` helpers — each pulls one structured signal from a
     :class:`TrajectorySummary`.
  2. ``score_distillation_quality`` — combines signals into a confidence.
  3. ``distill_summary_to_card`` — orchestrator that produces a final
     :class:`ProcessMemoryCard`, audited for leakage.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .config import RexConfig, default_config
from .memory_types import (
    EscalationPattern,
    FailurePattern,
    ProcessMemoryCard,
    RecoveryPattern,
    ToolTransition,
    TrajectorySummary,
    VerificationPattern,
    _now_ms,
    _stable_id,
    _truncate,
)
from .sanitize import audit_text, sanitize_text


# ---------------------------------------------------------------------------
# Heuristic vocabulary
# ---------------------------------------------------------------------------
_MUTATING_PREFIXES = (
    "cancel", "modify", "return", "exchange", "book", "update", "send",
    "transfer", "delete", "create", "add", "remove",
)

_VERIFIER_KEYWORDS = (
    "get_", "find_", "search_", "list_", "lookup", "details", "calculate",
    "fetch", "view", "read",
)


def _looks_mutating(tool_name: str) -> bool:
    if not tool_name:
        return False
    low = tool_name.lower()
    return low.startswith(_MUTATING_PREFIXES)


def _looks_verifier(tool_name: str) -> bool:
    if not tool_name:
        return False
    low = tool_name.lower()
    return any(k in low for k in _VERIFIER_KEYWORDS)


# ---------------------------------------------------------------------------
# Extractors — each returns a structured pattern list
# ---------------------------------------------------------------------------
def extract_tool_transitions(tool_calls: Sequence[str]) -> List[ToolTransition]:
    """Adjacent (a -> b) transitions, with counts."""
    counts: Dict[Tuple[str, str], int] = {}
    for i in range(len(tool_calls) - 1):
        a = str(tool_calls[i] or "")
        b = str(tool_calls[i + 1] or "")
        if not a or not b:
            continue
        counts[(a, b)] = counts.get((a, b), 0) + 1
    return [ToolTransition(from_tool=a, to_tool=b, count=c) for (a, b), c in counts.items()]


def extract_verification_patterns(tool_calls: Sequence[str]) -> List[VerificationPattern]:
    """A verifier call immediately preceding a mutating call counts as verification."""
    out: List[VerificationPattern] = []
    seen: Set[Tuple[str, str]] = set()
    for i in range(len(tool_calls) - 1):
        a = str(tool_calls[i] or "")
        b = str(tool_calls[i + 1] or "")
        if not a or not b:
            continue
        if _looks_verifier(a) and _looks_mutating(b):
            key = (a, b)
            if key in seen:
                continue
            seen.add(key)
            out.append(VerificationPattern(
                verifier_tool=a,
                purpose=f"verify before {b}",
            ))
    return out


def extract_failure_patterns(
    summary: TrajectorySummary,
    *,
    max_patterns: int = 3,
) -> List[FailurePattern]:
    """Failure modes inferred from observations and the recorded error.

    Looks at:

      * ``summary.error`` — the controller-level error message
      * ``summary.tool_observations_redacted`` — tool stderr / "not found"
        style observations
    """
    out: List[FailurePattern] = []
    seen: Set[str] = set()

    if summary.error:
        sig = re.sub(r"\s+", " ", summary.error.lower()).strip()[:160]
        if sig and sig not in seen:
            out.append(FailurePattern(
                signature=sig,
                description=f"Controller error: {summary.error}",
            ))
            seen.add(sig)

    for i, obs in enumerate(summary.tool_observations_redacted or []):
        low = (obs or "").lower()
        if any(k in low for k in ("error", "not found", "invalid", "failed", "denied")):
            sig = re.sub(r"\s+", " ", low).strip()[:160]
            if not sig or sig in seen:
                continue
            seen.add(sig)
            tool = ""
            if i < len(summary.tool_calls):
                tool = summary.tool_calls[i]
            out.append(FailurePattern(
                signature=sig,
                description=_truncate(obs, 220),
                triggering_tool=tool,
            ))
            if len(out) >= max_patterns:
                break
    return out


def extract_recovery_patterns(
    summary: TrajectorySummary,
    failures: Sequence[FailurePattern],
) -> List[RecoveryPattern]:
    """If an error message appears mid-trajectory and the run still ended
    with reward >= 1.0, the tool calls *after* the failure are the recovery
    sequence."""
    if summary.reward < 1.0:
        return []
    if not failures:
        return []
    # Find the index of the first observation that triggered a failure.
    obs_observations = summary.tool_observations_redacted or []
    cut: Optional[int] = None
    for i, obs in enumerate(obs_observations):
        low = (obs or "").lower()
        if any(k in low for k in ("error", "not found", "invalid")):
            cut = i
            break
    if cut is None or cut >= len(summary.tool_calls):
        return []
    recovery_seq = summary.tool_calls[cut + 1 :]
    if not recovery_seq:
        return []
    return [RecoveryPattern(
        after_failure=failures[0].signature,
        sequence=list(recovery_seq[:6]),
        description=f"Recovered after observing: {failures[0].description[:120]}",
    )]


def extract_escalation_patterns(tool_calls: Sequence[str]) -> List[EscalationPattern]:
    """Detect transfers to human agents — the canonical escalation."""
    out: List[EscalationPattern] = []
    if any(t and "transfer_to_human" in t for t in tool_calls):
        out.append(EscalationPattern(
            trigger="cannot resolve within available tools or policy",
            response="transfer_to_human_agents with a concise reason",
        ))
    return out


def extract_anti_patterns(
    summary: TrajectorySummary,
) -> List[str]:
    """Anti-patterns derived from the trajectory shape.

    Examples:

      * Same tool called >3x in a row → "do not loop on the same tool".
      * Mutating tool called without any preceding verifier → "verify first".
      * Long trajectory (> 20 tool calls) with reward < 1.0 → "do not stall".
    """
    out: List[str] = []
    calls = list(summary.tool_calls)
    n = len(calls)

    # Rule 1: repeated identical calls
    if n >= 3:
        for i in range(n - 2):
            if calls[i] == calls[i + 1] == calls[i + 2]:
                out.append(
                    f"Do not loop on `{calls[i]}` more than twice; "
                    "reassess evidence before retrying."
                )
                break

    # Rule 2: mutation without verifier
    for i, t in enumerate(calls):
        if _looks_mutating(t):
            preceded_by_verifier = any(_looks_verifier(p) for p in calls[max(0, i - 2):i])
            if not preceded_by_verifier:
                out.append(
                    f"Do not call `{t}` without first verifying state with a read/get tool."
                )
                break

    # Rule 3: long stalls
    if n >= 20 and summary.reward < 1.0:
        out.append(
            "Do not exceed ~20 tool calls without making progress; "
            "ask the user for clarification or escalate instead."
        )

    return out


def extract_successful_patterns(summary: TrajectorySummary) -> List[str]:
    """Successful patterns from a fully-rewarded trajectory."""
    if summary.reward < 1.0:
        return []
    out: List[str] = []
    calls = list(summary.tool_calls)
    if not calls:
        return []
    out.append(
        f"Sequence `{ ' -> '.join(calls[:8]) }` succeeded; "
        "reproduce its shape, not its arguments."
    )
    if any(_looks_verifier(t) for t in calls):
        out.append(
            "Verification calls (get_/find_/search_) preceded mutations; keep this discipline."
        )
    if any(_looks_mutating(t) for t in calls):
        out.append(
            "Confirm exact write target and irreversible effect with the user before mutating tools."
        )
    return out


# ---------------------------------------------------------------------------
# Intent / category labels — domain-aware
# ---------------------------------------------------------------------------
def _intent_for(environment: str, tool_calls: Sequence[str]) -> str:
    """Deterministic intent label derived from the tool sequence."""
    if not tool_calls:
        return "answer_or_clarify"
    uniq: List[str] = []
    for t in tool_calls:
        if t not in uniq:
            uniq.append(t)
    if len(uniq) == 1:
        return uniq[0]
    if environment == "retail":
        return "multi_order_or_mixed_retail_work"
    if environment == "airline":
        return "multi_reservation_airline_work"
    if environment == "ace":
        return "multi_step_api_sequence"
    return "multi_step_tool_work"


def _outcome_label(reward: float) -> str:
    if reward >= 1.0:
        return "successful"
    if reward > 0.0:
        return "partial"
    return "avoid"


def _category_for(intent: str, outcome: str) -> str:
    return f"{intent}__{outcome}"


def _required_evidence(environment: str, tool_calls: Sequence[str]) -> List[str]:
    evidence = ["authenticate the user using current conversation facts"]
    if environment == "retail":
        if any(("order" in t or "return" in t or "exchange" in t) for t in tool_calls):
            evidence.append(
                "fetch order details and use item/product/payment IDs from tool observations"
            )
        if any(("items" in t or "exchange" in t) for t in tool_calls):
            evidence.append(
                "fetch product variants and apply hard constraints before preferences"
            )
    elif environment == "airline":
        evidence.append(
            "fetch user profile once, then choose reservations by active route/date/cabin frame"
        )
        if any(("flight" in t or "book" in t) for t in tool_calls):
            evidence.append("search flights only for the currently requested route and date")
        if any("baggage" in t for t in tool_calls):
            evidence.append("calculate free versus paid bags from membership/cabin policy")
    elif environment == "ace":
        evidence.append("inspect tool descriptions and required parameters before calling")
    else:
        evidence.append("use only tool-schema arguments supported by the current task")
    return evidence


def _confirmation_for(tool_calls: Sequence[str], outcome: str) -> str:
    if outcome == "avoid":
        return "stop and ask the user before writing; previous attempt did not succeed"
    if any(_looks_mutating(t) for t in tool_calls):
        return (
            "ask the user to confirm the exact target, selected option, "
            "payment/refund method, and irreversible effect"
        )
    return "no write confirmation needed; answer from retrieved evidence"


def _common_trap_for(environment: str, tool_calls: Sequence[str], failures: Sequence[FailurePattern]) -> str:
    if failures:
        return _truncate(failures[0].description, 200)
    if environment == "retail":
        if any("exchange" in t or "items" in t for t in tool_calls):
            return "do not copy example item IDs; select current task variants from current product details"
        if any("return" in t for t in tool_calls):
            return "do not invent payment IDs; use a current account payment method or ask"
        return "do not loop on placeholder emails; use current user-provided identity facts"
    if environment == "airline":
        if any("flight" in t or "book" in t for t in tool_calls):
            return "preserve active route/date frame; do not drift to old reservation segments"
        if any("cancel" in t for t in tool_calls):
            return "basic economy cannot be modified; cancellation depends on policy and user confirmation"
        return "do not repeatedly ask for user_id after profile has been fetched"
    if environment == "ace":
        return "similar tool names are decoys; use parameters and descriptions to disambiguate"
    return "follow the current tool schema; examples teach sequence, not arguments"


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------
def score_distillation_quality(
    summary: TrajectorySummary,
    *,
    failures: Sequence[FailurePattern],
    successes: Sequence[str],
) -> float:
    """Heuristic confidence in [0, 1] for the distilled card.

    Higher when:

      * trajectory length is moderate (3–15 tool calls)
      * the run completed (no controller error)
      * the run produced a positive reward OR a clearly diagnostic failure

    Lower when:

      * extremely short (1 tool call) or extremely long (>30 calls)
      * controller crashed
    """
    n = len(summary.tool_calls)
    if n == 0:
        return 0.0
    base = 0.5
    if 3 <= n <= 15:
        base += 0.2
    elif n > 30:
        base -= 0.2
    if summary.reward >= 1.0:
        base += 0.25
    elif summary.reward > 0:
        base += 0.10
    if summary.status == "error":
        base -= 0.15
    if failures:
        # A clean failure with a clear signature is more useful than a
        # vague one.
        base += 0.05
    if successes:
        base += 0.05
    return max(0.0, min(1.0, round(base, 3)))


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
@dataclass
class DistillationResult:
    """Wrapped output of distillation, including audit info."""

    card: Optional[ProcessMemoryCard]
    rejected_reason: str = ""
    diagnostics: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.diagnostics is None:
            self.diagnostics = {}


def distill_summary_to_card(
    summary: TrajectorySummary,
    *,
    config: Optional[RexConfig] = None,
    forbidden_tokens: Optional[Iterable[str]] = None,
) -> DistillationResult:
    """Produce one ``ProcessMemoryCard`` from one trajectory summary.

    Returns ``DistillationResult(card=None, rejected_reason=...)`` when the
    summary cannot be distilled (no tool calls, audit failed, etc).
    """
    cfg = config or default_config()

    if not summary.is_eligible_for_distillation(min_messages=cfg.promotion_min_messages):
        return DistillationResult(card=None, rejected_reason="ineligible")

    tool_calls = list(summary.tool_calls)
    if not tool_calls:
        return DistillationResult(card=None, rejected_reason="no_tool_calls")

    failures = extract_failure_patterns(summary)
    successes = extract_successful_patterns(summary)
    recoveries = extract_recovery_patterns(summary, failures)
    verifications = extract_verification_patterns(tool_calls)
    escalations = extract_escalation_patterns(tool_calls)
    anti_patterns = extract_anti_patterns(summary)

    intent = _intent_for(summary.environment, tool_calls)
    outcome = _outcome_label(summary.reward)
    category = _category_for(intent, outcome)
    confidence = score_distillation_quality(summary, failures=failures, successes=successes)
    if confidence < cfg.promotion_min_confidence:
        return DistillationResult(
            card=None,
            rejected_reason=f"low_confidence:{confidence:.2f}",
            diagnostics={"confidence": confidence},
        )

    sequence_hint: List[str] = []
    seen_seq: Set[str] = set()
    if summary.environment == "retail" or summary.environment == "airline":
        sequence_hint.append("ground_identity")
        seen_seq.add("ground_identity")
    for t in tool_calls:
        if t in seen_seq:
            continue
        sequence_hint.append(t)
        seen_seq.add(t)

    recommended_next: List[str] = []
    seen_rec: Set[str] = set()
    for t in tool_calls:
        if t not in seen_rec:
            recommended_next.append(t)
            seen_rec.add(t)
        if len(recommended_next) >= 5:
            break

    successful_text: List[str] = list(successes)
    failure_text: List[str] = [f.description for f in failures]
    recovery_text: List[str] = [
        f"after `{r.after_failure[:80]}`: {' -> '.join(r.sequence)}"
        for r in recoveries
    ]
    forbidden: List[str] = [
        "do not copy IDs, emails, payment methods, dates, or argument values from examples",
    ]
    forbidden.extend(anti_patterns)
    if escalations:
        forbidden.append(
            f"escalate when: {escalations[0].trigger} via {escalations[0].response}"
        )

    common_trap = _common_trap_for(summary.environment, tool_calls, failures)
    confirmation = _confirmation_for(tool_calls, outcome)
    evidence = _required_evidence(summary.environment, tool_calls)

    instruction_template = sanitize_text(summary.initial_user, limit=360) or "runtime-distilled-card"
    procedural_summary = sanitize_text(
        f"{intent}: {' -> '.join(tool_calls[:8])}",
        limit=240,
    )

    card_id = _stable_id(
        f"{summary.environment}-runtime",
        summary.task_id,
        str(summary.trial),
        category,
        "->".join(tool_calls[:6]),
    )
    pmc = ProcessMemoryCard(
        card_id=card_id,
        timestamp_ms=_now_ms(),
        benchmark=summary.benchmark or "unknown",
        environment=summary.environment or "generic",
        controller_source=summary.controller or "unknown",
        task_category=category,
        source=f"runtime_{outcome}",
        procedural_summary=procedural_summary,
        instruction_template=instruction_template,
        successful_patterns=successful_text,
        failure_patterns=failure_text,
        recovery_heuristics=recovery_text,
        recommended_next_tools=recommended_next,
        forbidden_behaviors=forbidden,
        required_evidence=evidence,
        tool_ordering_hints=sequence_hint,
        confirmation=confirmation,
        common_trap=common_trap,
        retrieval_text="",
        confidence=confidence,
        usage_count=0,
        success_count=1 if outcome == "successful" else 0,
        failure_count=1 if outcome == "avoid" else 0,
        outcome=outcome,
    )

    # Final audit: every textual field must be free of unredacted PII and
    # forbidden tokens.
    audit_blob = "\n".join([
        pmc.procedural_summary,
        pmc.instruction_template,
        pmc.confirmation,
        pmc.common_trap,
        " | ".join(pmc.successful_patterns),
        " | ".join(pmc.failure_patterns),
        " | ".join(pmc.recovery_heuristics),
        " | ".join(pmc.forbidden_behaviors),
        " | ".join(pmc.required_evidence),
        " | ".join(pmc.tool_ordering_hints),
        " | ".join(pmc.recommended_next_tools),
        pmc.retrieval_text,
    ])
    ok, reason = audit_text(audit_blob, forbidden_tokens=forbidden_tokens or ())
    if not ok:
        return DistillationResult(
            card=None,
            rejected_reason=reason,
            diagnostics={"confidence": confidence},
        )

    return DistillationResult(
        card=pmc,
        rejected_reason="",
        diagnostics={
            "confidence": confidence,
            "outcome": outcome,
            "num_tool_calls": len(tool_calls),
            "num_failures": len(failures),
            "num_recoveries": len(recoveries),
            "num_verifications": len(verifications),
            "num_escalations": len(escalations),
            "num_anti_patterns": len(anti_patterns),
        },
    )


def distill_record_to_card(
    record: Dict[str, Any],
    *,
    config: Optional[RexConfig] = None,
    forbidden_tokens: Optional[Iterable[str]] = None,
) -> DistillationResult:
    """Convenience: parse + distill in one step."""
    from .trajectory_parser import parse_record

    summary = parse_record(record)
    if summary is None:
        return DistillationResult(card=None, rejected_reason="unparseable")
    return distill_summary_to_card(summary, config=config, forbidden_tokens=forbidden_tokens)


__all__ = [
    "DistillationResult",
    "distill_summary_to_card",
    "distill_record_to_card",
    "extract_tool_transitions",
    "extract_verification_patterns",
    "extract_failure_patterns",
    "extract_recovery_patterns",
    "extract_escalation_patterns",
    "extract_anti_patterns",
    "extract_successful_patterns",
    "score_distillation_quality",
]
