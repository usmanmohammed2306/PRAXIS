"""Practical operational inference for experience-conditioned execution.

Provides lightweight structured inference about the current execution state
to help the agent decide which prior experiences to retrieve and what to
watch out for.

NOT a policy hierarchy. NOT a doctrine system.
Just: "what does the current situation suggest?" so the agent can fetch
the most relevant prior operational experiences.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from .state_graph import OperationalStateGraph, OperationalPhase


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SituationAssessment:
    """Lightweight snapshot of current execution situation.

    Summarises what the agent should be aware of and which experience
    types to prioritise for the next retrieval.
    """

    risk_level: RiskLevel
    key_observations: List[str]     # What stands out in the current state
    suggested_experience_types: List[str]  # Types of prior experiences to seek
    escalation_ready: bool
    recovery_advisable: bool
    retry_within_ceiling: bool      # True when retries are still worth trying
    retry_ceiling: int = 3


class SituationInference:
    """Infers current execution situation for experience retrieval guidance.

    Produces a SituationAssessment that describes what the agent should
    focus on — without prescribing policy or doctrine.
    """

    def assess(self, state_graph: OperationalStateGraph) -> SituationAssessment:
        """Assess current execution situation."""
        observations: List[str] = []
        experience_types: List[str] = []
        risk = RiskLevel.LOW

        phase = state_graph.get_operational_phase()
        recent_failures = state_graph.get_recent_failures(count=3)
        failed_tools = state_graph.get_failed_tools()
        escalation_ready = state_graph.infer_escalation_readiness()
        retry_ceiling = 3
        total_retries = sum(
            state_graph.infer_retry_count(t) for t in set(failed_tools)
        )

        # ── Escalation ──────────────────────────────────────────────────────
        if escalation_ready:
            observations.append("Recovery paths appear exhausted; escalation may be needed")
            experience_types.append("escalation")
            risk = RiskLevel.CRITICAL

        # ── Active failures ──────────────────────────────────────────────────
        if recent_failures:
            tool_names = ", ".join(f.tool for f in recent_failures[:3])
            observations.append(f"Recent tool failures: {tool_names}")
            experience_types.append("recovery")
            experience_types.append("failure_avoidance")
            if risk == RiskLevel.LOW:
                risk = RiskLevel.MEDIUM

        # ── Retry ceiling ────────────────────────────────────────────────────
        if total_retries >= retry_ceiling:
            observations.append(
                f"Retry ceiling ({retry_ceiling}) reached; continuing to retry will not help"
            )
            experience_types.append("anti_pattern")
            risk = RiskLevel.HIGH

        # ── Verification pending ─────────────────────────────────────────────
        unverified = state_graph.get_unresolved_verifications()
        if unverified:
            observations.append(
                f"{len(unverified)} verification step(s) outstanding"
            )
            experience_types.append("verification")
            if risk == RiskLevel.LOW:
                risk = RiskLevel.MEDIUM

        # ── No observations yet ──────────────────────────────────────────────
        if not observations:
            if phase == OperationalPhase.INITIAL:
                observations.append("Task just started; retrieve planning experiences")
                experience_types.append("planning")
            elif phase == OperationalPhase.EXPLORATION:
                observations.append("Gathering information; normal progress")
                experience_types.append("sequencing")
            elif phase == OperationalPhase.COMPLETION:
                observations.append("Goals appear resolved")
                experience_types.append("verification")

        return SituationAssessment(
            risk_level=risk,
            key_observations=observations,
            suggested_experience_types=list(dict.fromkeys(experience_types)),  # dedup, preserve order
            escalation_ready=escalation_ready,
            recovery_advisable=bool(recent_failures) and not escalation_ready,
            retry_within_ceiling=total_retries < retry_ceiling,
            retry_ceiling=retry_ceiling,
        )


__all__ = [
    "SituationInference",
    "SituationAssessment",
    "RiskLevel",
]
