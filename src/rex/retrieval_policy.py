"""Retrieval trigger control — decides when and how to retrieve experiences.

The system retrieves experiences from permanent memory during execution to
improve tool-calling decisions. This module determines:

  * WHEN to retrieve (startup, after failure, on retry, etc.)
  * WHAT type of experiences to retrieve (similar failures, recoveries, etc.)
  * HOW MANY experiences to retrieve per situation

The runtime agent retrieves prior experiences to condition its next action.
This is experience-conditioned execution, not policy evolution.

Eight retrieval trigger conditions:
  1. startup     — initial planning at task start
  2. recovery    — after a failed recovery attempt
  3. failure     — immediately after a tool failure
  4. verification — when verification state is unclear
  5. escalation  — when escalation conditions are met
  6. retry       — before retrying a previously failed tool
  7. planning    — periodic re-retrieval during normal execution
  8. anti_pattern — when retry count is high (avoid known traps)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .state_graph import OperationalStateGraph, OperationalPhase


class RetrievalMode(str, Enum):
    """What kind of prior experiences to retrieve."""
    STARTUP = "startup"
    RECOVERY = "recovery"
    FAILURE = "failure"
    VERIFICATION = "verification"
    ESCALATION = "escalation"
    RETRY = "retry"
    PLANNING = "planning"
    ANTI_PATTERN = "anti_pattern"


@dataclass
class RetrievalDecision:
    """Decision: should we retrieve experiences, and with what focus?"""

    should_retrieve: bool
    mode: RetrievalMode
    top_k: int = 3
    urgency: float = 0.0   # 0 = low urgency, 1 = immediate
    rationale: str = ""


class RetrievalTrigger:
    """Determines when the runtime agent should retrieve prior experiences.

    The agent calls this at each step to decide whether to refresh its
    retrieved experience guidance and which type of experiences to fetch.
    """

    def __init__(
        self,
        default_top_k: int = 3,
        startup_top_k: int = 5,
        recovery_top_k: int = 4,
    ):
        self.default_top_k = default_top_k
        self.startup_top_k = startup_top_k
        self.recovery_top_k = recovery_top_k

    def evaluate(
        self,
        state_graph: OperationalStateGraph,
        step_count: int = 0,
        last_retrieval_step: int = 0,
        min_steps_between_retrieval: int = 3,
    ) -> RetrievalDecision:
        """Evaluate whether to retrieve experiences at this step.

        Returns a RetrievalDecision describing what to retrieve and why.
        """
        phase = state_graph.get_operational_phase()
        failed_tools = state_graph.get_failed_tools()
        recent_failures = state_graph.get_recent_failures(count=5)
        unverified = len(state_graph.get_unresolved_verifications())
        has_escalations = len(state_graph.escalations) > 0
        is_escalation_ready = state_graph.infer_escalation_readiness()
        retry_count = sum(
            state_graph.infer_retry_count(t)
            for t in set(failed_tools + state_graph.get_successful_tools())
        )

        # Always retrieve at startup
        if step_count == 0:
            return RetrievalDecision(
                should_retrieve=True,
                mode=RetrievalMode.STARTUP,
                top_k=self.startup_top_k,
                urgency=1.0,
                rationale="Retrieve prior experiences for initial planning",
            )

        # Escalation — retrieve similar escalation experiences
        if is_escalation_ready and not has_escalations:
            return RetrievalDecision(
                should_retrieve=True,
                mode=RetrievalMode.ESCALATION,
                top_k=self.default_top_k,
                urgency=0.95,
                rationale="Escalation conditions met; retrieve escalation experiences",
            )

        # Recovery in progress — retrieve recovery experiences
        if recent_failures and any(f.recovery_attempted for f in recent_failures):
            return RetrievalDecision(
                should_retrieve=True,
                mode=RetrievalMode.RECOVERY,
                top_k=self.recovery_top_k,
                urgency=0.9,
                rationale=f"Active recovery; retrieve {len(recent_failures)} failure contexts",
            )

        # New failure — retrieve similar failure experiences immediately
        if recent_failures and not any(f.recovery_attempted for f in recent_failures):
            return RetrievalDecision(
                should_retrieve=True,
                mode=RetrievalMode.FAILURE,
                top_k=self.default_top_k,
                urgency=0.8,
                rationale=f"Tool failure: {recent_failures[0].error_type}",
            )

        # High retry count — retrieve anti-pattern experiences
        if retry_count > 2:
            return RetrievalDecision(
                should_retrieve=True,
                mode=RetrievalMode.ANTI_PATTERN,
                top_k=self.default_top_k,
                urgency=0.75,
                rationale=f"High retry count ({retry_count}); retrieve failure-avoidance experiences",
            )

        # Pending verification — retrieve verification experiences
        if unverified > 0:
            return RetrievalDecision(
                should_retrieve=True,
                mode=RetrievalMode.VERIFICATION,
                top_k=self.default_top_k,
                urgency=0.7,
                rationale=f"{unverified} unresolved verification steps",
            )

        # Retry opportunity
        if recent_failures:
            retry_ceiling = 3
            if recent_failures[0].retry_count < retry_ceiling:
                return RetrievalDecision(
                    should_retrieve=True,
                    mode=RetrievalMode.RETRY,
                    top_k=max(1, self.default_top_k - 1),
                    urgency=0.6,
                    rationale=f"Retry opportunity ({recent_failures[0].retry_count}/{retry_ceiling})",
                )

        # Periodic re-retrieval during normal execution
        steps_since = step_count - last_retrieval_step
        if steps_since >= min_steps_between_retrieval:
            return RetrievalDecision(
                should_retrieve=True,
                mode=RetrievalMode.PLANNING,
                top_k=self.default_top_k,
                urgency=0.3,
                rationale=f"Periodic re-retrieval ({steps_since} steps since last)",
            )

        return RetrievalDecision(
            should_retrieve=False,
            mode=RetrievalMode.PLANNING,
            urgency=0.0,
            rationale="No retrieval needed at this step",
        )

    def mode_description(self, mode: RetrievalMode) -> str:
        """Human-readable description of the retrieval mode."""
        return {
            RetrievalMode.STARTUP: "Retrieve prior experiences for initial task planning",
            RetrievalMode.RECOVERY: "Retrieve recovery experiences for ongoing failure",
            RetrievalMode.FAILURE: "Retrieve similar failure experiences",
            RetrievalMode.VERIFICATION: "Retrieve verification-step experiences",
            RetrievalMode.ESCALATION: "Retrieve escalation-path experiences",
            RetrievalMode.RETRY: "Retrieve retry-strategy experiences",
            RetrievalMode.PLANNING: "Retrieve general planning experiences",
            RetrievalMode.ANTI_PATTERN: "Retrieve failure-avoidance experiences",
        }.get(mode, "Retrieve relevant experiences")


# Backward-compatible alias kept so existing imports continue to work.
RuntimeRetrievalPolicyEngine = RetrievalTrigger


__all__ = [
    "RetrievalTrigger",
    "RetrievalDecision",
    "RetrievalMode",
    "RuntimeRetrievalPolicyEngine",  # alias
]
