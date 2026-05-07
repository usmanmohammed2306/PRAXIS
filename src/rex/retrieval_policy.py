"""Runtime retrieval policy control for decision-conditioned retrieval.

The retrieval policy engine determines:
  * when retrieval is necessary
  * what type of memory to retrieve
  * how much memory to retrieve
  * which retrieval mode to use

The system dynamically decides based on operational state whether to
retrieve guidance for startup, recovery, failure handling, verification,
escalation, retries, planning, or anti-pattern avoidance.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from .state_graph import OperationalStateGraph, OperationalPhase, TransitionOutcome


class RetrievalMode(str, Enum):
    """Retrieval mode determines what type of guidance to fetch."""
    STARTUP = "startup"
    RECOVERY = "recovery"
    FAILURE = "failure"
    VERIFICATION = "verification"
    ESCALATION = "escalation"
    RETRY = "retry"
    PLANNING = "planning"
    ANTI_PATTERN = "anti_pattern"


@dataclass
class RetrievalPolicy:
    """Policy decision: should we retrieve, and if so, how?"""

    should_retrieve: bool
    retrieval_mode: RetrievalMode
    top_k: int = 3
    urgency: float = 0.0  # 0=low, 1=high (for prioritization)
    rationale: str = ""


class RuntimeRetrievalPolicyEngine:
    """Dynamically determines retrieval needs based on operational state."""

    def __init__(
        self,
        default_top_k: int = 3,
        startup_top_k: int = 5,
        recovery_top_k: int = 4,
        escalation_top_k: int = 3,
    ):
        """Initialize policy engine with k-value overrides per mode."""
        self.default_top_k = default_top_k
        self.startup_top_k = startup_top_k
        self.recovery_top_k = recovery_top_k
        self.escalation_top_k = escalation_top_k
        self._retrieval_count = 0

    def determine_policy(
        self,
        state_graph: OperationalStateGraph,
        step_count: int = 0,
        last_retrieval_step: int = 0,
        min_steps_between_retrieval: int = 3,
    ) -> RetrievalPolicy:
        """Determine whether and how to retrieve based on operational state."""
        phase = state_graph.get_operational_phase()
        failed_tools = state_graph.get_failed_tools()
        successful_tools = state_graph.get_successful_tools()
        recent_failures = state_graph.get_recent_failures(count=5)
        unverified = len(state_graph.get_unresolved_verifications())
        has_escalations = len(state_graph.escalations) > 0
        is_escalation_ready = state_graph.infer_escalation_readiness()
        is_verification_ready = state_graph.infer_verification_ready()
        retry_count = sum(
            state_graph.infer_retry_count(tool)
            for tool in set(failed_tools + successful_tools)
        )
        failure_patterns = state_graph.get_failure_patterns()

        # STARTUP retrieval: Initial brief at trajectory start
        if step_count == 0:
            return RetrievalPolicy(
                should_retrieve=True,
                retrieval_mode=RetrievalMode.STARTUP,
                top_k=self.startup_top_k,
                urgency=1.0,
                rationale="Initial trajectory startup",
            )

        # ESCALATION retrieval: When escalation conditions are met
        if is_escalation_ready and not has_escalations:
            return RetrievalPolicy(
                should_retrieve=True,
                retrieval_mode=RetrievalMode.ESCALATION,
                top_k=self.escalation_top_k,
                urgency=0.95,
                rationale="Escalation path exhausted, need guidance",
            )

        # RECOVERY retrieval: Active failures with recovery attempts
        if recent_failures and any(
            f.recovery_attempted for f in recent_failures
        ):
            return RetrievalPolicy(
                should_retrieve=True,
                retrieval_mode=RetrievalMode.RECOVERY,
                top_k=self.recovery_top_k,
                urgency=0.9,
                rationale=f"Recovery needed for {len(recent_failures)} recent failures",
            )

        # FAILURE retrieval: Tool failed but recovery not yet attempted
        if recent_failures and not any(
            f.recovery_attempted for f in recent_failures
        ):
            return RetrievalPolicy(
                should_retrieve=True,
                retrieval_mode=RetrievalMode.FAILURE,
                top_k=self.default_top_k,
                urgency=0.8,
                rationale=f"Tool failure detected: {recent_failures[0].error_type}",
            )

        # ANTI_PATTERN retrieval: High retry count or repeated failures
        if retry_count > 2:
            return RetrievalPolicy(
                should_retrieve=True,
                retrieval_mode=RetrievalMode.ANTI_PATTERN,
                top_k=self.default_top_k,
                urgency=0.75,
                rationale=f"Retry count high: {retry_count}",
            )

        # VERIFICATION retrieval: Unresolved verifications
        if unverified > 0:
            return RetrievalPolicy(
                should_retrieve=True,
                retrieval_mode=RetrievalMode.VERIFICATION,
                top_k=self.default_top_k,
                urgency=0.7,
                rationale=f"{unverified} unresolved verifications",
            )

        # RETRY retrieval: Recent failure with potential for retry
        if recent_failures:
            recent_failure = recent_failures[0]
            if not recent_failure.recovery_attempted:
                retry_ceiling = 3
                if recent_failure.retry_count < retry_ceiling:
                    return RetrievalPolicy(
                        should_retrieve=True,
                        retrieval_mode=RetrievalMode.RETRY,
                        top_k=self.default_top_k - 1,
                        urgency=0.6,
                        rationale=f"Retry available ({recent_failure.retry_count}/{retry_ceiling})",
                    )

        # PLANNING retrieval: During normal exploration/execution
        if phase in (OperationalPhase.EXPLORATION, OperationalPhase.COMPLETION):
            # Check if enough steps have passed since last retrieval
            steps_since_retrieval = step_count - last_retrieval_step
            if steps_since_retrieval >= min_steps_between_retrieval:
                return RetrievalPolicy(
                    should_retrieve=True,
                    retrieval_mode=RetrievalMode.PLANNING,
                    top_k=self.default_top_k,
                    urgency=0.3,
                    rationale=f"Periodic re-planning ({steps_since_retrieval} steps passed)",
                )

        # No retrieval needed at this step
        return RetrievalPolicy(
            should_retrieve=False,
            retrieval_mode=RetrievalMode.PLANNING,
            urgency=0.0,
            rationale="No retrieval needed at this step",
        )

    def should_refresh_brief(
        self,
        state_graph: OperationalStateGraph,
        step_count: int = 0,
        last_retrieval_step: int = 0,
        refresh_every: int = 3,
    ) -> bool:
        """Determine if a brief refresh is needed.

        This is a simpler check than full policy determination, used for
        periodic refreshes during normal execution.
        """
        if step_count == 0:
            return True

        # Refresh on phase changes
        phase = state_graph.get_operational_phase()
        if phase in (OperationalPhase.RECOVERY, OperationalPhase.ESCALATION):
            if step_count > last_retrieval_step:
                return True

        # Refresh on periodic schedule
        steps_since = step_count - last_retrieval_step
        if steps_since >= refresh_every:
            return True

        return False

    def estimate_urgency(
        self,
        state_graph: OperationalStateGraph,
    ) -> float:
        """Estimate overall urgency of retrieval [0, 1]."""
        phase = state_graph.get_operational_phase()
        failed_tools = state_graph.get_failed_tools()
        recent_failures = state_graph.get_recent_failures(count=3)
        is_escalation_ready = state_graph.infer_escalation_readiness()

        urgency = 0.0

        # Phase-based urgency
        phase_urgency_map = {
            OperationalPhase.INITIAL: 0.8,
            OperationalPhase.EXPLORATION: 0.4,
            OperationalPhase.VERIFICATION: 0.6,
            OperationalPhase.RECOVERY: 0.85,
            OperationalPhase.ESCALATION: 0.95,
            OperationalPhase.COMPLETION: 0.2,
        }
        urgency = phase_urgency_map.get(phase, 0.4)

        # Boost urgency for unrecovered failures
        if recent_failures:
            unrecovered = [
                f for f in recent_failures
                if f.recovery_attempted and not f.recovery_successful
            ]
            if unrecovered:
                urgency = max(urgency, 0.85)

        # Escalation is extremely urgent
        if is_escalation_ready:
            urgency = max(urgency, 0.95)

        return min(1.0, urgency)

    def get_mode_description(self, mode: RetrievalMode) -> str:
        """Get human-readable description of a retrieval mode."""
        descriptions = {
            RetrievalMode.STARTUP: "Initial task analysis and planning",
            RetrievalMode.RECOVERY: "Recovery strategies for failures",
            RetrievalMode.FAILURE: "Guidance for tool failure handling",
            RetrievalMode.VERIFICATION: "Verification requirements and evidence",
            RetrievalMode.ESCALATION: "Escalation procedures and criteria",
            RetrievalMode.RETRY: "Retry limits and strategies",
            RetrievalMode.PLANNING: "General planning and next-step guidance",
            RetrievalMode.ANTI_PATTERN: "Anti-patterns and common mistakes",
        }
        return descriptions.get(mode, "Unknown retrieval mode")


__all__ = [
    "RuntimeRetrievalPolicyEngine",
    "RetrievalPolicy",
    "RetrievalMode",
]
