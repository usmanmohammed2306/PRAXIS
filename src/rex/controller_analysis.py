"""Cross-controller procedural meta-learning.

Analyzes trajectories from different controllers (baseline, ACT, ReAct, REx)
to extract controller-independent procedural insights and learn which
controllers excel at specific recovery/escalation/verification patterns.

Learns:
  * which controller recovers best
  * which controller escalates prematurely
  * which controller over-retries
  * which controller verifies effectively
  * which controller hallucinates tools
  * which controller sequences tools best
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


class ControllerType:
    """Controller types in the benchmark suite."""
    BASELINE = "baseline"
    ACT = "act"
    REACT = "react"
    REX = "rex"
    SEED = "seed"
    POLICY = "policy"


@dataclass
class ControllerMetrics:
    """Metrics for a specific controller's performance."""

    controller: str
    total_trajectories: int = 0
    successful_trajectories: int = 0
    partial_trajectories: int = 0
    failed_trajectories: int = 0

    # Recovery metrics
    recovery_attempts: int = 0
    recovery_successes: int = 0  # Recoveries that led to success
    recovery_failures: int = 0   # Recoveries that failed further

    # Escalation metrics
    escalations_triggered: int = 0
    escalations_premature: int = 0  # Escalated when recovery was possible
    escalations_appropriate: int = 0  # Escalated when necessary

    # Retry metrics
    total_retries: int = 0
    retry_ceiling_exceeded: int = 0  # Times retry ceiling was exceeded
    retries_helpful: int = 0  # Times retry led to success
    retries_harmful: int = 0  # Times retry made things worse

    # Verification metrics
    verification_attempts: int = 0
    verification_failures: int = 0  # Skipped verification
    verification_successes: int = 0  # Proper verification

    # Tool hallucination
    tool_calls_invalid: int = 0  # Called tools that don't exist
    tool_calls_valid: int = 0

    # Tool sequencing
    sequencing_efficient: int = 0  # Good order
    sequencing_inefficient: int = 0  # Poor order (backtracking)

    def success_rate(self) -> float:
        """Overall success rate."""
        total = self.total_trajectories
        if total == 0:
            return 0.0
        return self.successful_trajectories / total

    def recovery_success_rate(self) -> float:
        """Recovery success rate when attempted."""
        total = self.recovery_attempts
        if total == 0:
            return 0.0
        return self.recovery_successes / total

    def escalation_appropriateness(self) -> float:
        """Proportion of escalations that were appropriate."""
        total = self.escalations_triggered
        if total == 0:
            return 0.0
        return self.escalations_appropriate / total

    def retry_helpfulness(self) -> float:
        """Proportion of retries that were helpful."""
        total = self.total_retries
        if total == 0:
            return 0.0
        return self.retries_helpful / total

    def verification_effectiveness(self) -> float:
        """How often verification was done effectively."""
        total = self.verification_attempts
        if total == 0:
            return 0.0
        return self.verification_successes / total

    def tool_validity_rate(self) -> float:
        """Proportion of tool calls that were valid."""
        total = self.tool_calls_valid + self.tool_calls_invalid
        if total == 0:
            return 0.0
        return self.tool_calls_valid / total

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "controller": self.controller,
            "total_trajectories": self.total_trajectories,
            "success_rate": self.success_rate(),
            "recovery_success_rate": self.recovery_success_rate(),
            "escalation_appropriateness": self.escalation_appropriateness(),
            "retry_helpfulness": self.retry_helpfulness(),
            "verification_effectiveness": self.verification_effectiveness(),
            "tool_validity_rate": self.tool_validity_rate(),
        }


class ControllerAnalyzer:
    """Analyzes cross-controller patterns and meta-learns procedural insights."""

    def __init__(self):
        """Initialize analyzer."""
        self.metrics: Dict[str, ControllerMetrics] = {}

    def record_trajectory(
        self,
        controller: str,
        outcome: str,  # "successful", "partial", "failed"
        recovery_attempts: int = 0,
        recovery_successes: int = 0,
        escalations: int = 0,
        escalations_appropriate: int = 0,
        total_retries: int = 0,
        retries_helpful: int = 0,
        verification_attempts: int = 0,
        verification_successes: int = 0,
        invalid_tools: int = 0,
        valid_tools: int = 0,
        sequencing_efficient: bool = True,
    ) -> None:
        """Record metrics from a single trajectory."""
        if controller not in self.metrics:
            self.metrics[controller] = ControllerMetrics(controller=controller)

        m = self.metrics[controller]
        m.total_trajectories += 1

        # Outcome
        if outcome == "successful":
            m.successful_trajectories += 1
        elif outcome == "partial":
            m.partial_trajectories += 1
        else:
            m.failed_trajectories += 1

        # Recovery
        m.recovery_attempts += recovery_attempts
        m.recovery_successes += recovery_successes
        m.recovery_failures += max(0, recovery_attempts - recovery_successes)

        # Escalation
        m.escalations_triggered += escalations
        m.escalations_appropriate += escalations_appropriate
        m.escalations_premature += max(0, escalations - escalations_appropriate)

        # Retries
        m.total_retries += total_retries
        m.retries_helpful += retries_helpful
        m.retries_harmful += max(0, total_retries - retries_helpful)

        # Verification
        m.verification_attempts += verification_attempts
        m.verification_successes += verification_successes
        m.verification_failures += max(0, verification_attempts - verification_successes)

        # Tools
        m.tool_calls_invalid += invalid_tools
        m.tool_calls_valid += valid_tools

        # Sequencing
        if sequencing_efficient:
            m.sequencing_efficient += 1
        else:
            m.sequencing_inefficient += 1

    def get_best_controller_for_recovery(self) -> Tuple[str, float]:
        """Identify which controller recovers best from failures."""
        best_controller = ""
        best_rate = -1.0

        for controller, metrics in self.metrics.items():
            rate = metrics.recovery_success_rate()
            if rate > best_rate and metrics.recovery_attempts > 0:
                best_rate = rate
                best_controller = controller

        return best_controller, best_rate

    def get_best_controller_for_escalation(self) -> Tuple[str, float]:
        """Identify which controller escalates most appropriately."""
        best_controller = ""
        best_rate = -1.0

        for controller, metrics in self.metrics.items():
            rate = metrics.escalation_appropriateness()
            if rate > best_rate and metrics.escalations_triggered > 0:
                best_rate = rate
                best_controller = controller

        return best_controller, best_rate

    def get_controller_retry_tendency(self) -> Dict[str, float]:
        """Compute retry tendency for each controller.

        Returns dict mapping controller -> retry_helpfulness_rate
        """
        return {
            controller: metrics.retry_helpfulness()
            for controller, metrics in self.metrics.items()
        }

    def get_controller_verification_quality(self) -> Dict[str, float]:
        """Compute verification quality for each controller."""
        return {
            controller: metrics.verification_effectiveness()
            for controller, metrics in self.metrics.items()
        }

    def get_controller_tool_accuracy(self) -> Dict[str, float]:
        """Compute tool accuracy for each controller."""
        return {
            controller: metrics.tool_validity_rate()
            for controller, metrics in self.metrics.items()
        }

    def extract_controller_strengths(self) -> Dict[str, List[str]]:
        """Extract strengths of each controller.

        Returns dict mapping controller -> list of strengths
        """
        strengths: Dict[str, List[str]] = {}

        for controller, metrics in self.metrics.items():
            controller_strengths: List[str] = []

            # Recovery strength
            if metrics.recovery_success_rate() > 0.6:
                controller_strengths.append("Strong recovery capability")

            # Escalation strength
            if metrics.escalation_appropriateness() > 0.7:
                controller_strengths.append("Appropriate escalation decisions")

            # Retry strength
            if metrics.retry_helpfulness() > 0.5:
                controller_strengths.append("Effective retry strategy")

            # Verification strength
            if metrics.verification_effectiveness() > 0.7:
                controller_strengths.append("Thorough verification")

            # Tool accuracy
            if metrics.tool_validity_rate() > 0.95:
                controller_strengths.append("High tool accuracy")

            # Sequencing
            total_sequences = metrics.sequencing_efficient + metrics.sequencing_inefficient
            if total_sequences > 0:
                efficiency = metrics.sequencing_efficient / total_sequences
                if efficiency > 0.8:
                    controller_strengths.append("Efficient tool sequencing")

            strengths[controller] = controller_strengths if controller_strengths else ["Baseline capability"]

        return strengths

    def extract_procedural_policies(self) -> Dict[str, str]:
        """Extract controller-independent procedural policies.

        Returns dict mapping policy -> description
        """
        policies: Dict[str, str] = {}

        # Recovery policy
        best_recovery_controller, best_recovery_rate = self.get_best_controller_for_recovery()
        if best_recovery_rate > 0.5:
            policies["recovery_effectiveness"] = (
                f"Recovery attempts by {best_recovery_controller} succeed {best_recovery_rate:.0%} of the time. "
                "Establish recovery before escalation."
            )

        # Escalation policy
        best_escalation_controller, best_escalation_rate = self.get_best_controller_for_escalation()
        if best_escalation_rate > 0.6:
            policies["escalation_readiness"] = (
                f"{best_escalation_controller} escalates appropriately {best_escalation_rate:.0%} of the time. "
                "Escalate only after recovery paths exhausted."
            )

        # Verification policy
        verification_quality = self.get_controller_verification_quality()
        if verification_quality:
            best_verifier = max(verification_quality, key=verification_quality.get)
            if verification_quality[best_verifier] > 0.6:
                policies["verification_precedence"] = (
                    "Verification precedes mutation. Do not modify state without confirming prerequisites."
                )

        # Retry policy
        retry_tendencies = self.get_controller_retry_tendency()
        if retry_tendencies:
            helpful_retries = [
                (ctrl, rate) for ctrl, rate in retry_tendencies.items() if rate > 0.5
            ]
            if helpful_retries:
                policies["retry_ceiling"] = (
                    "Limit retries to 3 per tool. Excessive retries indicate misconfiguration."
                )

        return policies

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics across all controllers."""
        if not self.metrics:
            return {}

        all_metrics = list(self.metrics.values())

        return {
            "controllers_analyzed": len(self.metrics),
            "total_trajectories": sum(m.total_trajectories for m in all_metrics),
            "avg_success_rate": sum(m.success_rate() for m in all_metrics) / len(all_metrics) if all_metrics else 0.0,
            "avg_recovery_success": sum(m.recovery_success_rate() for m in all_metrics if m.recovery_attempts > 0) / len([m for m in all_metrics if m.recovery_attempts > 0]) if [m for m in all_metrics if m.recovery_attempts > 0] else 0.0,
            "controller_metrics": {controller: metrics.to_dict() for controller, metrics in self.metrics.items()},
        }


__all__ = [
    "ControllerAnalyzer",
    "ControllerMetrics",
    "ControllerType",
]
