"""Structured procedural reasoning for operational decisions.

NOT chain-of-thought. Instead, structured inference of:
  * operational dependencies
  * missing evidence
  * required verification
  * likely failure risks
  * escalation readiness
  * retry exhaustion
  * recovery opportunities

This module provides deterministic reasoning about operational state
without natural language reasoning chains.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Set

from .state_graph import OperationalStateGraph, OperationalPhase


class RiskLevel(str, Enum):
    """Risk assessment levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class OperationalDependency:
    """A dependency relationship in the operational sequence."""

    dependent: str  # What depends
    required: str   # What is required
    satisfied: bool = False
    evidence: List[str] = None

    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []


@dataclass
class RiskAssessment:
    """Risk assessment for current operational state."""

    risk_level: RiskLevel
    primary_risks: List[str]  # Key risks to address
    mitigation_strategies: List[str]  # How to mitigate
    escalation_readiness: bool
    recovery_possible: bool
    retry_advisable: bool
    retry_ceiling: int  # Max recommended retries


@dataclass
class VerificationRequirement:
    """A verification requirement that must be met."""

    requirement: str
    satisfied: bool
    evidence_needed: List[str]
    evidence_present: List[str]


@dataclass
class ProceduralReasoning:
    """Result of procedural reasoning about current state."""

    phase: str
    dependencies: List[OperationalDependency]
    verification_requirements: List[VerificationRequirement]
    missing_evidence: List[str]
    risk_assessment: RiskAssessment
    next_actions: List[str]
    avoid_actions: List[str]


class ProceduralReasoningEngine:
    """Structured procedural reasoning without chain-of-thought."""

    def __init__(self):
        """Initialize reasoning engine."""
        self._standard_dependencies = self._build_dependency_graph()
        self._verification_templates = self._build_verification_templates()

    def _build_dependency_graph(self) -> dict[str, Set[str]]:
        """Build standard operational dependencies.

        Returns dict mapping action -> set of prerequisites
        """
        return {
            "book_reservation": {"verify_customer", "check_availability"},
            "cancel_reservation": {"verify_customer", "retrieve_reservation"},
            "modify_reservation": {"verify_customer", "retrieve_reservation", "verify_changes"},
            "return_items": {"verify_customer", "verify_order", "verify_return_eligible"},
            "exchange_items": {"verify_customer", "verify_order", "verify_exchange_eligible"},
            "process_refund": {"verify_customer", "verify_order", "calculate_refund"},
            "transfer_to_human": {"collect_context"},
        }

    def _build_verification_templates(self) -> dict[str, List[VerificationRequirement]]:
        """Build standard verification templates for common scenarios."""
        return {
            "customer_lookup": [
                VerificationRequirement(
                    requirement="Customer exists",
                    satisfied=False,
                    evidence_needed=["customer_id", "customer_name"],
                    evidence_present=[],
                ),
                VerificationRequirement(
                    requirement="Customer account is active",
                    satisfied=False,
                    evidence_needed=["account_status"],
                    evidence_present=[],
                ),
            ],
            "order_lookup": [
                VerificationRequirement(
                    requirement="Order exists",
                    satisfied=False,
                    evidence_needed=["order_id"],
                    evidence_present=[],
                ),
                VerificationRequirement(
                    requirement="Order belongs to customer",
                    satisfied=False,
                    evidence_needed=["order_customer_match"],
                    evidence_present=[],
                ),
            ],
            "mutation": [
                VerificationRequirement(
                    requirement="User has confirmed the action",
                    satisfied=False,
                    evidence_needed=["user_confirmation"],
                    evidence_present=[],
                ),
                VerificationRequirement(
                    requirement="All prerequisites verified",
                    satisfied=False,
                    evidence_needed=["verification_complete"],
                    evidence_present=[],
                ),
            ],
        }

    def infer_dependencies(
        self,
        state_graph: OperationalStateGraph,
        proposed_tool: Optional[str] = None,
    ) -> List[OperationalDependency]:
        """Infer operational dependencies given current state."""
        dependencies: List[OperationalDependency] = []

        # If proposing a tool, check standard dependencies
        if proposed_tool and proposed_tool in self._standard_dependencies:
            required_actions = self._standard_dependencies[proposed_tool]
            completed_tools = set(state_graph.get_successful_tools())

            for required in required_actions:
                satisfied = required in completed_tools
                dependencies.append(
                    OperationalDependency(
                        dependent=proposed_tool,
                        required=required,
                        satisfied=satisfied,
                    )
                )

        return dependencies

    def identify_missing_evidence(
        self,
        state_graph: OperationalStateGraph,
    ) -> List[str]:
        """Identify what evidence is missing for current goals."""
        missing: List[str] = []

        # Check what verifications are unresolved
        for verification in state_graph.get_unresolved_verifications():
            if not verification.verified:
                missing.append(f"Verify: {verification.requirement}")

        # Check if entities are missing
        if not state_graph.entities.get("customer"):
            missing.append("Customer identity missing")
        if not state_graph.entities.get("order"):
            missing.append("Order information missing")

        return missing

    def assess_risks(
        self,
        state_graph: OperationalStateGraph,
    ) -> RiskAssessment:
        """Assess operational risks in current state."""
        risks: List[str] = []
        mitigations: List[str] = []
        escalation_ready = state_graph.infer_escalation_readiness()
        recovery_possible = len(state_graph.get_failed_tools()) > 0
        retry_ceiling = 3
        risk_level = RiskLevel.LOW

        # Risk: Recent failures
        recent_failures = state_graph.get_recent_failures(count=3)
        if recent_failures:
            risks.append("Recent tool failures detected")
            mitigations.append("Review failure patterns before retrying")
            risk_level = RiskLevel.MEDIUM

        # Risk: High retry count
        total_retries = sum(
            state_graph.infer_retry_count(tool)
            for tool in state_graph.get_failed_tools()
        )
        if total_retries >= retry_ceiling:
            risks.append(f"Retry ceiling ({retry_ceiling}) approaching or exceeded")
            mitigations.append("Consider escalation or alternative strategy")
            risk_level = RiskLevel.HIGH

        # Risk: Escalation triggered
        if state_graph.escalations:
            risks.append("Escalation has been triggered")
            mitigations.append("Prepare for human handoff")
            risk_level = RiskLevel.CRITICAL

        # Risk: Unresolved verifications
        unverified = len(state_graph.get_unresolved_verifications())
        if unverified > 0:
            risks.append(f"{unverified} unresolved verifications")
            mitigations.append("Complete verification before mutation")
            if not risk_level == RiskLevel.CRITICAL:
                risk_level = RiskLevel.HIGH

        # Risk: Unknown state
        if not state_graph.tools and not state_graph.entities:
            risks.append("Minimal operational context")
            mitigations.append("Perform initial lookup/verification")
            risk_level = RiskLevel.MEDIUM

        return RiskAssessment(
            risk_level=risk_level,
            primary_risks=risks,
            mitigation_strategies=mitigations,
            escalation_readiness=escalation_ready,
            recovery_possible=recovery_possible,
            retry_advisable=total_retries < retry_ceiling,
            retry_ceiling=retry_ceiling,
        )

    def infer_verification_requirements(
        self,
        state_graph: OperationalStateGraph,
        scenario: str = "mutation",
    ) -> List[VerificationRequirement]:
        """Infer verification requirements for a scenario."""
        if scenario not in self._verification_templates:
            return []

        requirements = []
        for template_req in self._verification_templates[scenario]:
            # Check if requirement is already satisfied
            satisfied = template_req.requirement in state_graph.verifications and \
                       state_graph.verifications[template_req.requirement].verified

            req = VerificationRequirement(
                requirement=template_req.requirement,
                satisfied=satisfied,
                evidence_needed=template_req.evidence_needed,
                evidence_present=(
                    state_graph.verifications[template_req.requirement].evidence
                    if satisfied else []
                ),
            )
            requirements.append(req)

        return requirements

    def reason_about_state(
        self,
        state_graph: OperationalStateGraph,
        proposed_tool: Optional[str] = None,
    ) -> ProceduralReasoning:
        """Perform complete procedural reasoning about operational state."""
        phase = state_graph.get_operational_phase()

        # Infer dependencies
        dependencies = self.infer_dependencies(state_graph, proposed_tool)

        # Infer verification requirements
        verification_reqs = self.infer_verification_requirements(
            state_graph,
            scenario="mutation" if proposed_tool else "customer_lookup",
        )

        # Identify missing evidence
        missing = self.identify_missing_evidence(state_graph)

        # Assess risks
        risk = self.assess_risks(state_graph)

        # Determine next actions
        next_actions: List[str] = []
        avoid_actions: List[str] = []

        if phase == OperationalPhase.INITIAL:
            next_actions.append("Perform customer lookup")
            next_actions.append("Retrieve relevant order/reservation")
        elif phase == OperationalPhase.EXPLORATION:
            if missing:
                next_actions.append("Collect missing evidence")
            if proposed_tool:
                unsatisfied_deps = [d for d in dependencies if not d.satisfied]
                if unsatisfied_deps:
                    next_actions.extend([f"Complete: {d.required}" for d in unsatisfied_deps])
                else:
                    next_actions.append(f"Execute: {proposed_tool}")
            else:
                next_actions.append("Continue with next logical step")
        elif phase == OperationalPhase.VERIFICATION:
            next_actions.append("Complete verification requirements")
            avoid_actions.append("Do not mutate state without verification")
        elif phase == OperationalPhase.RECOVERY:
            next_actions.append("Execute recovery strategy")
            if not risk.retry_advisable:
                avoid_actions.append("Do not retry; escalation may be needed")
        elif phase == OperationalPhase.ESCALATION:
            next_actions.append("Prepare for escalation to human agent")
            avoid_actions.append("Do not continue retrying failed paths")

        return ProceduralReasoning(
            phase=phase.value,
            dependencies=dependencies,
            verification_requirements=verification_reqs,
            missing_evidence=missing,
            risk_assessment=risk,
            next_actions=next_actions,
            avoid_actions=avoid_actions,
        )


__all__ = [
    "ProceduralReasoningEngine",
    "ProceduralReasoning",
    "RiskAssessment",
    "OperationalDependency",
    "VerificationRequirement",
    "RiskLevel",
]
