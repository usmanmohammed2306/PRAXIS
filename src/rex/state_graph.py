"""Operational state graph for decision-conditioned retrieval.

The operational state graph tracks the evolving operational state during
a task execution. Unlike the flat working_state.py, this graph structures:

  * tool execution history with outcomes
  * transitions between tools (success, failure, retry, escalation, recovery)
  * failure and recovery records
  * escalation events
  * verification dependencies
  * pending and resolved goals
  * entity and evidence graphs
  * operational phase inference

The graph is rebuilt at every retrieval refresh and discarded after
the trajectory ends (only the distilled card survives).

The graph enables:
  * decision-conditioned retrieval (not just similarity)
  * failure-pattern matching
  * recovery-path reconstruction
  * escalation-readiness inference
  * verification-dependency tracking
  * operational-phase classification
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from datetime import datetime


class TransitionOutcome(str, Enum):
    """Outcome type for tool transitions."""
    SUCCESS = "success"
    FAILURE = "failure"
    RETRY = "retry"
    ESCALATION = "escalation"
    RECOVERY = "recovery"
    VERIFICATION = "verification"


class OperationalPhase(str, Enum):
    """Inferred operational phase during execution."""
    INITIAL = "initial"
    EXPLORATION = "exploration"
    VERIFICATION = "verification"
    RECOVERY = "recovery"
    ESCALATION = "escalation"
    COMPLETION = "completion"


@dataclass
class ToolState:
    """Execution state of a single tool."""
    tool_name: str
    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_executed: Optional[str] = None  # ISO timestamp
    evidence_produced: List[str] = field(default_factory=list)
    error_types: List[str] = field(default_factory=list)  # Last N error types

    def success_rate(self) -> float:
        """Return success rate as decimal [0, 1]."""
        total = self.execution_count
        if total == 0:
            return 0.0
        return self.success_count / total


@dataclass
class Transition:
    """A tool-to-tool transition with outcome and evidence."""
    from_tool: str
    to_tool: str
    outcome: TransitionOutcome
    timestamp: str  # ISO timestamp
    evidence_consumed: List[str] = field(default_factory=list)
    evidence_produced: List[str] = field(default_factory=list)
    reason: Optional[str] = None
    retry_attempt_number: int = 0


@dataclass
class FailureRecord:
    """Record of a tool failure."""
    tool: str
    timestamp: str  # ISO timestamp
    error_type: str
    error_detail: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: Optional[bool] = None
    retry_count: int = 0
    escalation_triggered: bool = False


@dataclass
class EscalationEvent:
    """Record of an escalation decision."""
    timestamp: str  # ISO timestamp
    from_tool: str
    reason: str
    escalated_to: Optional[str] = None
    recovery_path_exhausted: bool = False


@dataclass
class VerificationStep:
    """Record of a verification requirement."""
    requirement: str
    verified: bool
    timestamp: str  # ISO timestamp
    evidence: List[str] = field(default_factory=list)


class OperationalStateGraph:
    """Structured operational state graph for a single trajectory."""

    def __init__(self):
        """Initialize empty state graph."""
        self.tools: Dict[str, ToolState] = {}
        self.transitions: List[Transition] = []
        self.failures: List[FailureRecord] = []
        self.escalations: List[EscalationEvent] = []
        self.verifications: Dict[str, VerificationStep] = {}
        self.pending_goals: List[str] = []
        self.resolved_goals: List[str] = []
        self.entities: Dict[str, Any] = {}
        # evidence_id -> [dependent_evidence_ids/requirements]
        self.evidence_graph: Dict[str, List[str]] = {}
        self._tool_sequence: List[str] = []  # Ordered execution sequence

    def record_tool_execution(
        self,
        tool_name: str,
        outcome: TransitionOutcome,
        evidence_produced: Optional[List[str]] = None,
        error_type: Optional[str] = None,
    ) -> None:
        """Record a tool execution."""
        if tool_name not in self.tools:
            self.tools[tool_name] = ToolState(tool_name=tool_name)

        tool_state = self.tools[tool_name]
        tool_state.execution_count += 1
        tool_state.last_executed = datetime.utcnow().isoformat() + "Z"

        if outcome == TransitionOutcome.SUCCESS:
            tool_state.success_count += 1
        elif outcome == TransitionOutcome.FAILURE:
            tool_state.failure_count += 1
            if error_type:
                tool_state.error_types.append(error_type)
                # Keep last 5 error types
                tool_state.error_types = tool_state.error_types[-5:]
            # Auto-record failure if error_type is provided
            if error_type:
                self.record_failure(tool_name, error_type)

        if evidence_produced:
            tool_state.evidence_produced.extend(evidence_produced)
            # Keep last 20 pieces of evidence
            tool_state.evidence_produced = tool_state.evidence_produced[-20:]

    def record_transition(
        self,
        from_tool: str,
        to_tool: str,
        outcome: TransitionOutcome,
        evidence_consumed: Optional[List[str]] = None,
        evidence_produced: Optional[List[str]] = None,
        reason: Optional[str] = None,
        retry_attempt: int = 0,
    ) -> None:
        """Record a transition between tools."""
        self.transitions.append(
            Transition(
                from_tool=from_tool,
                to_tool=to_tool,
                outcome=outcome,
                timestamp=datetime.utcnow().isoformat() + "Z",
                evidence_consumed=evidence_consumed or [],
                evidence_produced=evidence_produced or [],
                reason=reason,
                retry_attempt_number=retry_attempt,
            )
        )
        self._tool_sequence.append(to_tool)

    def record_failure(
        self,
        tool: str,
        error_type: str,
        error_detail: Optional[str] = None,
    ) -> None:
        """Record a tool failure."""
        self.failures.append(
            FailureRecord(
                tool=tool,
                timestamp=datetime.utcnow().isoformat() + "Z",
                error_type=error_type,
                error_detail=error_detail,
                recovery_attempted=False,
                retry_count=0,
            )
        )

    def record_recovery_attempt(
        self,
        tool: str,
        successful: bool,
    ) -> None:
        """Record a recovery attempt for the last failure of a tool."""
        # Find and update the most recent failure for this tool
        for failure in reversed(self.failures):
            if failure.tool == tool and not failure.recovery_attempted:
                failure.recovery_attempted = True
                failure.recovery_successful = successful
                return

    def record_escalation(
        self,
        from_tool: str,
        reason: str,
        escalated_to: Optional[str] = None,
        recovery_exhausted: bool = False,
    ) -> None:
        """Record an escalation event."""
        self.escalations.append(
            EscalationEvent(
                timestamp=datetime.utcnow().isoformat() + "Z",
                from_tool=from_tool,
                reason=reason,
                escalated_to=escalated_to,
                recovery_path_exhausted=recovery_exhausted,
            )
        )

    def record_verification(
        self,
        requirement: str,
        verified: bool,
        evidence: Optional[List[str]] = None,
    ) -> None:
        """Record a verification step."""
        self.verifications[requirement] = VerificationStep(
            requirement=requirement,
            verified=verified,
            timestamp=datetime.utcnow().isoformat() + "Z",
            evidence=evidence or [],
        )

    def add_pending_goal(self, goal: str) -> None:
        """Add a pending goal."""
        if goal not in self.pending_goals:
            self.pending_goals.append(goal)

    def resolve_goal(self, goal: str) -> None:
        """Mark a goal as resolved."""
        if goal in self.pending_goals:
            self.pending_goals.remove(goal)
        if goal not in self.resolved_goals:
            self.resolved_goals.append(goal)

    def set_entity(self, entity_id: str, entity_data: Any) -> None:
        """Store an entity (e.g., customer info, order details)."""
        self.entities[entity_id] = entity_data

    def get_entity(self, entity_id: str) -> Optional[Any]:
        """Retrieve an entity."""
        return self.entities.get(entity_id)

    def add_evidence_dependency(self, evidence: str, depends_on: List[str]) -> None:
        """Record that evidence depends on other evidence or requirements."""
        self.evidence_graph[evidence] = depends_on

    def get_operational_phase(self) -> OperationalPhase:
        """Infer the current operational phase."""
        # Phase classification logic
        if not self._tool_sequence:
            return OperationalPhase.INITIAL

        has_failures = len(self.failures) > 0
        has_escalations = len(self.escalations) > 0
        unverified_count = sum(
            1 for v in self.verifications.values() if not v.verified
        )
        verified_count = sum(
            1 for v in self.verifications.values() if v.verified
        )
        has_pending_goals = len(self.pending_goals) > 0

        # Decision tree for phase inference
        if has_escalations:
            return OperationalPhase.ESCALATION

        if has_failures and not has_escalations:
            # Check if recovery is in progress
            recent_transitions = self.transitions[-5:]
            recovery_transitions = [
                t for t in recent_transitions
                if t.outcome == TransitionOutcome.RECOVERY
            ]
            if recovery_transitions:
                return OperationalPhase.RECOVERY

        if verified_count > 0 or unverified_count > 0:
            return OperationalPhase.VERIFICATION

        if has_pending_goals or self._tool_sequence:
            if not has_failures:
                return OperationalPhase.EXPLORATION

        if not has_pending_goals and len(self.resolved_goals) > 0:
            return OperationalPhase.COMPLETION

        return OperationalPhase.INITIAL

    def get_recent_transitions(self, count: int = 5) -> List[Transition]:
        """Get the most recent N transitions."""
        return self.transitions[-count:]

    def get_recent_failures(self, count: int = 5) -> List[FailureRecord]:
        """Get the most recent N failures."""
        return self.failures[-count:]

    def get_failed_tools(self) -> List[str]:
        """Get tools that have failed."""
        failed = set()
        for failure in self.failures:
            failed.add(failure.tool)
        return sorted(list(failed))

    def get_successful_tools(self) -> List[str]:
        """Get tools that have succeeded."""
        successful = set()
        for transition in self.transitions:
            if transition.outcome == TransitionOutcome.SUCCESS:
                successful.add(transition.to_tool)
        return sorted(list(successful))

    def infer_retry_count(self, tool: str) -> int:
        """Count retries for a specific tool."""
        retries = sum(
            1 for t in self.transitions
            if t.to_tool == tool and t.outcome == TransitionOutcome.RETRY
        )
        return retries

    def infer_escalation_readiness(self) -> bool:
        """Determine if escalation conditions are met."""
        if not self.failures:
            return False

        # Escalation is ready if:
        # 1. There are unrecovered failures, AND
        # 2. Recovery attempts have been tried and failed
        unrecovered = [
            f for f in self.failures
            if f.recovery_attempted and not f.recovery_successful
        ]
        return len(unrecovered) > 0

    def infer_verification_ready(self) -> bool:
        """Determine if verification requirements are met."""
        # Verification is ready if:
        # 1. Some verifications exist, AND
        # 2. All verifications are satisfied, AND
        # 3. We're proceeding to mutations
        if not self.verifications:
            return False
        all_verified = all(v.verified for v in self.verifications.values())
        return all_verified

    def get_unresolved_verifications(self) -> List[VerificationStep]:
        """Get verification requirements that haven't been met."""
        return [v for v in self.verifications.values() if not v.verified]

    def compute_transition_scores(self) -> Dict[str, float]:
        """Compute confidence scores for each tool transition type."""
        scores = {
            "success": 0.0,
            "failure": 0.0,
            "retry": 0.0,
            "recovery": 0.0,
            "escalation": 0.0,
        }

        total = len(self.transitions)
        if total == 0:
            return scores

        for outcome_type in scores:
            count = sum(
                1 for t in self.transitions
                if t.outcome.value == outcome_type
            )
            scores[outcome_type] = count / total

        return scores

    def get_failure_patterns(self) -> Dict[str, int]:
        """Count failures by error type."""
        patterns: Dict[str, int] = {}
        for failure in self.failures:
            patterns[failure.error_type] = patterns.get(failure.error_type, 0) + 1
        return patterns

    def get_tool_transition_graph(self) -> Dict[str, List[str]]:
        """Get tool-to-tool transition graph."""
        graph: Dict[str, List[str]] = {}
        for transition in self.transitions:
            if transition.from_tool not in graph:
                graph[transition.from_tool] = []
            if transition.to_tool not in graph[transition.from_tool]:
                graph[transition.from_tool].append(transition.to_tool)
        return graph

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "tools": {
                name: {
                    "tool_name": ts.tool_name,
                    "execution_count": ts.execution_count,
                    "success_count": ts.success_count,
                    "failure_count": ts.failure_count,
                    "last_executed": ts.last_executed,
                    "error_types": list(ts.error_types),
                }
                for name, ts in self.tools.items()
            },
            "transitions": [
                {
                    "from_tool": t.from_tool,
                    "to_tool": t.to_tool,
                    "outcome": t.outcome.value,
                    "timestamp": t.timestamp,
                    "reason": t.reason,
                    "retry_attempt_number": t.retry_attempt_number,
                }
                for t in self.transitions
            ],
            "failures": [
                {
                    "tool": f.tool,
                    "timestamp": f.timestamp,
                    "error_type": f.error_type,
                    "error_detail": f.error_detail,
                    "recovery_attempted": f.recovery_attempted,
                    "recovery_successful": f.recovery_successful,
                    "retry_count": f.retry_count,
                    "escalation_triggered": f.escalation_triggered,
                }
                for f in self.failures
            ],
            "escalations": [
                {
                    "timestamp": e.timestamp,
                    "from_tool": e.from_tool,
                    "reason": e.reason,
                    "escalated_to": e.escalated_to,
                    "recovery_path_exhausted": e.recovery_path_exhausted,
                }
                for e in self.escalations
            ],
            "verifications": {
                name: {
                    "requirement": v.requirement,
                    "verified": v.verified,
                    "timestamp": v.timestamp,
                }
                for name, v in self.verifications.items()
            },
            "pending_goals": list(self.pending_goals),
            "resolved_goals": list(self.resolved_goals),
            "entities": self.entities,
            "operational_phase": self.get_operational_phase().value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OperationalStateGraph:
        """Deserialize from dictionary."""
        graph = cls()

        # Restore tools
        for name, tool_data in data.get("tools", {}).items():
            graph.tools[name] = ToolState(
                tool_name=tool_data["tool_name"],
                execution_count=tool_data["execution_count"],
                success_count=tool_data["success_count"],
                failure_count=tool_data["failure_count"],
                last_executed=tool_data["last_executed"],
                error_types=tool_data.get("error_types", []),
            )

        # Restore transitions
        for t_data in data.get("transitions", []):
            graph.transitions.append(
                Transition(
                    from_tool=t_data["from_tool"],
                    to_tool=t_data["to_tool"],
                    outcome=TransitionOutcome(t_data["outcome"]),
                    timestamp=t_data["timestamp"],
                    reason=t_data.get("reason"),
                    retry_attempt_number=t_data.get("retry_attempt_number", 0),
                )
            )

        # Restore failures
        for f_data in data.get("failures", []):
            graph.failures.append(
                FailureRecord(
                    tool=f_data["tool"],
                    timestamp=f_data["timestamp"],
                    error_type=f_data["error_type"],
                    error_detail=f_data.get("error_detail"),
                    recovery_attempted=f_data["recovery_attempted"],
                    recovery_successful=f_data.get("recovery_successful"),
                    retry_count=f_data.get("retry_count", 0),
                    escalation_triggered=f_data.get("escalation_triggered", False),
                )
            )

        # Restore escalations
        for e_data in data.get("escalations", []):
            graph.escalations.append(
                EscalationEvent(
                    timestamp=e_data["timestamp"],
                    from_tool=e_data["from_tool"],
                    reason=e_data["reason"],
                    escalated_to=e_data.get("escalated_to"),
                    recovery_path_exhausted=e_data.get("recovery_path_exhausted", False),
                )
            )

        # Restore verifications
        for name, v_data in data.get("verifications", {}).items():
            graph.verifications[name] = VerificationStep(
                requirement=v_data["requirement"],
                verified=v_data["verified"],
                timestamp=v_data["timestamp"],
                evidence=v_data.get("evidence", []),
            )

        # Restore goals and entities
        graph.pending_goals = data.get("pending_goals", [])
        graph.resolved_goals = data.get("resolved_goals", [])
        graph.entities = data.get("entities", {})

        return graph


__all__ = [
    "OperationalStateGraph",
    "ToolState",
    "Transition",
    "FailureRecord",
    "EscalationEvent",
    "VerificationStep",
    "TransitionOutcome",
    "OperationalPhase",
]
