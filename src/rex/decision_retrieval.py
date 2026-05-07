"""Decision-conditioned retrieval using operational state graph.

Instead of similarity-based retrieval over trajectories, this system
retrieves based on:

  * current operational phase
  * previous tool transitions
  * failure patterns
  * retry state
  * escalation readiness
  * verification state
  * unresolved goals
  * recovery state
  * tool-transition graph patterns
  * multi-hop retrieval paths

The retrieval query is not semantic ("find similar tasks").
The retrieval query is operational ("what should I do given my current state?").

The decision retriever builds operational context vectors and matches them
against memory cards' operational annotations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .memory_types import ProcessMemoryCard
from .state_graph import OperationalStateGraph, OperationalPhase, TransitionOutcome


@dataclass
class OperationalContext:
    """Operational context extracted from current state graph."""

    phase: OperationalPhase
    failed_tools: List[str]
    successful_tools: List[str]
    recent_transitions: List[Tuple[str, str, TransitionOutcome]]  # (from, to, outcome)
    recent_failures: List[Tuple[str, str]]  # (tool, error_type)
    retry_count: int
    escalation_triggered: bool
    escalation_ready: bool
    verification_done: bool
    verification_ready: bool
    unresolved_verifications: int
    pending_goals: List[str]
    resolved_goals: List[str]
    tool_sequence: List[str]
    failure_patterns: Dict[str, int]  # error_type -> count
    transition_scores: Dict[str, float]  # outcome -> proportion


def extract_operational_context(graph: OperationalStateGraph) -> OperationalContext:
    """Extract operational context from state graph."""
    failed_tools = graph.get_failed_tools()
    successful_tools = graph.get_successful_tools()

    recent_transitions = [
        (t.from_tool, t.to_tool, t.outcome)
        for t in graph.get_recent_transitions(count=10)
    ]

    recent_failures = [
        (f.tool, f.error_type)
        for f in graph.get_recent_failures(count=5)
    ]

    unresolved_verifications = len(graph.get_unresolved_verifications())
    verification_done = unresolved_verifications < len(graph.verifications)
    verification_ready = graph.infer_verification_ready()
    escalation_ready = graph.infer_escalation_readiness()

    return OperationalContext(
        phase=graph.get_operational_phase(),
        failed_tools=failed_tools,
        successful_tools=successful_tools,
        recent_transitions=recent_transitions,
        recent_failures=recent_failures,
        retry_count=sum(
            graph.infer_retry_count(tool)
            for tool in set(failed_tools + successful_tools)
        ),
        escalation_triggered=len(graph.escalations) > 0,
        escalation_ready=escalation_ready,
        verification_done=verification_done,
        verification_ready=verification_ready,
        unresolved_verifications=unresolved_verifications,
        pending_goals=graph.pending_goals,
        resolved_goals=graph.resolved_goals,
        tool_sequence=graph._tool_sequence,
        failure_patterns=graph.get_failure_patterns(),
        transition_scores=graph.compute_transition_scores(),
    )


@dataclass
class OperationalRetrievalQuery:
    """Query for decision-conditioned retrieval."""

    operational_context: OperationalContext
    retrieval_mode: str  # "startup", "recovery", "failure", "verification", etc.
    focus_tool: Optional[str] = None  # If targeting specific tool
    focus_error: Optional[str] = None  # If targeting specific error type


class OperationalDecisionRetriever:
    """Retriever that conditions on operational state, not text similarity."""

    def __init__(self, cards: Sequence[ProcessMemoryCard]):
        """Initialize with a corpus of memory cards."""
        self.cards = list(cards)
        self._index_cards()

    def _index_cards(self) -> None:
        """Build indices for operational matching."""
        # Index by task category for quick filtering
        self.by_category: Dict[str, List[ProcessMemoryCard]] = {}
        self.by_outcome: Dict[str, List[ProcessMemoryCard]] = {}

        for card in self.cards:
            category = card.task_category
            if category not in self.by_category:
                self.by_category[category] = []
            self.by_category[category].append(card)

            outcome = card.outcome
            if outcome not in self.by_outcome:
                self.by_outcome[outcome] = []
            self.by_outcome[outcome].append(card)

    def retrieve(
        self,
        query: OperationalRetrievalQuery,
        *,
        top_k: int = 3,
    ) -> List[ProcessMemoryCard]:
        """Retrieve cards based on operational context.

        Implements 10 retrieval features:
        1. Tool-transition-conditioned
        2. Failure-state-conditioned
        3. Retry-aware
        4. Escalation-aware
        5. Goal-state
        6. Verification-state
        7. Recovery-state
        8. Operational-phase
        9. Multi-hop
        10. Transition-pattern

        Returns top_k most relevant cards ranked by operational relevance.
        """
        ctx = query.operational_context
        mode = query.retrieval_mode

        candidates: List[Tuple[ProcessMemoryCard, float]] = []

        for card in self.cards:
            score = 0.0

            # 1. Tool-transition-conditioned retrieval
            # Cards matching tools in recent transitions get higher scores
            for from_tool, to_tool, outcome in ctx.recent_transitions:
                if card.task_category == to_tool:
                    score += 2.0  # High relevance for next tool
                if card.task_category == from_tool:
                    score += 0.5  # Lower relevance for source tool

            # 2. Failure-state-conditioned retrieval
            # If we're in failure state, prioritize recovery/partial cards
            if ctx.recent_failures:
                if card.outcome == "recovery":
                    score += 3.0
                if card.outcome == "partial":
                    score += 1.5
                # Cards with recovery_heuristics are relevant
                if card.recovery_heuristics:
                    score += 2.0

            # 3. Retry-aware retrieval
            # High retry count → prioritize anti-patterns and retry limits
            if ctx.retry_count > 2:
                if card.common_trap:
                    score += 2.0
                if "retry" in str(card.common_trap or "").lower():
                    score += 1.5

            # 4. Escalation-aware retrieval
            # If escalation is ready/triggered, get escalation cards
            if ctx.escalation_ready or ctx.escalation_triggered:
                if "escalat" in str(card.common_trap or "").lower():
                    score += 3.0
                if card.outcome == "avoid":
                    score += 1.5

            # 5. Goal-state retrieval
            # Cards matching unresolved goals
            goal_match = False
            for goal in ctx.pending_goals:
                if goal.lower() in card.task_category.lower():
                    score += 2.0
                    goal_match = True
            if ctx.resolved_goals and goal_match:
                score += 1.0

            # 6. Verification-state retrieval
            # If verification is pending, prioritize verification cards
            if ctx.unresolved_verifications > 0:
                if "verif" in card.task_category.lower():
                    score += 2.5
                if card.required_evidence:
                    score += 1.5
            # If verification is done, deprioritize verification cards
            elif ctx.verification_done:
                if "verif" not in card.task_category.lower():
                    score += 0.5

            # 7. Recovery-state retrieval
            # If recent failures exist, prioritize recovery cards
            if ctx.recent_failures:
                if card.recovery_heuristics:
                    score += 2.5
                if card.failure_patterns:
                    score += 1.5
                # Match failure patterns
                for tool, error_type in ctx.recent_failures:
                    for pattern in card.failure_patterns:
                        if error_type.lower() in pattern.lower():
                            score += 2.0

            # 8. Operational-phase retrieval
            # Prefer cards matching current phase
            phase = ctx.phase
            if phase == OperationalPhase.VERIFICATION:
                if "verif" in card.task_category.lower():
                    score += 2.0
            elif phase == OperationalPhase.RECOVERY:
                if card.outcome == "recovery" or card.recovery_heuristics:
                    score += 2.5
            elif phase == OperationalPhase.ESCALATION:
                if card.outcome == "avoid" or "escalat" in str(card.common_trap or "").lower():
                    score += 2.5
            elif phase == OperationalPhase.EXPLORATION:
                if card.recommended_next_tools:
                    score += 1.5

            # 9. Multi-hop retrieval
            # Cards with sequencing guidance for multi-step recovery
            if ctx.recent_failures and len(ctx.recent_failures) > 1:
                if card.tool_ordering_hints and len(card.tool_ordering_hints) > 1:
                    score += 2.0

            # 10. Transition-pattern retrieval
            # Cards matching recent transition patterns
            if ctx.recent_transitions:
                for from_t, to_t, outcome in ctx.recent_transitions[-3:]:
                    if outcome == TransitionOutcome.SUCCESS:
                        # Positive transitions → look for similar successful patterns
                        if card.outcome == "successful":
                            score += 1.5
                    elif outcome == TransitionOutcome.FAILURE:
                        # Failure transitions → look for recovery/avoidance
                        if card.outcome in ["avoid", "recovery", "partial"]:
                            score += 2.0

            # Retrieval mode-specific adjustments
            if mode == "failure":
                if card.outcome in ["avoid", "partial", "recovery"]:
                    score += 2.0
            elif mode == "recovery":
                if card.recovery_heuristics or card.outcome == "recovery":
                    score += 3.0
            elif mode == "verification":
                if card.required_evidence or "verif" in card.task_category.lower():
                    score += 2.5
            elif mode == "escalation":
                if card.outcome == "avoid" or "escalat" in str(card.common_trap or "").lower():
                    score += 3.0
            elif mode == "anti-pattern":
                if card.common_trap or card.failure_patterns:
                    score += 2.5

            # Focus on specific tool if provided
            if query.focus_tool:
                if card.task_category == query.focus_tool:
                    score += 2.0
                if query.focus_tool in card.recommended_next_tools:
                    score += 1.5

            # Focus on specific error if provided
            if query.focus_error:
                for pattern in card.failure_patterns:
                    if query.focus_error.lower() in pattern.lower():
                        score += 2.5
                if query.focus_error.lower() in str(card.common_trap or "").lower():
                    score += 1.5

            # Penalize cards with same failure pattern if we're trying to recover
            if ctx.recent_failures and card.failure_patterns:
                for tool, error_type in ctx.recent_failures:
                    for pattern in card.failure_patterns:
                        if error_type.lower() in pattern.lower() and card.outcome != "recovery":
                            score -= 1.0

            if score > 0:
                candidates.append((card, score))

        # Sort by score (highest first) and return top_k
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [card for card, _ in candidates[:top_k]]

    def retrieve_for_phase(
        self,
        graph: OperationalStateGraph,
        *,
        top_k: int = 3,
    ) -> List[ProcessMemoryCard]:
        """Retrieve cards for the current operational phase."""
        ctx = extract_operational_context(graph)
        phase = ctx.phase

        # Map phase to retrieval mode
        mode_map = {
            OperationalPhase.INITIAL: "startup",
            OperationalPhase.EXPLORATION: "planning",
            OperationalPhase.VERIFICATION: "verification",
            OperationalPhase.RECOVERY: "recovery",
            OperationalPhase.ESCALATION: "escalation",
            OperationalPhase.COMPLETION: "planning",
        }

        mode = mode_map.get(phase, "planning")

        query = OperationalRetrievalQuery(
            operational_context=ctx,
            retrieval_mode=mode,
        )

        return self.retrieve(query, top_k=top_k)

    def retrieve_for_failure(
        self,
        graph: OperationalStateGraph,
        error_type: Optional[str] = None,
        *,
        top_k: int = 3,
    ) -> List[ProcessMemoryCard]:
        """Retrieve recovery cards for a specific failure."""
        ctx = extract_operational_context(graph)

        query = OperationalRetrievalQuery(
            operational_context=ctx,
            retrieval_mode="failure",
            focus_error=error_type,
        )

        return self.retrieve(query, top_k=top_k)

    def retrieve_for_tool(
        self,
        graph: OperationalStateGraph,
        tool_name: str,
        *,
        top_k: int = 3,
    ) -> List[ProcessMemoryCard]:
        """Retrieve cards relevant to a specific tool."""
        ctx = extract_operational_context(graph)

        query = OperationalRetrievalQuery(
            operational_context=ctx,
            retrieval_mode="planning",
            focus_tool=tool_name,
        )

        return self.retrieve(query, top_k=top_k)


__all__ = [
    "OperationalDecisionRetriever",
    "OperationalContext",
    "OperationalRetrievalQuery",
    "extract_operational_context",
]
