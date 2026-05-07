"""Experience retrieval context builder.

Extracts a concise context description from the current execution state
so the agent can retrieve the most relevant prior experiences.

This is purely about answering: "What situation am I in right now, and
what prior experiences should I look for?"

NOT policy reasoning. NOT doctrine. Just:
  current state → relevant experience query.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Dict, Any

from .state_graph import OperationalStateGraph, OperationalPhase
from .working_state import WorkingState


@dataclass
class ExperienceQuery:
    """Compact description of current situation for experience retrieval."""

    # Freetext description of current situation (used for text search)
    situation_text: str

    # Structured signals that bias retrieval toward relevant experiences
    recent_failed_tools: List[str]
    recent_successful_tools: List[str]
    in_recovery: bool          # Agent is trying to recover from failures
    in_escalation: bool        # Escalation has been triggered
    verification_pending: bool # Verification steps outstanding
    high_retry_count: bool     # Too many retries attempted
    environment: str           # "retail" | "airline" | "ace" | "generic"
    controller: str = "rex"    # Which agent is running
    step_index: int = 0


def build_experience_query(
    initial_user: str,
    messages: Sequence[Dict[str, Any]],
    working_state: Optional[WorkingState] = None,
    state_graph: Optional[OperationalStateGraph] = None,
    environment: str = "generic",
    controller: str = "rex",
    step_index: int = 0,
) -> ExperienceQuery:
    """Build an experience query from current execution state.

    Combines the flat working state (fast, always available) with the
    structured state graph (richer, available after Phase 3 upgrade).
    """
    # Defaults from working state
    failed_tools: List[str] = []
    successful_tools: List[str] = []
    in_recovery = False
    in_escalation = False
    verification_pending = False
    high_retry_count = False

    if working_state:
        failed_tools = list(working_state.failed_tools)
        successful_tools = list(working_state.successful_tools)
        in_escalation = working_state.escalation_triggered
        verification_pending = working_state.verification_done is False and bool(successful_tools)
        high_retry_count = working_state.retry_count > 2

    # Override / enrich with state graph if available
    if state_graph:
        graph_failed = state_graph.get_failed_tools()
        graph_successful = state_graph.get_successful_tools()
        if graph_failed:
            failed_tools = graph_failed
        if graph_successful:
            successful_tools = graph_successful
        in_recovery = (
            state_graph.get_operational_phase() == OperationalPhase.RECOVERY
        )
        in_escalation = in_escalation or len(state_graph.escalations) > 0
        verification_pending = len(state_graph.get_unresolved_verifications()) > 0
        total_retries = sum(
            state_graph.infer_retry_count(t)
            for t in set(graph_failed + graph_successful)
        )
        high_retry_count = total_retries > 2

    # Build situation text — compact phrase the retriever can use
    situation_parts: List[str] = [initial_user[:200] if initial_user else ""]
    if failed_tools:
        situation_parts.append(f"failed:{','.join(failed_tools[-3:])}")
    if successful_tools:
        situation_parts.append(f"succeeded:{','.join(successful_tools[-3:])}")
    if in_recovery:
        situation_parts.append("recovery_in_progress")
    if in_escalation:
        situation_parts.append("escalation_triggered")
    if high_retry_count:
        situation_parts.append("high_retry_count")

    situation_text = " ".join(p for p in situation_parts if p)[:400]

    return ExperienceQuery(
        situation_text=situation_text,
        recent_failed_tools=failed_tools[-5:],
        recent_successful_tools=successful_tools[-5:],
        in_recovery=in_recovery,
        in_escalation=in_escalation,
        verification_pending=verification_pending,
        high_retry_count=high_retry_count,
        environment=environment,
        controller=controller,
        step_index=step_index,
    )


def situation_summary(query: ExperienceQuery) -> str:
    """One-line human-readable summary of the query for logging."""
    flags = []
    if query.in_recovery:
        flags.append("recovering")
    if query.in_escalation:
        flags.append("escalated")
    if query.verification_pending:
        flags.append("awaiting-verification")
    if query.high_retry_count:
        flags.append("high-retries")
    if query.recent_failed_tools:
        flags.append(f"failed={','.join(query.recent_failed_tools[:2])}")
    flag_str = f" [{', '.join(flags)}]" if flags else ""
    return f"step={query.step_index} env={query.environment}{flag_str}"


__all__ = [
    "ExperienceQuery",
    "build_experience_query",
    "situation_summary",
]
