"""Structured runtime working memory for a single trajectory.

The working state is rebuilt at every retrieval refresh from the current
conversation. It is *not* persistent — once the trajectory ends, the
state is discarded (only the distilled card survives).

Tracked categories:

  * verified facts — observations confirmed by tool returns
  * attempted tools / failed tools — used to bias recovery retrieval
  * retrieved entities — sanitized snippets that look like ids/names
  * escalation state, verification state, retry state
  * outstanding goals / completed goals — goal tracking
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .sanitize import sanitize_text


_ERROR_KEYWORDS = ("error", "not found", "invalid", "failed", "denied", "unavailable")
_VERIFY_PREFIXES = ("get_", "find_", "search_", "list_", "lookup", "view_", "fetch_", "read_")
_MUTATING_PREFIXES = (
    "cancel", "modify", "return", "exchange", "book", "update", "send",
    "transfer", "delete", "create", "add", "remove",
)


def _looks_verifier(name: str) -> bool:
    n = (name or "").lower()
    return any(n.startswith(p) for p in _VERIFY_PREFIXES)


def _looks_mutating(name: str) -> bool:
    n = (name or "").lower()
    return any(n.startswith(p) for p in _MUTATING_PREFIXES)


@dataclass
class WorkingState:
    """Structured snapshot of what the agent currently knows about the task."""

    initial_user: str = ""
    last_user: str = ""
    verified_facts: List[str] = field(default_factory=list)
    attempted_tools: List[str] = field(default_factory=list)
    failed_tools: List[str] = field(default_factory=list)
    successful_tools: List[str] = field(default_factory=list)
    retrieved_entities: List[str] = field(default_factory=list)
    escalation_triggered: bool = False
    verification_done: bool = False
    retry_count: int = 0
    outstanding_goals: List[str] = field(default_factory=list)
    completed_goals: List[str] = field(default_factory=list)

    def summary_for_query(self) -> str:
        """Return a compact string used in retrieval queries."""
        parts: List[str] = []
        if self.last_user and self.last_user != self.initial_user:
            parts.append(self.last_user)
        if self.failed_tools:
            parts.append("failed:" + ",".join(self.failed_tools[-3:]))
        if self.successful_tools:
            parts.append("succeeded:" + ",".join(self.successful_tools[-3:]))
        if self.escalation_triggered:
            parts.append("escalated")
        return sanitize_text(" ".join(parts), limit=320)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_user": self.initial_user,
            "last_user": self.last_user,
            "verified_facts": list(self.verified_facts),
            "attempted_tools": list(self.attempted_tools),
            "failed_tools": list(self.failed_tools),
            "successful_tools": list(self.successful_tools),
            "retrieved_entities": list(self.retrieved_entities),
            "escalation_triggered": self.escalation_triggered,
            "verification_done": self.verification_done,
            "retry_count": self.retry_count,
            "outstanding_goals": list(self.outstanding_goals),
            "completed_goals": list(self.completed_goals),
        }


def update_working_state(
    state: WorkingState,
    messages: Sequence[Dict[str, Any]],
    *,
    initial_user: Optional[str] = None,
) -> WorkingState:
    """Rebuild fields of ``state`` from the current message log.

    The function is idempotent: passing the same ``messages`` twice yields
    identical state.
    """
    if initial_user is not None:
        state.initial_user = sanitize_text(initial_user, limit=360)

    last_user = ""
    attempted: List[str] = []
    failed: List[str] = []
    succeeded: List[str] = []
    verified_facts: List[str] = []
    retrieved_entities: List[str] = []
    pending_call: Optional[str] = None
    escalated = False
    verification_done = False
    retry = 0
    last_tool_called: Optional[str] = None
    last_tool_args: Optional[str] = None

    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "user":
            content = m.get("content")
            if content:
                last_user = sanitize_text(content, limit=360)
        elif role == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") if isinstance(tc, dict) else None
                if isinstance(fn, dict):
                    name = str(fn.get("name") or "")
                    args = str(fn.get("arguments") or "")
                    pending_call = name
                    attempted.append(name)
                    if _looks_verifier(name):
                        verification_done = True
                    if "transfer_to_human" in name:
                        escalated = True
                    if last_tool_called == name and last_tool_args == args:
                        retry += 1
                    last_tool_called = name
                    last_tool_args = args
        elif role == "tool":
            tool_name = str(m.get("name") or "")
            content = sanitize_text(m.get("content") or "", limit=320)
            low = content.lower()
            if any(k in low for k in _ERROR_KEYWORDS):
                if pending_call:
                    failed.append(pending_call)
            else:
                if pending_call:
                    succeeded.append(pending_call)
                if content:
                    verified_facts.append(f"{tool_name}: {content[:160]}")
                    # Surface anything that *looks* like a placeholder identifier
                    # the redactor inserted (these are safe by construction).
                    for m2 in re.findall(r"<[A-Z_]+>", content):
                        if m2 not in retrieved_entities:
                            retrieved_entities.append(m2)
            pending_call = None

    state.last_user = last_user
    state.attempted_tools = attempted
    state.failed_tools = failed
    state.successful_tools = succeeded
    state.verified_facts = verified_facts[-12:]
    state.retrieved_entities = retrieved_entities[-12:]
    state.escalation_triggered = escalated
    state.verification_done = verification_done
    state.retry_count = retry
    return state


def working_state_for_messages(
    messages: Sequence[Dict[str, Any]],
    *,
    initial_user: str = "",
) -> WorkingState:
    return update_working_state(WorkingState(initial_user=sanitize_text(initial_user, limit=360)), messages)


__all__ = [
    "WorkingState",
    "update_working_state",
    "working_state_for_messages",
]
