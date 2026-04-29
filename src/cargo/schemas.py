"""Dataclasses used across the CARGO agent: tool schemas, proposed actions, gate results."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .risk_class import RiskClass


@dataclass
class ToolEffectSchema:
    """Auto-induced schema for a single tool.

    Built once per tool (cached by name + signature) at agent startup. The
    LLM-induced version is preferred when available; falls back to a
    rule-based classifier on the tool name + parameter shape so the agent
    is robust to induction failures.
    """
    name: str
    cls: RiskClass
    irreversible: bool = False
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    # Param names whose values look like opaque IDs (regex-grounded).
    arg_id_fields: List[str] = field(default_factory=list)
    # Raw parameter property dict from the OpenAI schema (for type lookups).
    param_properties: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Required parameter names (for missing-arg checks).
    required_params: List[str] = field(default_factory=list)
    # Free-form description; useful for the proposer prompt.
    description: str = ""


@dataclass
class ProposedAction:
    """One step's output from the proposer LLM call.

    All fields are optional except ``name``: a missing or malformed JSON
    response degrades gracefully (the agent ABSTAINs and re-prompts).
    """
    name: str
    args: Dict[str, Any] = field(default_factory=dict)
    declared_class: RiskClass = RiskClass.READ
    declared_pre: List[str] = field(default_factory=list)
    declared_post: List[str] = field(default_factory=list)
    informational_intent: str = ""
    raw_thought: str = ""
    # Free-form text for FINAL / ASK_USER actions (the user-facing message).
    user_text: str = ""
    # Convenience: the raw JSON or text that produced this action.
    raw_response: str = ""

    def signature(self) -> str:
        """Canonical (name, sorted args) signature for repeat detection."""
        try:
            parts = [f"{k}={self.args[k]!r}" for k in sorted(self.args.keys())]
        except Exception:
            parts = []
        return self.name + "(" + ",".join(parts) + ")"


@dataclass
class GateResult:
    """Outcome of a single gate's check on a proposed action."""
    ok: bool
    gate: str
    reason: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def passing(cls, gate: str, **diag: Any) -> "GateResult":
        return cls(ok=True, gate=gate, reason="ok", diagnostics=dict(diag))

    @classmethod
    def failing(cls, gate: str, reason: str, **diag: Any) -> "GateResult":
        return cls(ok=False, gate=gate, reason=reason, diagnostics=dict(diag))


@dataclass
class StepDiagnostics:
    """Per-step diagnostic record written into ``info.cargo_stats`` aggregates."""
    step_idx: int
    declared_class: str = ""
    proposer_thought: str = ""
    action_name: str = ""
    action_args: Dict[str, Any] = field(default_factory=dict)
    gates_run: List[str] = field(default_factory=list)
    gates_failed: List[str] = field(default_factory=list)
    abstain_reason: str = ""
    repair_action: str = ""
    sc_agreement: Optional[float] = None
    cf_predicted_blocking: Optional[bool] = None
    fast_path: bool = False


__all__ = [
    "ToolEffectSchema",
    "ProposedAction",
    "GateResult",
    "StepDiagnostics",
]
