"""CARGO — Calibrated Action-Risk Gating with Outcome-rollouts.

Public surface (everything callers should import from ``src.cargo``):

* :class:`RiskClass`              — five-class action typing (READ / WRITE / IRREVERSIBLE / FINAL / ASK_USER).
* :class:`ToolEffectSchema`       — auto-induced schema for one tool.
* :class:`ProposedAction`         — output of the proposer.
* :class:`WorkingMemory`          — typed, deterministic state.
* :func:`induce_schemas`          — auto-induce schemas from OpenAI tool specs.
* :data:`SYSTEM_PROMPT`           — proposer system prompt (compact, JSON-only).
* :func:`default_calibration`     — per-class SC thresholds & CF policy.
* :func:`run_cargo`               — ACEBench loop entry point.

The tau-bench :class:`CargoAgent` lives in :mod:`.cargo_agent` and
imports tau-bench lazily so this module can be imported in environments
that don't have tau-bench installed (e.g. unit tests).
"""
from __future__ import annotations

from .ace_loop import run_cargo
from .calibration import CalibrationConfig, default_calibration, fit_thresholds
from .core import (
    BaseCargoAdapter,
    CandidateObject,
    CandidateSelection,
    CandidateSet,
    CommitCertificate,
    Constraint,
    ConstraintPriorityEngine,
    DecisionEngine,
    FallbackRule,
    GoalActionCandidate,
    GoalField,
    GoalFieldDecision,
    GoalHypothesis,
    GenericCargoKernel,
    Obligation,
    Preference,
    ProofObligation,
    SoftGoalFieldRouter,
    TaskState,
)
from .proposer import (
    SYSTEM_PROMPT,
    parse_proposer_response,
    render_proposer_user_message,
    render_tools_block,
)
from .risk_class import RiskClass, is_gated, is_irreversible_or_final
from .schema_inducer import induce_schema, induce_schemas, reset_cache
from .schemas import GateResult, ProposedAction, StepDiagnostics, ToolEffectSchema
from .working_memory import WorkingMemory


__all__ = [
    "CalibrationConfig",
    "BaseCargoAdapter",
    "CandidateObject",
    "CandidateSelection",
    "CandidateSet",
    "CommitCertificate",
    "Constraint",
    "ConstraintPriorityEngine",
    "DecisionEngine",
    "FallbackRule",
    "GateResult",
    "GoalActionCandidate",
    "GoalField",
    "GoalFieldDecision",
    "GoalHypothesis",
    "GenericCargoKernel",
    "Obligation",
    "ProposedAction",
    "Preference",
    "ProofObligation",
    "RiskClass",
    "StepDiagnostics",
    "SoftGoalFieldRouter",
    "SYSTEM_PROMPT",
    "ToolEffectSchema",
    "TaskState",
    "WorkingMemory",
    "default_calibration",
    "fit_thresholds",
    "induce_schema",
    "induce_schemas",
    "is_gated",
    "is_irreversible_or_final",
    "parse_proposer_response",
    "render_proposer_user_message",
    "render_tools_block",
    "reset_cache",
    "run_cargo",
]
