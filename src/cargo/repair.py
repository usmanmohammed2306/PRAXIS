"""Deterministic repair policy for ABSTAIN events.

When the calibrated gate rejects an action, the agent MUST NOT
silently retry. The repair policy decides exactly one of:

  * ``RETRY`` — re-prompt the proposer with a critique that names the
    failed gate. Cap: ``max_retries`` per environment turn.
  * ``ASK_USER`` — emit a clarifying message to the user via the
    benchmark's ``respond`` tool.
  * ``FINALIZE_GENERIC`` — retry budget exhausted; finalize with a
    safe generic message rather than burn the full step budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .schemas import GateResult, ProposedAction


@dataclass
class RepairDecision:
    action: str          # one of: RETRY | ASK_USER | FINALIZE_GENERIC
    critique: str = ""   # used when action=RETRY
    user_message: str = ""  # used when action=ASK_USER / FINALIZE_GENERIC


def decide(
    failed_gate: GateResult,
    *,
    proposed: Optional[ProposedAction] = None,
    retries_used: int = 0,
    max_retries: int = 2,
    budget_steps_remaining: int = 30,
) -> RepairDecision:
    """Pick a repair action based on the failing gate name."""
    gate = failed_gate.gate
    reason = failed_gate.reason

    # Hard budget exhaustion: just finalize.
    if budget_steps_remaining <= 1:
        return RepairDecision(
            action="FINALIZE_GENERIC",
            user_message=(
                "I'm unable to confidently complete this request right now. "
                "Could you confirm the details and try again?"
            ),
        )

    # Grounding / pre-cond failures → ask the user for the missing fact.
    if gate in ("arg_grounding",):
        ungrounded = failed_gate.diagnostics.get("ungrounded") or []
        ask = (
            "Could you confirm the following so I can proceed? "
            + ", ".join(str(u) for u in ungrounded)
            if ungrounded
            else "Could you provide a bit more detail (e.g. the relevant ID) so I can proceed?"
        )
        return RepairDecision(action="ASK_USER", user_message=ask[:600])

    if gate == "preconditions":
        unmet = failed_gate.diagnostics.get("unmet") or []
        missing_args = failed_gate.diagnostics.get("missing_args") or []
        if missing_args:
            crit = (
                "Required argument(s) missing: "
                + ", ".join(str(a) for a in missing_args)
                + ". Either provide them or pick a different tool."
            )
            if retries_used < max_retries:
                return RepairDecision(action="RETRY", critique=crit)
            return RepairDecision(
                action="ASK_USER",
                user_message="Could you give me more detail so I can answer?",
            )
        if unmet:
            return RepairDecision(
                action="ASK_USER",
                user_message=(
                    "Before I take that action, could you confirm: "
                    + "; ".join(str(u) for u in unmet)
                )[:600],
            )
        return RepairDecision(action="RETRY", critique=f"Pre-conditions not met: {reason}")

    # Self-consistency / counterfactual failures → re-prompt with critique.
    if gate in ("self_consistency", "counterfactual"):
        crit = (
            "The verifier is uncertain about this action.\n"
            f"  - failed gate: {gate}\n"
            f"  - reason: {reason}\n"
        )
        if "alternatives" in failed_gate.diagnostics:
            alts = failed_gate.diagnostics["alternatives"]
            crit += f"  - alternatives the model also considered: {alts}\n"
        if "cf_reason" in failed_gate.diagnostics:
            crit += f"  - counterfactual concern: {failed_gate.diagnostics['cf_reason']}\n"
        crit += (
            "Either gather more information first (a READ tool), or "
            "revise your action / arguments. If the goal is already "
            "complete, finalize with name='respond' and class='FINAL'."
        )
        if retries_used < max_retries:
            return RepairDecision(action="RETRY", critique=crit)
        return RepairDecision(
            action="ASK_USER",
            user_message=(
                "I'm not fully confident in the next step. Could you confirm "
                "the action you'd like me to take?"
            ),
        )

    if gate == "repeat_loop":
        crit = (
            "You proposed the same action you already attempted recently. "
            "Pick a different tool, different arguments, or finalize."
        )
        if retries_used < max_retries:
            return RepairDecision(action="RETRY", critique=crit)
        return RepairDecision(
            action="FINALIZE_GENERIC",
            user_message=(
                "I'm not making further progress with the available "
                "information. Could you clarify what you'd like next?"
            ),
        )

    if gate == "json_parse":
        crit = (
            "Your previous response was not valid JSON. "
            "Output STRICT JSON exactly matching the schema in the system prompt."
        )
        if retries_used < max_retries:
            return RepairDecision(action="RETRY", critique=crit)
        return RepairDecision(
            action="FINALIZE_GENERIC",
            user_message="Sorry — could you rephrase your request?",
        )

    # Fallback: retry once, then ASK_USER.
    if retries_used < max_retries:
        return RepairDecision(
            action="RETRY",
            critique=f"Action rejected by gate '{gate}': {reason}",
        )
    return RepairDecision(
        action="ASK_USER",
        user_message="Could you provide more detail so I can help?",
    )


__all__ = ["RepairDecision", "decide"]
