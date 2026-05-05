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
        # Detect hallucinated email/credential.
        email_fields = [u for u in ungrounded if "@" in str(u)]
        if email_fields:
            # First, try a RETRY so the auth override in CargoAgent can
            # construct the right tool call from user_facts without consuming
            # a respond turn.  Only fall through to ASK_USER when retries are
            # exhausted or budget is critically low.
            if retries_used < max_retries and budget_steps_remaining > 3:
                crit = (
                    "Your email argument is a placeholder and was blocked.\n"
                    "Check STATE user_facts for the user's name and ZIP code.\n"
                    "  - If BOTH name and zip are present: use "
                    "find_user_id_by_name_zip(first_name=..., last_name=..., zip=...).\n"
                    "  - If name but no zip: use respond+ASK_USER asking for zip.\n"
                    "  - If nothing: use respond+ASK_USER asking for email OR name+zip.\n"
                    "ASK_USER format: {\"name\":\"respond\",\"args\":{},"
                    "\"declared_class\":\"ASK_USER\","
                    "\"user_text\":\"your question here\"}"
                )
                return RepairDecision(action="RETRY", critique=crit)
            ask = (
                "To look up your account, could you please provide your email "
                "address, or your full name and ZIP code?"
            )
            return RepairDecision(action="ASK_USER", user_message=ask[:600])
        elif ungrounded:
            ask = (
                "Could you confirm the following so I can proceed? "
                + ", ".join(str(u) for u in ungrounded)
            )
        else:
            ask = "Could you provide a bit more detail (e.g. the relevant ID) so I can proceed?"
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

    if gate == "confirmation":
        return RepairDecision(
            action="ASK_USER",
            user_message=(
                "Before I make that change, please confirm that you want me "
                "to proceed with the requested update."
            ),
        )

    if gate == "state_validity":
        level = failed_gate.diagnostics.get("validation_level") or ""
        if "read" in str(level):
            crit = (
                "The READ you proposed contradicts a bound semantic slot. "
                "READs are allowed for gathering information, but their route, "
                "date, or other ordinary semantic arguments must match STATE. "
                "Use the grounded values already in STATE or choose a different "
                "READ that fills an open obligation."
            )
        else:
            crit = (
                "The action conflicts with the current task state or a completed "
                "phase. Do not re-enter completed phases, repeat cached reads, "
                "or commit against stale state. Choose the next open obligation."
            )
        if retries_used < max_retries and budget_steps_remaining > 3:
            return RepairDecision(action="RETRY", critique=crit)
        return RepairDecision(
            action="ASK_USER",
            user_message="I need one more grounded detail before I can safely continue.",
        )

    if gate == "completeness":
        crit = (
            "The proposed write is incomplete or does not match the complete "
            "grounded action. Do NOT execute partial solutions. Gather any "
            "missing order/product details first, then propose exactly one "
            "complete write covering all requested items and constraints."
        )
        if retries_used < max_retries and budget_steps_remaining > 3:
            return RepairDecision(action="RETRY", critique=crit)
        return RepairDecision(
            action="ASK_USER",
            user_message=(
                "I do not yet have a complete, verified set of changes for "
                "all requested items. Could you confirm the missing details?"
            ),
        )

    if gate == "final_completeness":
        crit = (
            "The final answer is premature: this task still has unresolved "
            "account/order work. Do not finalize yet. Authenticate if needed, "
            "retrieve the required order/product state, then execute the "
            "complete grounded mutation or ask for the exact missing fact."
        )
        if retries_used < max_retries and budget_steps_remaining > 3:
            return RepairDecision(action="RETRY", critique=crit)
        return RepairDecision(
            action="ASK_USER",
            user_message=(
                "I still need verified account or order details before I can "
                "complete every requested change."
            ),
        )

    if gate == "completed_task":
        return RepairDecision(
            action="FINALIZE_GENERIC",
            user_message="The requested change has already been completed.",
        )

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
        # For respond actions the repeat means we asked the same question
        # twice.  Pushing the model to "pick a different tool" is more
        # actionable than a generic retry message.
        is_respond = (
            proposed is not None
            and proposed.name.lower() in ("respond", "send_user", "finish", "final", "answer")
        )
        if is_respond:
            crit = (
                "You sent the same message to the user again without acting on their reply. "
                "Do NOT repeat the question — instead, use a READ tool "
                "(e.g. get_user, find_order, get_order, search_products) to look up the "
                "required information from the database using the facts already in STATE. "
                "Only use respond again when you have a NEW question or a final answer."
            )
        else:
            crit = (
                "You proposed the same action you already attempted recently. "
                "DO NOT repeat it.\n"
                "  - If you are missing required user info (email, name, zip): "
                "use respond+ASK_USER to request it — NEVER guess or fabricate "
                "values like 'user@example.com'.\n"
                "  - If you have the user's info: choose a DIFFERENT tool or use "
                "different arguments (e.g. find_user_id_by_name_zip instead of "
                "find_user_id_by_email).\n"
                "  - Only finalize if the goal is already complete."
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
