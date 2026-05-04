"""Self-consistency gate: sample k=3 alternative proposals at T=0.7 and
require at least τ_c agreement on the (name, canonical_args) key.

We use the OpenAI ``n=k`` parameter to get all k samples in a single batched
request — this is exactly one extra LLM call per gated step (not k calls).
On τ-bench with vLLM serving Qwen2.5-7B/32B, ``n=3`` adds <30% latency
over a single completion.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..proposer import parse_proposer_response
from ..risk_class import RiskClass
from ..schemas import GateResult, ProposedAction, ToolEffectSchema


def _canonical(action: ProposedAction) -> str:
    try:
        args_canon = json.dumps(action.args, sort_keys=True, default=str)
    except Exception:
        args_canon = str(action.args)
    text = ""
    if action.declared_class in (RiskClass.FINAL, RiskClass.ASK_USER):
        text = f"::{(action.user_text or '').strip()[:300]}"
    return f"{action.name}::{args_canon}{text}"


def check_self_consistency(
    action: ProposedAction,
    schema: ToolEffectSchema,
    *,
    client,
    model: str,
    proposer_messages: List[Dict[str, Any]],
    schemas_for_parse: Optional[Dict[str, ToolEffectSchema]] = None,
    k: int = 3,
    threshold: float = 2.0 / 3.0,
    sc_temperature: float = 0.7,
    max_tokens: int = 600,
) -> GateResult:
    """Run the self-consistency vote.

    The gate passes if at least ``threshold`` fraction of the k samples
    canonicalize to the same (name, args) tuple as ``action``.

    Failure modes that pass-through (return ok=True) so the agent isn't
    blocked by infrastructure:
      * client is None or absent.
      * vLLM rejects ``n``: we fall back to k separate calls.
      * fewer than 1 valid sample parsed.
    """
    if client is None:
        return GateResult.passing("self_consistency", reason="no_client_skipped")
    if k < 1:
        return GateResult.passing("self_consistency", reason="k_zero_skipped")

    samples: List[ProposedAction] = []
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=proposer_messages,
            temperature=sc_temperature,
            n=k,
            max_tokens=max_tokens,
        )
        for ch in (getattr(resp, "choices", []) or []):
            try:
                text = (ch.message.content or "")
            except Exception:
                text = ""
            parsed = parse_proposer_response(text, schemas=schemas_for_parse)
            if parsed is not None:
                samples.append(parsed)
    except Exception as exc:
        # Some providers don't support n>1; fall back to sequential calls.
        try:
            for _ in range(k):
                r2 = client.chat.completions.create(
                    model=model,
                    messages=proposer_messages,
                    temperature=sc_temperature,
                    max_tokens=max_tokens,
                )
                ch = r2.choices[0]
                text = (ch.message.content or "")
                parsed = parse_proposer_response(text, schemas=schemas_for_parse)
                if parsed is not None:
                    samples.append(parsed)
        except Exception as exc2:
            return GateResult.passing(
                "self_consistency",
                reason="sc_call_failed_passthrough",
                exc=str(exc2)[:200],
            )

    if not samples:
        return GateResult.passing(
            "self_consistency",
            reason="no_parsable_samples_passthrough",
        )

    target = _canonical(action)
    matches = sum(1 for s in samples if _canonical(s) == target)
    agreement = matches / len(samples)
    if agreement >= threshold:
        return GateResult.passing(
            "self_consistency",
            agreement=agreement,
            samples=len(samples),
        )
    # Build a compact alt-summary so the repair-policy critique can name
    # what the alternatives looked like.
    alt_keys = [_canonical(s) for s in samples]
    return GateResult.failing(
        "self_consistency",
        f"low_agreement:{agreement:.2f}",
        agreement=agreement,
        samples=len(samples),
        alternatives=alt_keys[:5],
    )


__all__ = ["check_self_consistency"]
