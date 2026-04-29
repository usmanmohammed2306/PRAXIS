"""Counterfactual rollout gate (IRREVERSIBLE / FINAL only).

Asks the LLM, in a single short call, to imagine the result of executing
``action`` and judge whether the agent's *goal* would still be reachable
afterwards. The output is a tiny JSON object; we only use the
``goal_still_reachable`` flag and ABSTAIN on False.

Failure modes that pass-through (return ok=True) so the agent isn't
blocked by infrastructure: client missing, parse failure, network error.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from ..schemas import GateResult, ProposedAction, ToolEffectSchema
from ..working_memory import WorkingMemory


_SYSTEM = (
    "You are a verifier. The agent is about to execute the action below. "
    "Imagine the action's tool result and decide whether the user's GOAL "
    "is still reachable afterwards in <=5 more steps. Output STRICT JSON: "
    "{\"predicted_obs\": str, \"goal_still_reachable\": bool, \"reason\": str}. "
    "Be conservative: if the action looks premature (no DB confirmation of "
    "required entities, missing post-conditions, or contradicts the goal), "
    "set goal_still_reachable=false."
)


_JSON_OBJ_RE = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}", re.DOTALL)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```\s*$", "", s,
               flags=re.IGNORECASE | re.MULTILINE)
    for c in [s] + _JSON_OBJ_RE.findall(s):
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def check_counterfactual(
    action: ProposedAction,
    schema: ToolEffectSchema,
    wm: WorkingMemory,
    *,
    client,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 250,
) -> GateResult:
    if client is None:
        return GateResult.passing("counterfactual", reason="no_client_skipped")

    user = (
        f"GOAL: {wm.goal[:300]}\n"
        f"ACTION:\n  name: {action.name}\n  args: {json.dumps(action.args, default=str)[:600]}\n"
        f"  declared_class: {action.declared_class.value}\n"
        f"  declared_post: {action.declared_post}\n"
        f"WORKING MEMORY (compact):\n{wm.render_compact(max_chars=900)}\n\n"
        "Output JSON only."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = (resp.choices[0].message.content or "")
    except Exception as exc:
        return GateResult.passing(
            "counterfactual",
            reason="cf_call_failed_passthrough",
            exc=str(exc)[:200],
        )
    obj = _extract_json(text)
    if not obj:
        return GateResult.passing(
            "counterfactual",
            reason="cf_unparsable_passthrough",
            raw=text[:200],
        )
    reachable = obj.get("goal_still_reachable")
    if reachable is False:
        return GateResult.failing(
            "counterfactual",
            f"cf_blocks_goal:{str(obj.get('reason', ''))[:160]}",
            predicted_obs=str(obj.get("predicted_obs", ""))[:200],
            cf_reason=str(obj.get("reason", ""))[:200],
        )
    return GateResult.passing("counterfactual", predicted_obs=str(obj.get("predicted_obs", ""))[:200])


__all__ = ["check_counterfactual"]
