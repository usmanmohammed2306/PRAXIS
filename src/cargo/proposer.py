"""CARGO proposer — single LLM call per step, returns a ProposedAction.

The proposer prompt is intentionally compact so a 7B–32B model can hit
JSON-only output reliably. We do NOT use the OpenAI tool-calling API
directly here, because we need the model to also declare its risk class
and pre/post-conditions alongside the action; that metadata doesn't fit
the OpenAI tool-call schema. Robust JSON extraction handles the common
failure modes (code-fenced output, leading/trailing prose).
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .risk_class import RiskClass
from .schemas import ProposedAction, ToolEffectSchema
from .working_memory import WorkingMemory


SYSTEM_PROMPT = (
    "You are a tool-using agent operating under a deterministic verifier. "
    "Each action you propose will be inspected before execution. If the "
    "verifier finds your action cites unverified facts, ungrounded IDs, or "
    "violates a post-condition you declared, it will REJECT your action and "
    "ask you to revise. Therefore: declare your preconditions and "
    "post-conditions explicitly, and only mutate state when you are confident.\n"
    "\n"
    "Risk classes:\n"
    "- READ: pure information retrieval (get_/search_/list_).\n"
    "- WRITE: modifies state but recoverable in-domain.\n"
    "- IRREVERSIBLE: cancel/refund/charge/send — no undo in-domain.\n"
    "- FINAL: terminal answer to the user (use tool name 'respond').\n"
    "- ASK_USER: ask the user for a missing fact (use tool name 'respond').\n"
    "\n"
    "Output STRICT JSON only — no markdown, no commentary outside the JSON.\n"
    "Schema:\n"
    "{\n"
    "  \"thought\": str,\n"
    "  \"action\": {\n"
    "    \"name\": str,\n"
    "    \"args\": object,\n"
    "    \"declared_class\": one of [READ, WRITE, IRREVERSIBLE, FINAL, ASK_USER],\n"
    "    \"declared_pre\": [str, ...],\n"
    "    \"declared_post\": [str, ...],\n"
    "    \"informational_intent\": str,\n"
    "    \"user_text\": str   // only for FINAL or ASK_USER\n"
    "  }\n"
    "}\n"
)


def render_tools_block(schemas: Dict[str, ToolEffectSchema], max_chars: int = 4000) -> str:
    """Compact NL render of available tools + their auto-induced classes."""
    lines: List[str] = ["Tools available (with auto-induced risk classes):"]
    for name in sorted(schemas.keys()):
        sch = schemas[name]
        req = ",".join(sch.required_params) if sch.required_params else "-"
        desc = sch.description.strip().splitlines()[0] if sch.description else ""
        line = f"- {name} [{sch.cls.value}] required={req}"
        if desc:
            line += f" — {desc[:100]}"
        lines.append(line)
    text = "\n".join(lines)
    return text[:max_chars]


def render_proposer_user_message(
    *,
    wm: WorkingMemory,
    tools_block: str,
    history_tail: str = "",
    critique: str = "",
    domain_policy: str = "",
) -> str:
    """Assemble the per-step user message for the proposer."""
    parts: List[str] = []
    if domain_policy:
        parts.append("--- Domain policy ---\n" + domain_policy.strip())
    parts.append(tools_block)
    parts.append("--- Working memory ---\n" + wm.render_compact())
    if history_tail:
        parts.append("--- Recent turns ---\n" + history_tail.strip())
    if critique:
        parts.append(
            "--- Verifier critique (your previous proposal was rejected) ---\n"
            + critique.strip()
            + "\nRevise your action to satisfy the failed gate."
        )
    parts.append(
        "Return ONLY the JSON described in the system prompt. "
        "If you have enough DB-confirmed facts to finalize, choose action "
        "name='respond' with declared_class='FINAL' and put the user-facing "
        "text in 'user_text'. If you need a fact you don't have, choose "
        "action name='respond' with declared_class='ASK_USER' and put your "
        "question in 'user_text'."
    )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------
_JSON_OBJ_RE = re.compile(r"\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}", re.DOTALL)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```\s*$", "", s,
               flags=re.IGNORECASE | re.MULTILINE)
    candidates: List[str] = [s]
    candidates.extend(_JSON_OBJ_RE.findall(s))
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def parse_proposer_response(
    text: str,
    *,
    schemas: Optional[Dict[str, ToolEffectSchema]] = None,
) -> Optional[ProposedAction]:
    """Parse a proposer response into a :class:`ProposedAction`.

    Returns ``None`` only when the response is *completely* unparseable.
    Best-effort parsing applies — missing fields fall back to safe defaults.
    """
    obj = _extract_json(text)
    if not obj:
        return None
    thought = str(obj.get("thought", ""))[:600]
    raw_action = obj.get("action")
    if not isinstance(raw_action, dict) or not raw_action:
        # Permit a flat schema where the action lives at the top level.
        # Triggers when the model emits e.g. {"name":"x","args":{}} without
        # the {"action":{...}} wrapper.
        raw_action = obj
    name = raw_action.get("name") or raw_action.get("tool") or raw_action.get("function")
    if not isinstance(name, str) or not name.strip():
        return None
    args = raw_action.get("args") or raw_action.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}
    if not isinstance(args, dict):
        args = {}
    declared_class = RiskClass.parse(
        raw_action.get("declared_class")
        or raw_action.get("class")
        or raw_action.get("risk_class")
    )
    pre = raw_action.get("declared_pre") or raw_action.get("preconditions") or []
    post = raw_action.get("declared_post") or raw_action.get("postconditions") or []
    if not isinstance(pre, list):
        pre = []
    if not isinstance(post, list):
        post = []
    info = str(raw_action.get("informational_intent")
               or raw_action.get("intent") or "")[:300]
    user_text = str(raw_action.get("user_text")
                    or raw_action.get("content")
                    or raw_action.get("message") or "")[:1200]

    # If the LLM picked the "respond" tool but didn't declare FINAL/ASK_USER,
    # default to FINAL — this is the safe terminal class.
    nl = name.strip().lower()
    if nl in ("respond", "send_user", "final", "finish", "answer"):
        if declared_class not in (RiskClass.FINAL, RiskClass.ASK_USER):
            declared_class = RiskClass.FINAL

    # If a tool schema exists and the model's declared class is suspiciously
    # off, default to the schema's class (the schema is auto-induced and
    # generally correct on prefixed tool names).
    if schemas and name.strip() in schemas:
        sch = schemas[name.strip()]
        if declared_class == RiskClass.READ and sch.cls in (
            RiskClass.WRITE, RiskClass.IRREVERSIBLE
        ):
            declared_class = sch.cls

    return ProposedAction(
        name=name.strip(),
        args={k: v for k, v in args.items() if v is not None},
        declared_class=declared_class,
        declared_pre=[str(x)[:200] for x in pre if str(x).strip()][:8],
        declared_post=[str(x)[:200] for x in post if str(x).strip()][:8],
        informational_intent=info,
        raw_thought=thought,
        user_text=user_text,
        raw_response=text[:4000],
    )


def trim_history(messages: List[Dict[str, Any]], n_turns: int = 8,
                 max_chars: int = 1800) -> str:
    """Render the tail of the message history for the proposer prompt."""
    tail = messages[-n_turns:] if len(messages) > n_turns else messages
    parts: List[str] = []
    for m in tail:
        role = m.get("role", "?")
        content = m.get("content")
        if not content:
            tcs = m.get("tool_calls") or []
            if tcs:
                names = ",".join(tc.get("function", {}).get("name", "?") for tc in tcs)
                content = f"[tool_calls: {names}]"
        text = str(content or "")[:300]
        parts.append(f"{role}: {text}")
    out = "\n".join(parts)
    return out[:max_chars]


__all__ = [
    "SYSTEM_PROMPT",
    "render_tools_block",
    "render_proposer_user_message",
    "parse_proposer_response",
    "trim_history",
]
