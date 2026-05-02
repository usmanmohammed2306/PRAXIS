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
    "ANTI-HALLUCINATION (critical — violations are caught and blocked):\n"
    "- NEVER fabricate argument values. Only use values the user explicitly\n"
    "  stated or a previous tool call returned.\n"
    "- If you need a credential (email, name, zip code) the user has not yet\n"
    "  provided, use name='respond' + declared_class='ASK_USER' to ask for it.\n"
    "- Placeholder values like 'user@example.com' are NEVER valid tool arguments.\n"
    "\n"
    "CONTEXT DISCIPLINE (follow strictly to stay within token budget):\n"
    "- declared_pre / declared_post: SHORT PHRASES only, ≤8 words each, max 3 items.\n"
    "  Good: \"order exists\", \"user confirmed cancellation\"\n"
    "  Bad:  \"The order with the given ID has been confirmed to exist in the database\"\n"
    "- thought: one sentence maximum.\n"
    "- informational_intent: ≤10 words.\n"
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
        # Tip 7: short phrases — 80 chars/item, max 4 items each.
        declared_pre=[str(x)[:80] for x in pre if str(x).strip()][:4],
        declared_post=[str(x)[:80] for x in post if str(x).strip()][:4],
        informational_intent=info,
        raw_thought=thought,
        user_text=user_text,
        raw_response=text[:4000],
    )


# ---------------------------------------------------------------------------
# History compression helpers (tips 3, 4, 5)
# ---------------------------------------------------------------------------

def _compress_args(args: Dict[str, Any], max_total: int = 120) -> str:
    """Render tool args as 'k=v, k=v', truncated to max_total chars."""
    parts: List[str] = []
    for k, v in args.items():
        sv = str(v)
        if len(sv) > 40:
            sv = sv[:37] + "..."
        parts.append(f"{k}={sv}")
    return ", ".join(parts)[:max_total]


def _compress_assistant(content: str) -> str:
    """Compress a raw proposer JSON assistant message to 'name(k=v, …)'.

    The full proposer JSON contains thought, declared_pre/post, and
    informational_intent — none of which belong in history (tips 3, 6).
    Only the action name and args are kept.
    """
    if not content:
        return ""
    try:
        obj = json.loads(content.strip())
        action = obj.get("action") or {}
        name = str(action.get("name") or obj.get("name") or "")
        args = action.get("args") or obj.get("args") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        if name:
            return f"{name}({_compress_args(args, 80)})"
    except Exception:
        pass
    # Fallback: strip to first 80 chars of raw text.
    return content.strip()[:80]


def _compress_tool_obs(content: str) -> str:
    """Compress a tool-result JSON to canonical 'key: val' one-liners (tip 4, 5).

    Only scalar top-level values are included; nested dicts are flattened one
    level; lists are shown as counts.  This keeps observations within ~100–150
    chars while preserving every ID and status field.
    """
    if not content:
        return ""
    s = content.strip()
    if not s or s[0] not in "{[":
        return s[:150]
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            if not obj:
                return "[]"
            # Flatten first item + count.
            first = obj[0] if isinstance(obj[0], dict) else {}
            pairs = [f"{k}: {str(v)[:40]}" for k, v in first.items()
                     if isinstance(v, (str, int, float, bool))][:4]
            suffix = f" (+{len(obj)-1} more)" if len(obj) > 1 else ""
            return (", ".join(pairs) + suffix)[:150]
        if isinstance(obj, dict):
            pairs: List[str] = []
            for k, v in obj.items():
                if isinstance(v, (str, int, float, bool)) and v not in (None, ""):
                    sv = str(v)
                    pairs.append(f"{k}: {sv[:50]}")
                elif isinstance(v, dict):
                    # Flatten one level (e.g. "order": {"status": "X"} → status: X)
                    for ik, iv in v.items():
                        if isinstance(iv, (str, int, float, bool)):
                            pairs.append(f"{ik}: {str(iv)[:50]}")
                elif isinstance(v, list):
                    pairs.append(f"{k}: [{len(v)} items]")
            return ", ".join(pairs[:10])[:200]
    except Exception:
        pass
    return s[:150]


def trim_history(messages: List[Dict[str, Any]], n_turns: int = 8,
                 max_chars: int = 800) -> str:
    """Render a compact tail of the message history for the proposer prompt.

    Context-optimization rules applied here (tips 2, 3, 4, 6):
    - Older messages (beyond n_turns) are collapsed into a one-line action
      digest so the proposer knows what has already been tried.
    - Assistant messages: raw proposer JSON is stripped to just
      ``action(args)`` — thought, pre/post, and intent are dropped.
    - Tool messages: JSON observations are flattened to ``key: val`` pairs.
    - User messages: kept verbatim up to 200 chars.
    - System messages: skipped (already in the system prompt).

    Target: the returned string fits comfortably within ~300 tokens (tip 10).
    """
    # Separate the "detail" window (recent) from "digest" (older).
    # Each complete step adds roughly 3 messages (proposer JSON + tool_calls +
    # tool result), so n_turns=8 covers ~2-3 full steps in detail.
    if len(messages) > n_turns:
        detail_msgs = messages[-n_turns:]
        digest_msgs = messages[:-n_turns]
    else:
        detail_msgs = messages
        digest_msgs = []

    parts: List[str] = []

    # One-line digest of prior actions (tip 2: summarize older steps).
    if digest_msgs:
        prior_actions: List[str] = []
        for m in digest_msgs:
            if m.get("role") == "assistant":
                # Prefer tool_calls field (post-execution); else parse proposer JSON.
                tcs = m.get("tool_calls") or []
                if tcs:
                    for tc in tcs:
                        nm = tc.get("function", {}).get("name", "")
                        if nm and nm not in ("respond",):
                            prior_actions.append(nm)
                else:
                    name = _compress_assistant(m.get("content") or "")
                    # _compress_assistant returns "name(args)" — grab just "name".
                    nm = name.split("(")[0] if name else ""
                    if nm:
                        prior_actions.append(nm)
        if prior_actions:
            parts.append("[prior: " + ", ".join(prior_actions[-8:]) + "]")

    # Render the detail window with per-role compression.
    for m in detail_msgs:
        role = m.get("role", "?")
        if role == "system":
            continue  # system prompt already injected separately
        if role == "assistant":
            tcs = m.get("tool_calls") or []
            if tcs:
                for tc in tcs:
                    nm = tc.get("function", {}).get("name", "?")
                    args_raw = tc.get("function", {}).get("arguments", "{}")
                    try:
                        args = json.loads(args_raw)
                    except Exception:
                        args = {}
                    parts.append(f"action: {nm}({_compress_args(args, 80)})")
            else:
                compressed = _compress_assistant(m.get("content") or "")
                if compressed:
                    parts.append(f"action: {compressed}")
        elif role == "tool":
            nm = m.get("name", "tool")
            obs = _compress_tool_obs(m.get("content") or "")
            # Respond tool results ARE the user's reply — label them as such
            # so the model understands the user has answered (not a DB obs).
            if nm in ("respond", "send_user"):
                parts.append(f"user_reply: {obs}")
            else:
                parts.append(f"obs({nm}): {obs}")
        elif role == "user":
            text = str(m.get("content") or "")[:200]
            if text:
                parts.append(f"user: {text}")

    out = "\n".join(parts)
    return out[:max_chars]


__all__ = [
    "SYSTEM_PROMPT",
    "render_tools_block",
    "render_proposer_user_message",
    "parse_proposer_response",
    "trim_history",
    "_compress_args",
    "_compress_assistant",
    "_compress_tool_obs",
]
