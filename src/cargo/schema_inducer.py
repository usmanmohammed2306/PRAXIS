"""Auto-induce :class:`ToolEffectSchema` for each registered tool.

We use the LLM to *augment* a deterministic rule-based classifier:

  1. The rule-based classifier inspects the tool name and parameter shape
     and assigns a default risk class. This always succeeds and never
     calls the model.
  2. If the rule-based class is confident (e.g. ``cancel_order`` is
     IRREVERSIBLE), we don't call the model at all.
  3. Otherwise (or if ``CARGO_INDUCE_VIA_LLM=1``), we ask the LLM to
     classify ambiguous tools and emit pre/post-conditions in NL.

Schemas are cached at module level keyed by (name, sorted-required-params).
This means a tau-bench retail run with ~20 tools incurs at most ~20
induction calls per process (not per task), and ACEBench's smaller tool
sets cost a handful of calls each.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .risk_class import RiskClass
from .schemas import ToolEffectSchema


# ---------------------------------------------------------------------------
# Rule-based classification
# ---------------------------------------------------------------------------
_IRREVERSIBLE_PREFIXES = (
    "cancel_", "delete_", "remove_", "refund_", "return_", "charge_",
    "send_", "transfer_", "pay_", "ship_", "purge_", "destroy_",
)
_WRITE_PREFIXES = (
    "update_", "modify_", "change_", "edit_", "set_", "add_",
    "place_", "book_", "submit_", "exchange_", "create_", "register_",
    "post_", "patch_", "put_", "assign_", "schedule_",
)
_READ_PREFIXES = (
    "get_", "list_", "find_", "search_", "lookup_", "fetch_",
    "view_", "show_", "retrieve_", "describe_", "calculate_",
    "calc_", "check_", "read_", "query_",
)
_FINAL_NAMES = {"respond", "finish", "final", "send_user", "answer"}
_ASK_USER_NAMES = {"transfer_to_human_agents", "ask_user", "ask"}

# Param names that look like opaque IDs (regex-grounded at execution time).
_ID_PARAM_HINTS = {
    "id", "user_id", "customer_id", "order_id", "reservation_id",
    "booking_id", "item_id", "product_id", "flight_id",
    "payment_method_id", "payment_id", "account_id", "ticket_id",
    "session_id",
}


def _is_id_param(name: str, prop: Optional[Dict[str, Any]]) -> bool:
    n = (name or "").lower()
    if n in _ID_PARAM_HINTS:
        return True
    if n.endswith("_id") or n.endswith("id") or n.endswith("_number"):
        return True
    if prop and isinstance(prop, dict):
        # Common pattern: "type": "string", "description": "...id..."
        desc = str(prop.get("description", "")).lower()
        if "id" in desc and ("identifier" in desc or "unique" in desc):
            return True
    return False


def _rule_classify(name: str) -> Tuple[RiskClass, bool]:
    """Return (risk_class, confident_enough_to_skip_llm)."""
    n = (name or "").lower()
    if n in _FINAL_NAMES:
        return RiskClass.FINAL, True
    if n in _ASK_USER_NAMES:
        return RiskClass.ASK_USER, True
    for p in _IRREVERSIBLE_PREFIXES:
        if n.startswith(p):
            return RiskClass.IRREVERSIBLE, True
    for p in _WRITE_PREFIXES:
        if n.startswith(p):
            return RiskClass.WRITE, True
    for p in _READ_PREFIXES:
        if n.startswith(p):
            return RiskClass.READ, True
    return RiskClass.WRITE, False  # ambiguous → fail-closed default


def _default_pre_post(name: str, cls: RiskClass) -> Tuple[List[str], List[str]]:
    """Cheap NL post-condition templates used when LLM induction is skipped.

    Note: we deliberately leave the *pre-conditions* empty by default. The
    pre-condition gate then becomes "check whatever the LLM proposer declared"
    rather than "check generic placeholder text we made up". The structural
    required-args check + the argument-grounding gate provide the
    deterministic safety net for unverified entities.
    """
    if cls == RiskClass.READ:
        return ([], [f"information about '{name}' is returned"])
    if cls == RiskClass.WRITE:
        return ([], [f"'{name}' has updated the targeted entity"])
    if cls == RiskClass.IRREVERSIBLE:
        return ([], [f"'{name}' has applied a non-reversible state change"])
    if cls == RiskClass.FINAL:
        return ([], ["the user has been given a clear final answer"])
    if cls == RiskClass.ASK_USER:
        return ([], ["the user has been asked for the missing information"])
    return ([], [])


# ---------------------------------------------------------------------------
# Tool-spec parsing
# ---------------------------------------------------------------------------
def _tool_name(spec: Dict[str, Any]) -> str:
    fn = spec.get("function", spec) if isinstance(spec, dict) else {}
    return str(fn.get("name", "")) if isinstance(fn, dict) else ""


def _tool_description(spec: Dict[str, Any]) -> str:
    fn = spec.get("function", spec) if isinstance(spec, dict) else {}
    return str(fn.get("description", "")) if isinstance(fn, dict) else ""


def _tool_params(spec: Dict[str, Any]) -> Dict[str, Any]:
    fn = spec.get("function", spec) if isinstance(spec, dict) else {}
    params = fn.get("parameters", {}) if isinstance(fn, dict) else {}
    return params if isinstance(params, dict) else {}


def _tool_required(params: Dict[str, Any]) -> List[str]:
    req = params.get("required") if isinstance(params, dict) else None
    if isinstance(req, list):
        return [str(p) for p in req]
    return []


def _tool_properties(params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    props = params.get("properties") if isinstance(params, dict) else {}
    return props if isinstance(props, dict) else {}


# ---------------------------------------------------------------------------
# Optional LLM induction
# ---------------------------------------------------------------------------
_INDUCTION_SYSTEM = (
    "You classify tool functions for a risk-typed agent. "
    "Given a tool's name, description, and parameters, return a JSON object "
    "with the keys: cls (one of READ, WRITE, IRREVERSIBLE, FINAL, ASK_USER), "
    "irreversible (boolean), preconditions (list of short NL strings), "
    "postconditions (list of short NL strings). Be conservative: any "
    "tool that mutates state and cannot be undone in-domain is IRREVERSIBLE. "
    "Output ONLY valid JSON."
)


def _induce_via_llm(
    *, client, model: str, name: str, description: str,
    params: Dict[str, Any], temperature: float = 0.0,
) -> Optional[Dict[str, Any]]:
    if client is None:
        return None
    try:
        prompt = (
            f"Tool name: {name}\n"
            f"Description: {description[:600]}\n"
            f"Parameters JSON: {json.dumps(params, default=str)[:1200]}\n\n"
            "Classify and return JSON only."
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _INDUCTION_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=400,
        )
        text = (resp.choices[0].message.content or "").strip()
        return _extract_json_obj(text)
    except Exception:
        return None


def _extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    # Strip code fences
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(),
               flags=re.IGNORECASE | re.MULTILINE)
    candidates: List[str] = [s]
    candidates.extend(re.findall(r"\{(?:[^{}]|\{[^{}]*\})*\}", s, flags=re.DOTALL))
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
_SCHEMA_CACHE: Dict[str, ToolEffectSchema] = {}


def _cache_key(name: str, required: List[str]) -> str:
    return f"{name}::{','.join(sorted(required))}"


def induce_schema(
    spec: Dict[str, Any],
    *,
    client=None,
    model: str = "",
    use_llm: Optional[bool] = None,
    temperature: float = 0.0,
) -> ToolEffectSchema:
    """Build (or fetch from cache) the :class:`ToolEffectSchema` for ``spec``."""
    name = _tool_name(spec)
    if not name:
        return ToolEffectSchema(name="<unknown>", cls=RiskClass.WRITE)
    description = _tool_description(spec)
    params = _tool_params(spec)
    required = _tool_required(params)
    properties = _tool_properties(params)

    key = _cache_key(name, required)
    if key in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[key]

    rule_cls, confident = _rule_classify(name)
    if use_llm is None:
        use_llm = os.environ.get("CARGO_INDUCE_VIA_LLM", "0") == "1"

    cls = rule_cls
    pre, post = _default_pre_post(name, cls)
    irreversible = cls == RiskClass.IRREVERSIBLE

    if (not confident) and use_llm and client is not None:
        induced = _induce_via_llm(
            client=client, model=model, name=name,
            description=description, params=params, temperature=temperature,
        )
        if induced:
            try:
                cls = RiskClass.parse(induced.get("cls"))
                irreversible = bool(induced.get("irreversible", cls == RiskClass.IRREVERSIBLE))
                pre_in = induced.get("preconditions") or []
                post_in = induced.get("postconditions") or []
                if isinstance(pre_in, list):
                    pre = [str(x)[:200] for x in pre_in if str(x).strip()][:6]
                if isinstance(post_in, list):
                    post = [str(x)[:200] for x in post_in if str(x).strip()][:6]
                if not pre or not post:
                    d_pre, d_post = _default_pre_post(name, cls)
                    pre = pre or d_pre
                    post = post or d_post
            except Exception:
                # Fall back to rule-based result silently.
                pass

    arg_id_fields = [p for p in required if _is_id_param(p, properties.get(p))]
    schema = ToolEffectSchema(
        name=name,
        cls=cls,
        irreversible=irreversible or (cls == RiskClass.IRREVERSIBLE),
        preconditions=pre,
        postconditions=post,
        arg_id_fields=arg_id_fields,
        param_properties=dict(properties),
        required_params=list(required),
        description=description.strip(),
    )
    _SCHEMA_CACHE[key] = schema
    return schema


def induce_schemas(
    tool_specs: List[Dict[str, Any]],
    *,
    client=None,
    model: str = "",
    use_llm: Optional[bool] = None,
    temperature: float = 0.0,
) -> Dict[str, ToolEffectSchema]:
    """Induce a schema for every tool spec; returns name → schema."""
    out: Dict[str, ToolEffectSchema] = {}
    for spec in tool_specs or []:
        sch = induce_schema(
            spec, client=client, model=model,
            use_llm=use_llm, temperature=temperature,
        )
        if sch.name and sch.name != "<unknown>":
            out[sch.name] = sch
    # Always provide a synthetic "respond" schema for FINAL/ASK_USER paths.
    if "respond" not in out:
        # Note: required_params is intentionally empty. The proposer carries
        # the user-facing text in `user_text`; the agent translates that into
        # the env's actual `kwargs={'content': ...}` at execution time. So
        # the structural "required arg" check should not fire on respond.
        out["respond"] = ToolEffectSchema(
            name="respond", cls=RiskClass.FINAL, irreversible=False,
            preconditions=[],
            postconditions=["the user has been given a clear final answer"],
            arg_id_fields=[], param_properties={
                "content": {"type": "string", "description": "final answer text"}
            },
            required_params=[],
            description="Send the final answer (or a clarifying question) to the user.",
        )
    return out


def reset_cache() -> None:
    """Clear the module-level schema cache (used in tests)."""
    _SCHEMA_CACHE.clear()


__all__ = ["induce_schema", "induce_schemas", "reset_cache"]
