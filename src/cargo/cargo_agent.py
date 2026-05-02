"""CARGO agent for tau-bench.

Drop-in replacement for the baseline agents (vanilla / Act / ReAct). The
agent uses a custom JSON-emitting proposer (no native tool-calling) so it
can declare risk class, pre/post-conditions, and free-form user text in
one structured response. Each declared mutation passes through the
calibrated gate before reaching ``env.step``.

This file exists at the integration boundary with tau-bench. The kernel
of CARGO (gates / schemas / repair) lives in pure-Python modules with no
tau-bench dependency, so it can be unit-tested without a live env.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

try:  # tau-bench is optional at import-time so unit tests can run without it.
    from tau_bench.agents.base import Agent
    from tau_bench.types import Action, SolveResult
    _HAS_TAU = True
except Exception:  # noqa: BLE001
    Agent = object  # type: ignore[assignment]
    Action = None  # type: ignore[assignment]
    SolveResult = None  # type: ignore[assignment]
    _HAS_TAU = False

from . import repair
from .calibration import default_calibration
from .gates import (
    check_arg_grounding,
    check_counterfactual,
    check_postconditions,
    check_preconditions,
    check_repeat_loop,
    check_self_consistency,
)
from .proposer import (
    SYSTEM_PROMPT,
    parse_proposer_response,
    render_proposer_user_message,
    render_tools_block,
    trim_history,
)
from .risk_class import RiskClass, is_gated, is_irreversible_or_final
from .schema_inducer import induce_schemas
from .schemas import GateResult, ProposedAction, ToolEffectSchema
from .stats import CargoStats
from .working_memory import WorkingMemory


def _get_openai_client():
    """Lazy import of the project's OpenAI client (depends on ``openai`` SDK).

    Imported inside CargoAgent.__init__ so this module is importable in
    environments without the OpenAI SDK (e.g. unit tests that pass a
    MockClient instead of going through ``get_client``).
    """
    from ..common.openai_client import get_client
    return get_client()


RESPOND_TOOL_NAME = "respond"
RESPOND_MAX_CHARS = 800

# ---------------------------------------------------------------------------
# Authentication-override helpers
# ---------------------------------------------------------------------------
# Emails that are clearly fabricated placeholders — never a real user email.
_PLACEHOLDER_EMAIL_RE = re.compile(
    r"^(user|customer|test|example|demo|agent|admin|noreply|placeholder)"
    r"@(example|test|placeholder|sample|dummy|fake|mail)\.",
    re.I,
)

# Words that start sentences / common words that appear capitalised in prose
# but are NOT person first/last names.  Includes brand / product words so that
# user messages like "exchange the smart thermostat for one that works with
# Google Home" don't get parsed as the name "Google Home".
_NAME_STOPWORDS = frozenset({
    # Pronouns / sentence starters / closures
    "I", "My", "The", "A", "An", "Yes", "No", "Hi", "Hello", "Hey",
    "Could", "Can", "Please", "Sure", "Thank", "Thanks", "Will", "Is",
    "This", "That", "Also", "And", "But", "Or", "Not", "Do", "Did",
    "Would", "Should", "Have", "Has", "Had", "Does", "Good", "Great",
    "Ok", "Okay", "Sorry", "Note", "Here", "There", "We", "You", "He",
    "She", "They", "It", "Our", "Your", "His", "Her", "Their", "Its",
    # Domain nouns (auth)
    "ZIP", "Code", "ID", "Order", "Email", "Name", "Number", "Phone",
    "User", "Customer", "Account", "Profile", "Address", "Date",
    # Brand / product / tech nouns (retail domain)
    "Google", "Apple", "Amazon", "Microsoft", "Sony", "Samsung", "Nintendo",
    "Bose", "Logitech", "Asus", "Dell", "Lenovo", "Razer", "Bluetooth", "USB",
    "Home", "Office", "Pro", "Plus", "Max", "Mini", "Lite", "Ultra", "Smart",
    "Wireless", "Wired", "Mechanical", "Standard", "Premium", "Basic", "Light",
    "Dark", "Black", "White", "Blue", "Red", "Green", "Yellow", "Pink",
    "Stainless", "Steel", "Plastic", "Wood", "Glass", "Cotton", "Leather",
    "RGB", "LED", "OLED", "TV", "PC", "Mac", "Tablet", "Laptop", "Phone",
    "Keyboard", "Mouse", "Headset", "Charger", "Cable", "Speaker", "Camera",
    "Watch", "Thermostat", "Vacuum", "Cleaner", "Bottle", "Backpack", "Jacket",
    "Shirt", "Shoes", "Pants", "Dress", "Hat", "T", "Hoodie", "Sweater",
    "Switches", "Switch", "Battery", "Display", "Screen", "Buttons", "Charging",
    # Misc capitalised filler that occurs in commerce text
    "Color", "Size", "Style", "Type", "Model", "Version", "Series",
    "First", "Last", "Recent", "New", "Old", "Original",
})

_EMAIL_RE_MOD = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
# Loose pattern: any two adjacent Title-case words.  Used only as a fallback
# when we have just asked the user for their name (i.e. they're in
# "answering an auth question" mode).
_NAME_PAIR_RE = re.compile(r"\b([A-Z][a-z]{1,20})\s+([A-Z][a-z]{1,20})\b")
# Strict pattern: only extract names that follow an explicit introduction
# phrase ("my name is X Y", "I'm X Y", "this is X Y", etc.).  This is the
# default — it prevents random Title-case bigrams in user prose (e.g.
# "Google Home", "Smart Thermostat") from being mistaken for person names.
_NAME_INTRO_RE = re.compile(
    r"(?:my\s+name\s+(?:is|'s)|i\s+am\b|i'm\b|this\s+is|call\s+me|"
    r"name(?:\s*[:=]|d\b)|i\s+go\s+by)"
    r"[\s,]+([A-Z][a-z]{1,20})\s+([A-Z][a-z]{1,20})",
    re.IGNORECASE,
)
_ZIP_RE = re.compile(r"\b(\d{5})\b")

# Goal keywords that indicate a "no auth required" query.  When the goal
# matches AND no PII is in user_facts, the auth override redirects placeholder
# find_user_id_by_email proposals to list_all_product_types instead of
# asking for credentials.
_PRODUCT_QUERY_RE = re.compile(
    r"\b(t-?shirt|product|products|store|items?|options?|available|"
    r"how\s+many|stock|catalog|inventory|browse|brands?)\b",
    re.IGNORECASE,
)

# user_id pattern produced by tau-bench retail (e.g. "yusuf_rossi_9620").
_USER_ID_PATTERN = re.compile(r"\b([a-z]+_[a-z]+_\d{1,8})\b")

# Phrases that indicate the user is refusing to provide authentication info.
# When any of these are found we stop asking and issue a polite FINAL.
_AUTH_REFUSAL_PHRASES = (
    "prefer not",
    "don't want",
    "do not want",
    "don't have",
    "do not have",
    "rather not",
    "not comfortable",
    "won't share",
    "will not share",
    "can't provide",
    "cannot provide",
    "skip",
    "never mind",
    "forget it",
    "no thanks",
    "no thank you",
)

# Maximum number of ASK_USER auth prompts before we give up and issue a FINAL.
_MAX_AUTH_ASKS = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_context_overflow(exc: BaseException) -> bool:
    s = str(exc).lower()
    return (
        ("context" in s and "length" in s)
        or "maximum context length" in s
        or "contextwindowexceeded" in s
        or exc.__class__.__name__ == "ContextWindowExceededError"
    )


def _float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _obs_text(env_resp: Any) -> str:
    if env_resp is None:
        return ""
    obs = getattr(env_resp, "observation", env_resp)
    if isinstance(obs, (dict, list)):
        try:
            return json.dumps(obs, ensure_ascii=False, default=str)
        except Exception:
            return str(obs)
    return str(obs)


def _initial_user_message(env_reset: Any) -> str:
    if env_reset is None:
        return ""
    for attr in ("observation", "content", "message", "user_message"):
        v = getattr(env_reset, attr, None)
        if v:
            return str(v)
    if isinstance(env_reset, str):
        return env_reset
    return str(env_reset)


# ---------------------------------------------------------------------------
# CargoAgent
# ---------------------------------------------------------------------------
class CargoAgent(Agent):  # type: ignore[misc]
    """tau-bench Agent that runs the CARGO loop."""

    style_name = "cargo"

    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str = "openai",
        temperature: float = 0.0,
        env_hint: str = "",
    ) -> None:
        if not _HAS_TAU:
            # Graceful "not implemented" if someone instantiates this without
            # tau_bench installed; the test suite avoids this path.
            raise RuntimeError(
                "tau_bench is not installed; CargoAgent cannot be instantiated."
            )
        self.tools_info = tools_info or []
        self.wiki = wiki or ""
        self.model = model
        self.provider = provider
        self.temperature = float(temperature)
        self.env_hint = env_hint
        self.client = _get_openai_client()
        # Schema induction is cached at module level — repeated agent
        # instances within the same process never re-induce.
        self.schemas: Dict[str, ToolEffectSchema] = induce_schemas(
            self.tools_info,
            client=self.client,
            model=self.model,
            temperature=0.0,
        )
        self.calibration = default_calibration()

    # ------------------------------------------------------------------
    def _system_prompt(self) -> str:
        parts: List[str] = [SYSTEM_PROMPT]
        if self.wiki:
            parts.append("--- Domain policy ---\n" + self.wiki.strip())
        return "\n\n".join(parts)

    def _build_proposer_messages(
        self,
        wm: WorkingMemory,
        history: List[Dict[str, Any]],
        critique: str,
    ) -> List[Dict[str, Any]]:
        tools_block = render_tools_block(self.schemas)
        # n_turns=8 messages ≈ 2-3 complete steps; compression keeps it ≤300 tokens.
        history_tail = trim_history(history, n_turns=8, max_chars=800)
        user_msg = render_proposer_user_message(
            wm=wm,
            tools_block=tools_block,
            history_tail=history_tail,
            critique=critique,
            domain_policy="",  # already in the system prompt
        )
        return [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": user_msg},
        ]

    def _call_proposer(
        self,
        proposer_messages: List[Dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: int = 600,
    ) -> Tuple[Optional[ProposedAction], str]:
        T = self.temperature if temperature is None else float(temperature)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=proposer_messages,
                temperature=T,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            if _is_context_overflow(exc):
                # Trim and retry once.
                trimmed = [proposer_messages[0]] + proposer_messages[-1:]
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=trimmed,
                        temperature=T,
                        max_tokens=max_tokens,
                    )
                except Exception:
                    return None, ""
            else:
                return None, ""
        try:
            text = resp.choices[0].message.content or ""
        except Exception:
            text = ""
        parsed = parse_proposer_response(text, schemas=self.schemas)
        return parsed, text

    # ------------------------------------------------------------------
    # Gate orchestration
    # ------------------------------------------------------------------
    def _run_gates(
        self,
        action: ProposedAction,
        schema: ToolEffectSchema,
        wm: WorkingMemory,
        proposer_messages: List[Dict[str, Any]],
        stats: CargoStats,
    ) -> Tuple[Optional[GateResult], Dict[str, Any]]:
        """Run the gate stack on a proposed action.

        Returns (failing_gate, diagnostics). If failing_gate is None,
        the action passed every gate.
        """
        diag: Dict[str, Any] = {"gates_run": [], "gates_failed": []}

        # Repeat-loop is checked for every class — cheap and high-yield.
        rl = check_repeat_loop(action, schema, wm)
        diag["gates_run"].append("repeat_loop")
        stats.record_gate(rl)
        if not rl.ok:
            diag["gates_failed"].append("repeat_loop")
            return rl, diag

        if not is_gated(action.declared_class):
            # Even on the fast path, run arg_grounding so hallucinated
            # placeholder values (e.g. "user@example.com") are caught before
            # the tool executes and wastes the first step.
            ag = check_arg_grounding(action, schema, wm)
            diag["gates_run"].append("arg_grounding")
            stats.record_gate(ag)
            if not ag.ok:
                diag["gates_failed"].append("arg_grounding")
                return ag, diag
            return None, diag

        # 4a. Pre-conditions
        pc = check_preconditions(action, schema, wm)
        diag["gates_run"].append("preconditions")
        stats.record_gate(pc)
        if not pc.ok:
            diag["gates_failed"].append("preconditions")
            return pc, diag

        # 4b. Argument grounding
        ag = check_arg_grounding(action, schema, wm)
        diag["gates_run"].append("arg_grounding")
        stats.record_gate(ag)
        if not ag.ok:
            diag["gates_failed"].append("arg_grounding")
            return ag, diag

        # 4c. Self-consistency (k samples per class).
        k = self.calibration.sc_k.get(action.declared_class, 0)
        threshold = self.calibration.sc_thresholds.get(action.declared_class, 0.0)
        if k >= 1 and threshold > 0:
            sc = check_self_consistency(
                action, schema,
                client=self.client,
                model=self.model,
                proposer_messages=proposer_messages,
                schemas_for_parse=self.schemas,
                k=k,
                threshold=threshold,
            )
            diag["gates_run"].append("self_consistency")
            stats.record_gate(sc)
            diag["sc_agreement"] = sc.diagnostics.get("agreement")
            if not sc.ok:
                diag["gates_failed"].append("self_consistency")
                return sc, diag

        # 4d. Counterfactual rollout (IRREV / FINAL only).
        if self.calibration.run_cf.get(action.declared_class, False) and is_irreversible_or_final(action.declared_class):
            cf = check_counterfactual(
                action, schema, wm,
                client=self.client, model=self.model,
                temperature=0.0,
            )
            diag["gates_run"].append("counterfactual")
            stats.record_gate(cf)
            diag["cf_predicted_blocking"] = bool(cf.diagnostics.get("cf_reason")) and not cf.ok
            if not cf.ok:
                diag["gates_failed"].append("counterfactual")
                return cf, diag

        return None, diag

    # ------------------------------------------------------------------
    # Main loop (mirrors the baseline contract for tau-bench)
    # ------------------------------------------------------------------
    def solve(
        self,
        env,
        task_index: Optional[int] = None,
        max_num_steps: int = 30,
    ) -> "SolveResult":
        env_reset = env.reset(task_index=task_index)
        initial_user = _initial_user_message(env_reset)

        wm = WorkingMemory(goal=initial_user, budget_steps=max_num_steps)
        wm.absorb_user_message(initial_user)

        # ``messages`` is the *trajectory* surface tau-bench / scorers see.
        # We render it in the same OpenAI-chat shape the baselines use so
        # downstream tooling (judges, replay, label classifiers) is happy.
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt()},
        ]
        if initial_user:
            messages.append({"role": "user", "content": initial_user})

        stats = CargoStats()
        reward: float = 0.0
        info: Dict[str, Any] = {}
        total_cost: float = 0.0
        done = False
        step_error: str = ""

        critique = ""
        retries_used = 0
        consecutive_parse_failures = 0
        MAX_CONSEC_PARSE_FAIL = 3

        for step in range(max_num_steps):
            wm.budget_steps = max_num_steps - step
            stats.steps_total += 1

            proposer_messages = self._build_proposer_messages(wm, messages, critique)
            action, raw_text = self._call_proposer(proposer_messages)

            # Authentication override: if the proposer is stuck using a
            # placeholder email, replace the action with the best alternative
            # we can construct from what the user has actually provided.
            # This bypasses the model's inability to follow the name+zip hint
            # in critiques and is purely deterministic / zero-cost.
            if action is not None:
                auth_action = self._auth_override(action, wm)
                if auth_action is not None:
                    raw_text = json.dumps({
                        "thought": auth_action.raw_thought,
                        "action": {
                            "name": auth_action.name,
                            "args": auth_action.args,
                            "declared_class": auth_action.declared_class.value,
                            "declared_pre": auth_action.declared_pre,
                            "declared_post": auth_action.declared_post,
                            "informational_intent": auth_action.informational_intent,
                            "user_text": auth_action.user_text,
                        },
                    })
                    action = auth_action

            # Product-ID override: if the proposer passes a product *type name*
            # (e.g. "T-Shirt") where a numeric product_id is required, resolve it
            # from db_facts that were populated by list_all_product_types.
            if action is not None:
                prod_action = self._resolve_product_id_name(action, wm)
                if prod_action is not None:
                    raw_text = json.dumps({
                        "thought": prod_action.raw_thought,
                        "action": {
                            "name": prod_action.name,
                            "args": prod_action.args,
                            "declared_class": prod_action.declared_class.value,
                            "declared_pre": prod_action.declared_pre,
                            "declared_post": prod_action.declared_post,
                            "informational_intent": prod_action.informational_intent,
                            "user_text": prod_action.user_text,
                        },
                    })
                    action = prod_action

            # Product-list advance: if list_all_product_types is about to be
            # repeated (already in recent_signatures) and the user has mentioned
            # a product type name we can resolve, skip straight to
            # get_product_details with the resolved ID.
            if action is not None:
                adv_action = self._advance_after_product_list(action, wm)
                if adv_action is not None:
                    raw_text = json.dumps({
                        "thought": adv_action.raw_thought,
                        "action": {
                            "name": adv_action.name,
                            "args": adv_action.args,
                            "declared_class": adv_action.declared_class.value,
                            "declared_pre": adv_action.declared_pre,
                            "declared_post": adv_action.declared_post,
                            "informational_intent": adv_action.informational_intent,
                            "user_text": adv_action.user_text,
                        },
                    })
                    action = adv_action

            messages.append({"role": "assistant", "content": raw_text or ""})

            if action is None:
                consecutive_parse_failures += 1
                stats.json_parse_failures += 1
                if consecutive_parse_failures >= MAX_CONSEC_PARSE_FAIL:
                    step_error = "json_parse_failures_exceeded"
                    # Force a polite finalize.
                    env_resp = self._respond(
                        env,
                        "I'm having trouble formatting my response. "
                        "Could you confirm the details and try again?",
                    )
                    if env_resp is None:
                        break
                    user_reply = _obs_text(env_resp)
                    if user_reply:
                        messages.append({"role": "user", "content": user_reply})
                        wm.absorb_user_message(user_reply)
                    reward = _float(getattr(env_resp, "reward", reward), reward)
                    info = getattr(env_resp, "info", info) or info
                    done = bool(getattr(env_resp, "done", False))
                    if done:
                        break
                    consecutive_parse_failures = 0
                    continue
                # Soft retry with a JSON-format critique.
                gr = GateResult.failing("json_parse", "unparseable_proposer_output")
                stats.record_gate(gr)
                stats.abstain_total += 1
                rd = repair.decide(
                    gr,
                    retries_used=retries_used,
                    max_retries=2,
                    budget_steps_remaining=wm.budget_steps,
                )
                if rd.action == "RETRY":
                    critique = rd.critique
                    retries_used += 1
                    stats.repair_retry += 1
                    continue
                # Fallthrough = ASK_USER / FINALIZE_GENERIC: send a respond.
                env_resp = self._respond(env, rd.user_message)
                if env_resp is None:
                    break
                user_reply = _obs_text(env_resp)
                if user_reply:
                    messages.append({"role": "user", "content": user_reply})
                    wm.absorb_user_message(user_reply)
                reward = _float(getattr(env_resp, "reward", reward), reward)
                info = getattr(env_resp, "info", info) or info
                done = bool(getattr(env_resp, "done", False))
                if rd.action == "ASK_USER":
                    stats.repair_ask_user += 1
                else:
                    stats.repair_finalize += 1
                if done:
                    break
                critique = ""
                retries_used = 0
                continue

            consecutive_parse_failures = 0
            schema = self._schema_for(action)
            failing, diag = self._run_gates(action, schema, wm, proposer_messages, stats)

            step_record: Dict[str, Any] = {
                "step": step,
                "action_name": action.name,
                "declared_class": action.declared_class.value,
                "args": dict(action.args),
                "thought": action.raw_thought[:200],
                "fast_path": not is_gated(action.declared_class),
                "gates_run": diag.get("gates_run", []),
                "gates_failed": diag.get("gates_failed", []),
                "abstain_reason": failing.reason if failing else "",
                "sc_agreement": diag.get("sc_agreement"),
                "cf_predicted_blocking": diag.get("cf_predicted_blocking"),
            }

            if failing is not None:
                stats.steps_gated += 1
                stats.abstain_total += 1
                rd = repair.decide(
                    failing,
                    proposed=action,
                    retries_used=retries_used,
                    max_retries=2,
                    budget_steps_remaining=wm.budget_steps,
                )
                step_record["repair_action"] = rd.action
                stats.record_step(step_record)

                if rd.action == "RETRY":
                    critique = rd.critique
                    retries_used += 1
                    stats.repair_retry += 1
                    continue
                # ASK_USER / FINALIZE_GENERIC → send a respond.
                env_resp = self._respond(env, rd.user_message or
                                         "Could you provide a bit more detail so I can proceed?")
                if env_resp is None:
                    break
                user_reply = _obs_text(env_resp)
                if user_reply:
                    messages.append({"role": "user", "content": user_reply})
                    wm.absorb_user_message(user_reply)
                reward = _float(getattr(env_resp, "reward", reward), reward)
                info = getattr(env_resp, "info", info) or info
                done = bool(getattr(env_resp, "done", False))
                if rd.action == "ASK_USER":
                    stats.repair_ask_user += 1
                else:
                    stats.repair_finalize += 1
                if done:
                    break
                critique = ""
                retries_used = 0
                continue

            # --- Action passed every gate. Execute. --------------------
            critique = ""
            retries_used = 0
            if is_gated(action.declared_class):
                stats.steps_gated += 1
            else:
                stats.steps_fast_path += 1

            wm.record_action_signature(action.signature())

            # FINAL / ASK_USER → route through respond with the model's user_text.
            if (action.declared_class in (RiskClass.FINAL, RiskClass.ASK_USER)
                    or action.name.lower() in (RESPOND_TOOL_NAME, "send_user", "finish", "final", "answer")):
                content = (action.user_text or action.raw_thought or "").strip()
                if not content:
                    content = "Is there anything else I can help you with?"
                env_resp = self._respond(env, content)
                if env_resp is None:
                    break
                # Log the translated tool call for trajectory consumers.
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": f"cargo_{step}",
                        "type": "function",
                        "function": {
                            "name": RESPOND_TOOL_NAME,
                            "arguments": json.dumps({"content": content[:RESPOND_MAX_CHARS]}),
                        },
                    }],
                })
                user_reply = _obs_text(env_resp)
                messages.append({
                    "role": "tool",
                    "tool_call_id": f"cargo_{step}",
                    "name": RESPOND_TOOL_NAME,
                    "content": user_reply,
                })
                if user_reply:
                    wm.absorb_user_message(user_reply)
                reward = _float(getattr(env_resp, "reward", reward), reward)
                info = getattr(env_resp, "info", info) or info
                done = bool(getattr(env_resp, "done", False))
                stats.actions_executed += 1
                cls = action.declared_class.value
                stats.executed_by_class[cls] = stats.executed_by_class.get(cls, 0) + 1
                step_record["executed"] = True
                stats.record_step(step_record)
                if done:
                    break
                continue

            # Real tool dispatch.
            translated_call_id = f"cargo_{step}_{action.name}"
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": translated_call_id,
                    "type": "function",
                    "function": {
                        "name": action.name,
                        "arguments": json.dumps(action.args, default=str),
                    },
                }],
            })
            try:
                env_resp = env.step(Action(name=action.name, kwargs=dict(action.args)))
            except Exception as env_exc:
                if _is_context_overflow(env_exc):
                    step_error = f"env_step_context_overflow: {env_exc}"
                    done = True
                    break
                # Don't crash the trajectory: log the failure as an obs and
                # let the agent decide how to recover.
                err_obs = json.dumps({"status": "error", "error": str(env_exc)})
                messages.append({
                    "role": "tool",
                    "tool_call_id": translated_call_id,
                    "name": action.name,
                    "content": err_obs,
                })
                wm.last_error = str(env_exc)[:80]  # tip 8: minimal failure record
                wm.absorb_observation({"status": "error", "error": str(env_exc)[:80]})
                step_record["env_error"] = str(env_exc)[:80]
                stats.record_step(step_record)
                continue

            tool_obs = _obs_text(env_resp)
            messages.append({
                "role": "tool",
                "tool_call_id": translated_call_id,
                "name": action.name,
                "content": tool_obs,
            })
            # Surface tool-level errors in last_error so the proposer's STATE
            # block reflects why the previous step failed, not just the
            # post-condition label.
            if tool_obs and (
                tool_obs.lstrip().lower().startswith("error")
                or '"error"' in tool_obs.lower()
            ):
                wm.last_error = tool_obs[:80]
            wm.absorb_observation(tool_obs if not isinstance(env_resp, type(None))
                                  else "")

            # Track failed name+ZIP lookups so _auth_override doesn't retry the
            # same invalid ZIP on the next step.
            if action.name == "find_user_id_by_name_zip":
                zip_tried = str(action.args.get("zip", ""))
                obs_lower = tool_obs.lower() if tool_obs else ""
                lookup_failed = zip_tried and (
                    "not found" in obs_lower
                    or '"error"' in obs_lower
                    or obs_lower.lstrip().startswith("error")
                    or "no user" in obs_lower
                    or "invalid" in obs_lower
                )
                if lookup_failed:
                    if zip_tried not in wm.auth_failed_zips:
                        wm.auth_failed_zips.append(zip_tried)

            # Cache user_id on successful auth lookup so the override can
            # cleanly suppress subsequent find_user_id_* proposals.
            if action.name in ("find_user_id_by_email", "find_user_id_by_name_zip"):
                if tool_obs:
                    m = _USER_ID_PATTERN.search(tool_obs)
                    if m and not wm.auth_user_id:
                        wm.auth_user_id = m.group(1)
            # Post-condition check (advisory).
            try:
                post_obs = json.loads(tool_obs) if tool_obs.strip().startswith(("{", "[")) else tool_obs
            except Exception:
                post_obs = tool_obs
            pc = check_postconditions(action, schema, obs=post_obs)
            stats.record_gate(pc)
            if not pc.ok:
                wm.last_error = pc.reason[:80]  # tip 8: minimal failure record
                step_record["post_error"] = pc.reason

            reward = _float(getattr(env_resp, "reward", reward), reward)
            info = getattr(env_resp, "info", info) or info
            done = bool(getattr(env_resp, "done", False))
            stats.actions_executed += 1
            cls = action.declared_class.value
            stats.executed_by_class[cls] = stats.executed_by_class.get(cls, 0) + 1
            step_record["executed"] = True
            stats.record_step(step_record)
            if done:
                break

        info = dict(info) if info else {}
        if step_error:
            info["error"] = step_error
        info.setdefault("controller", self.style_name)
        info["cargo_stats"] = stats.snapshot()
        return SolveResult(  # type: ignore[call-arg]
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
        )

    # ------------------------------------------------------------------
    # Authentication override
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Identity extraction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_real_email(facts: List[str]) -> Optional[str]:
        for fact in facts:
            m = _EMAIL_RE_MOD.search(fact)
            if m and not _PLACEHOLDER_EMAIL_RE.match(m.group(0)):
                return m.group(0)
        return None

    @staticmethod
    def _extract_name_pair(
        facts: List[str], in_response_to_ask: bool
    ) -> Optional[Tuple[str, str]]:
        """Return (first_name, last_name) extracted from user facts.

        Strategy:
        - Always try the strict introduction-anchored pattern first.  This
          requires the name to follow phrases like "my name is", "I'm",
          "this is", which prevents random Title-case bigrams in product
          descriptions ("Google Home", "Smart Thermostat") from being
          mistaken for a person's name.
        - Only when ``in_response_to_ask`` is True (we just asked the user
          for their name) do we fall back to the loose Title-case pattern.
          Even in fallback we apply the stopword list so brand words don't
          slip through.
        """
        # Strict pass: introduction-anchored
        for fact in facts:
            m = _NAME_INTRO_RE.search(fact)
            if not m:
                continue
            fn, ln = m.group(1), m.group(2)
            if fn not in _NAME_STOPWORDS and ln not in _NAME_STOPWORDS:
                return fn, ln
        if not in_response_to_ask:
            return None
        # Loose fallback when we just asked for a name
        for fact in facts:
            for m in _NAME_PAIR_RE.finditer(fact):
                fn, ln = m.group(1), m.group(2)
                if fn not in _NAME_STOPWORDS and ln not in _NAME_STOPWORDS:
                    return fn, ln
        return None

    @staticmethod
    def _extract_zip(facts: List[str], failed_zips: List[str]) -> Optional[str]:
        for fact in facts:
            m = _ZIP_RE.search(fact)
            if m:
                candidate = m.group(1)
                if candidate not in failed_zips:
                    return candidate
        return None

    @staticmethod
    def _existing_user_id(wm: "WorkingMemory") -> Optional[str]:
        """Return a user_id from db_facts (or wm.auth_user_id), if any."""
        if wm.auth_user_id:
            return wm.auth_user_id
        for fact in wm.db_facts:
            m = _USER_ID_PATTERN.search(fact)
            if m:
                return m.group(1)
        return None

    @staticmethod
    def _is_no_auth_query(wm: "WorkingMemory") -> bool:
        """Heuristic: does the goal look like a pure product/store query?"""
        if not wm.goal:
            return False
        return bool(_PRODUCT_QUERY_RE.search(wm.goal))

    def _post_auth_action(self, wm: "WorkingMemory") -> Optional[ProposedAction]:
        """Build a sensible next action after the user has been authenticated.

        Called when the model proposes find_user_id_* even though we already
        have a user_id.  Replaces the redundant lookup with a progress action.
        """
        user_id = self._existing_user_id(wm)
        if not user_id:
            return None

        # If we haven't called get_user_details yet, do that.
        already_called = any(
            "get_user_details" in s and user_id in s
            for s in wm.recent_signatures
        )
        if not already_called:
            return ProposedAction(
                name="get_user_details",
                args={"user_id": user_id},
                declared_class=RiskClass.READ,
                declared_pre=["user authenticated"],
                declared_post=["user details retrieved"],
                informational_intent="fetch authenticated user's profile",
                raw_thought=(
                    "Post-auth override: replacing redundant find_user_id_* "
                    "with get_user_details to make progress on the goal."
                ),
                user_text="",
                raw_response="",
            )

        # Already have user details.  Route by goal keywords.
        if self._is_no_auth_query(wm):
            return ProposedAction(
                name="list_all_product_types",
                args={},
                declared_class=RiskClass.READ,
                declared_pre=["product query"],
                declared_post=["product types listed"],
                informational_intent="list product types for the user",
                raw_thought="Post-auth override: routing to list_all_product_types for product query.",
                user_text="",
                raw_response="",
            )

        # Default: return None and let the model figure out the next step.
        return None

    # ------------------------------------------------------------------
    # Authentication override
    # ------------------------------------------------------------------
    def _auth_override(
        self,
        action: ProposedAction,
        wm: "WorkingMemory",
    ) -> Optional[ProposedAction]:
        """Intercept a placeholder-email lookup and replace it with the
        right action based on what the user has actually told us.

        Failure modes this guards against (each has been observed in traces):
        - Model is biased toward ``find_user_id_by_email`` and ignores
          critiques telling it to use ``find_user_id_by_name_zip``.
        - Model proposes find_user_id_* AFTER successful authentication
          → loops on placeholder email forever.
        - User asks a no-auth query (e.g. "how many t-shirts?") → without
          this override the model still tries to authenticate.
        - User refuses to share credentials → must transition cleanly to a
          no-auth pathway, not loop on the same FINAL message.
        - Greedy name extraction picks up brand/product words ("Google Home",
          "Smart Thermostat") → the strict intro-anchored extractor below
          rejects these unless they follow a name introduction phrase.

        Return value:
          - A replacement ProposedAction if the override applies.
          - None if the model's proposal should proceed unchanged.
        """
        # Only override find_user_id_by_email / find_user_id_by_name_zip.
        if not action.name.startswith("find_user_id_"):
            return None

        # If the model's name-zip call already has clean grounded args (real
        # name, real zip, neither in failed list), let it through.  Only
        # override email-with-placeholder.
        if action.name == "find_user_id_by_name_zip":
            zp = str(action.args.get("zip", ""))
            fn = str(action.args.get("first_name", ""))
            ln = str(action.args.get("last_name", ""))
            # Block if the model is using stopword-tainted args (e.g. "Google Home")
            if fn in _NAME_STOPWORDS or ln in _NAME_STOPWORDS:
                # Tainted — fall through to the placeholder-rewrite logic.
                pass
            elif zp in wm.auth_failed_zips:
                # Reusing a known-bad ZIP — fall through.
                pass
            else:
                # Looks clean — let it proceed.
                return None

        # ---------------------------------------------------------------
        # Path 1: Already authenticated (user_id in db_facts).  The model is
        # in a placeholder-email loop and we need to redirect it.
        # ---------------------------------------------------------------
        existing_user_id = self._existing_user_id(wm)
        if existing_user_id:
            wm.auth_user_id = existing_user_id  # cache for downstream
            return self._post_auth_action(wm)

        # ---------------------------------------------------------------
        # Placeholder check (only applies to find_user_id_by_email)
        # ---------------------------------------------------------------
        if action.name == "find_user_id_by_email":
            email_val = str(action.args.get("email", ""))
            is_placeholder = (
                not email_val
                or _PLACEHOLDER_EMAIL_RE.match(email_val)
                or "@example." in email_val.lower()
                or "@test." in email_val.lower()
            )
            if not is_placeholder:
                return None  # model has a real email — let it proceed

        # ---------------------------------------------------------------
        # Path 2: Auth has been abandoned (refused / ask budget exhausted).
        # Don't ask again.  If goal is no-auth, redirect; else emit a
        # one-shot FINAL (guarded so we don't repeat it).
        # ---------------------------------------------------------------
        all_user_text = " ".join(wm.user_facts).lower()
        user_refused = any(phrase in all_user_text for phrase in _AUTH_REFUSAL_PHRASES)
        if user_refused or wm.auth_ask_count >= _MAX_AUTH_ASKS:
            wm.auth_abandoned = True

        if wm.auth_abandoned:
            if self._is_no_auth_query(wm):
                # Pivot to no-auth pathway.
                return ProposedAction(
                    name="list_all_product_types",
                    args={},
                    declared_class=RiskClass.READ,
                    declared_pre=["no-auth query"],
                    declared_post=["product types listed"],
                    informational_intent="answer product query without auth",
                    raw_thought=(
                        "Auth abandoned + product query → list_all_product_types."
                    ),
                    user_text="",
                    raw_response="",
                )
            # Otherwise emit a FINAL exactly once.
            if wm.auth_giveup_emitted:
                # We've already said this — break the loop by returning a
                # different short FINAL.
                return ProposedAction(
                    name=RESPOND_TOOL_NAME,
                    args={},
                    declared_class=RiskClass.FINAL,
                    declared_pre=[],
                    declared_post=[],
                    informational_intent="auth abandoned — terminate",
                    raw_thought="Auth abandoned, already gave up once — terminating.",
                    user_text="Is there anything else I can help you with?",
                    raw_response="",
                )
            wm.auth_giveup_emitted = True
            if user_refused:
                msg = (
                    "I understand. I'm unable to access account-specific data "
                    "without verifying your identity, but if you have a general "
                    "question I'm happy to help."
                )
            else:
                msg = (
                    "I'm having trouble verifying your identity with the "
                    "information provided. Please try again with your email "
                    "address or the name and ZIP code on your account."
                )
            return ProposedAction(
                name=RESPOND_TOOL_NAME,
                args={},
                declared_class=RiskClass.FINAL,
                declared_pre=[],
                declared_post=[],
                informational_intent="auth abandoned",
                raw_thought="Auth override: abandoning auth flow.",
                user_text=msg,
                raw_response="",
            )

        # ---------------------------------------------------------------
        # Path 3: No PII at all AND the goal is a no-auth query (product
        # query) → redirect to list_all_product_types.  This is a critical
        # fix: without it, the model's reflexive find_user_id_by_email gets
        # converted to "please give us your credentials" even though the
        # user only wanted to know how many t-shirts there are.
        # ---------------------------------------------------------------
        in_response_to_ask = wm.auth_ask_count > 0
        evidence = wm.user_facts
        real_email = self._extract_real_email(evidence)
        name_pair = self._extract_name_pair(evidence, in_response_to_ask)
        zip_code = self._extract_zip(evidence, wm.auth_failed_zips)
        has_any_pii = bool(real_email or name_pair or zip_code)

        if not has_any_pii and self._is_no_auth_query(wm):
            return ProposedAction(
                name="list_all_product_types",
                args={},
                declared_class=RiskClass.READ,
                declared_pre=["product query, no auth needed"],
                declared_post=["product types listed"],
                informational_intent="answer product query directly",
                raw_thought=(
                    "Auth override: product query with no PII → "
                    "list_all_product_types instead of asking for auth."
                ),
                user_text="",
                raw_response="",
            )

        # ---------------------------------------------------------------
        # Path 4: Use the best evidence we have.
        # ---------------------------------------------------------------
        if real_email:
            return ProposedAction(
                name="find_user_id_by_email",
                args={"email": real_email},
                declared_class=RiskClass.READ,
                declared_pre=["user provided email"],
                declared_post=["user_id retrieved"],
                informational_intent="authenticate via user-provided email",
                raw_thought="Auth override: using real email from user message.",
                user_text="",
                raw_response="",
            )

        if name_pair and zip_code:
            return ProposedAction(
                name="find_user_id_by_name_zip",
                args={
                    "first_name": name_pair[0],
                    "last_name": name_pair[1],
                    "zip": zip_code,
                },
                declared_class=RiskClass.READ,
                declared_pre=["user provided name and zip"],
                declared_post=["user_id retrieved"],
                informational_intent="authenticate via name+zip",
                raw_thought="Auth override: using user-provided name+zip.",
                user_text="",
                raw_response="",
            )

        if name_pair and wm.auth_failed_zips:
            wm.auth_ask_count += 1
            failed = wm.auth_failed_zips[-1]
            return ProposedAction(
                name=RESPOND_TOOL_NAME,
                args={},
                declared_class=RiskClass.ASK_USER,
                declared_pre=[],
                declared_post=["user provides corrected zip code"],
                informational_intent="ask user to re-verify zip code",
                raw_thought=f"Auth override: ZIP {failed} not found — asking for re-verification.",
                user_text=(
                    f"I wasn't able to find an account for {name_pair[0]} {name_pair[1]} "
                    f"with ZIP code {failed}. Could you double-check your ZIP code "
                    "and provide the correct one?"
                ),
                raw_response="",
            )

        if name_pair:
            wm.auth_ask_count += 1
            return ProposedAction(
                name=RESPOND_TOOL_NAME,
                args={},
                declared_class=RiskClass.ASK_USER,
                declared_pre=[],
                declared_post=["user provides zip code"],
                informational_intent="ask user for zip code",
                raw_thought="Auth override: have name, need zip code.",
                user_text=(
                    f"Thank you, {name_pair[0]}! To verify your identity, "
                    "could you also provide your ZIP code?"
                ),
                raw_response="",
            )

        # No usable identity info AND the goal needs auth — ask once.
        wm.auth_ask_count += 1
        return ProposedAction(
            name=RESPOND_TOOL_NAME,
            args={},
            declared_class=RiskClass.ASK_USER,
            declared_pre=[],
            declared_post=["user provides email or name+zip"],
            informational_intent="ask for authentication credentials",
            raw_thought="Auth override: need authentication credentials.",
            user_text=(
                "To look up your account, could you please provide your "
                "email address, or your full name and ZIP code?"
            ),
            raw_response="",
        )

    # ------------------------------------------------------------------
    # Product-ID / product-list helpers
    # ------------------------------------------------------------------
    def _resolve_product_id_name(
        self,
        action: ProposedAction,
        wm: "WorkingMemory",
    ) -> Optional[ProposedAction]:
        """If the proposer passes a type-name string (e.g. 'T-Shirt') where
        ``get_product_details`` expects a numeric ``product_id``, resolve the
        ID from db_facts that were populated when ``list_all_product_types``
        ran previously.

        db_facts entries look like: ``T-Shirt=6086499569`` or just ``6086499569``.
        We also absorb bare product names from the observation list.

        Returns a corrected action, or None if no fix is needed / possible.
        """
        if action.name != "get_product_details":
            return None
        pid = action.args.get("product_id")
        if pid is None:
            return None
        pid_str = str(pid).strip()
        # Already a numeric ID? Nothing to do.
        if re.fullmatch(r"\d+", pid_str):
            return None

        # Try to find a db_facts entry of the form "<type_name>=<numeric_id>"
        # where the type_name matches pid_str (case-insensitive prefix match).
        pid_lower = pid_str.lower()
        for fact in wm.db_facts:
            if "=" not in fact:
                continue
            name_part, _, id_part = fact.partition("=")
            if not re.fullmatch(r"\d+", id_part.strip()):
                continue
            if name_part.strip().lower() == pid_lower or pid_lower in name_part.strip().lower():
                numeric_id = int(id_part.strip())
                new_args = dict(action.args)
                new_args["product_id"] = numeric_id
                return ProposedAction(
                    name=action.name,
                    args=new_args,
                    declared_class=action.declared_class,
                    declared_pre=action.declared_pre,
                    declared_post=action.declared_post,
                    informational_intent=action.informational_intent,
                    raw_thought=(
                        f"Product-ID override: resolved '{pid_str}' → {numeric_id} "
                        "from db_facts."
                    ),
                    user_text=action.user_text,
                    raw_response=action.raw_response,
                )
        return None

    def _advance_after_product_list(
        self,
        action: ProposedAction,
        wm: "WorkingMemory",
    ) -> Optional[ProposedAction]:
        """If ``list_all_product_types`` is about to be repeated (already in
        recent_signatures) and the user mentioned a product type that we can
        resolve to a numeric ID in db_facts, skip straight to
        ``get_product_details`` instead.

        This avoids the infinite list_all_product_types → repeat_loop → retry
        cycle that wastes the entire step budget without making progress.
        """
        if action.name != "list_all_product_types":
            return None
        # Check whether this action is already in recent_signatures.
        sig = action.signature()
        if sig not in wm.recent_signatures:
            return None  # first call — let it run normally

        # It's a repeat.  Try to find a product ID we can use directly.
        # Scan user_facts for any word that appears as a key in db_facts
        # (i.e. was returned from a previous list_all_product_types call).
        all_user = " ".join(wm.user_facts)
        for fact in wm.db_facts:
            if "=" not in fact:
                continue
            name_part, _, id_part = fact.partition("=")
            name_part = name_part.strip()
            id_part = id_part.strip()
            if not re.fullmatch(r"\d+", id_part):
                continue
            # Case-insensitive check: does the user message mention this type?
            if name_part.lower() in all_user.lower():
                numeric_id = int(id_part)
                return ProposedAction(
                    name="get_product_details",
                    args={"product_id": numeric_id},
                    declared_class=RiskClass.READ,
                    declared_pre=["product types already listed"],
                    declared_post=["product details retrieved"],
                    informational_intent=f"get details for {name_part}",
                    raw_thought=(
                        f"Product-list advance: user mentioned '{name_part}', "
                        f"resolved to ID {numeric_id} — fetching details directly."
                    ),
                    user_text="",
                    raw_response="",
                )
        return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _schema_for(self, action: ProposedAction) -> ToolEffectSchema:
        """Return the schema for ``action.name``, synthesising one for unknown
        tools so the gate logic always has something to check."""
        if action.name in self.schemas:
            return self.schemas[action.name]
        # Synthesise a minimal schema using the declared class.
        from .schemas import ToolEffectSchema
        return ToolEffectSchema(
            name=action.name,
            cls=action.declared_class,
            irreversible=is_irreversible_or_final(action.declared_class),
            preconditions=[],
            postconditions=[],
            arg_id_fields=[],
            param_properties={},
            required_params=[],
            description="(unknown tool — synthesised schema)",
        )

    def _respond(self, env, content: str):
        """Issue a ``respond`` action to the env. Returns the env response,
        or ``None`` if the env raised a non-recoverable error."""
        msg = content.strip()[:RESPOND_MAX_CHARS] or "Is there anything else I can help with?"
        try:
            return env.step(Action(name=RESPOND_TOOL_NAME, kwargs={"content": msg}))
        except Exception as env_exc:
            if _is_context_overflow(env_exc):
                return None
            raise


__all__ = ["CargoAgent"]
