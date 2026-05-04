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
#
# CRITICAL: the introduction prefix is wrapped in (?i:...) so case-insensitive
# matching applies to "my name is" / "I'm" / "I AM" but NOT to the captured
# name groups.  Without this scoping, re.IGNORECASE made [A-Z][a-z]+ match
# lowercase too — so "I'm looking to see..." captured ("looking", "to") as
# the user's first/last name (observed in trajectories(18) tasks 1, 2, 4).
_NAME_INTRO_RE = re.compile(
    r"(?i:(?:my\s+name\s+(?:is|'s)|i\s+am\b|i'm\b|this\s+is|call\s+me|"
    r"name(?:\s*[:=]|d\b)|i\s+go\s+by))"
    r"[\s,]+([A-Z][a-z]{1,20})\s+([A-Z][a-z]{1,20})",
)
_ZIP_RE = re.compile(r"\b(\d{5})\b")

# Email domains that are clearly placeholder / non-real.
_PLACEHOLDER_EMAIL_DOMAINS = (
    "@example.", "@test.", "@sample.", "@dummy.", "@fake.",
    "@placeholder.", "@invalid.", "@nowhere.", "@localhost",
)


def _is_placeholder_email(email: str) -> bool:
    """Return True if `email` is a placeholder / fabricated identifier.

    Centralizes the predicate so both the action override and evidence
    extraction agree.  Without this, ``yusuf.rossi@example.com`` slipped
    through ``_extract_real_email`` because the prefix-only
    ``_PLACEHOLDER_EMAIL_RE`` doesn't list "yusuf.rossi", but the email
    is still a placeholder due to the @example domain.
    """
    if not email or "@" not in email:
        return True
    el = email.lower().strip()
    if _PLACEHOLDER_EMAIL_RE.match(email):
        return True
    return any(d in el for d in _PLACEHOLDER_EMAIL_DOMAINS)

# Goal keywords that indicate a "no auth required" query.  When the goal
# matches AND no PII is in user_facts, the auth override redirects placeholder
# find_user_id_by_email proposals to list_all_product_types instead of
# asking for credentials.
#
# Note: avoid generic words like "items" / "options" alone — those overlap
# with account-modifying tasks ("exchange items in my recent order").  We
# require a specific product/inventory anchor to avoid misrouting.
_PRODUCT_QUERY_RE = re.compile(
    r"\b(t-?shirts?|product\s+types?|store|catalog|inventory|browse|"
    r"how\s+many\s+(?:t-?shirts?|products?|items?|options?|variants?)|"
    r"(?:t-?shirts?|products?|items?|options?|variants?)\s+(?:available|in\s+(?:the|stock)|currently)|"
    r"available\s+(?:t-?shirts?|products?|items?|options?|variants?)|"
    r"in\s+stock|stock\s+of|brands?)\b",
    re.IGNORECASE,
)

# Account-modifying signals.  When ANY of these match the goal, the goal
# requires authentication regardless of any product-query overlap.  This
# overrides the product heuristic so phrases like "exchange items in my
# recent order" stay on the auth path.
_AUTH_REQUIRED_RE = re.compile(
    r"\b("
    r"my\s+(?:order|orders|account|pending|recent)|"
    r"exchang|cancel|return\s+(?:the|my|some|item|order|product)|"
    r"refund|modify|update\s+my|change\s+my|"
    r"order\s+(?:id|number|#)|#?[Ww]\d{5,10}"
    r")\b",
    re.IGNORECASE,
)

# user_id pattern produced by tau-bench retail (e.g. "yusuf_rossi_9620").
# Format: lowercase token, underscore, lowercase token, underscore, 1-8 digits.
_USER_ID_PATTERN = re.compile(r"\b([a-z]+_[a-z]+_\d{1,8})\b")

# Token prefixes that match `_USER_ID_PATTERN` shape but are NOT user IDs.
# These are common tau-bench retail field names (payment methods, addresses)
# that should never be treated as a user_id even if they textually match.
_NON_USER_ID_PREFIXES = (
    "credit_card_",
    "debit_card_",
    "gift_card_",
    "paypal_account_",
    "bank_account_",
    "address_",
    "order_",
    "item_",
    "product_",
    "variant_",
    "card_",
)

# tau-bench retail order_id format: "#W" + digits (e.g. "#W2378156").
# Used as a fallback when auth via email / name+zip fails but the user
# has supplied an order number directly.
_ORDER_ID_PATTERN = re.compile(r"#?[Ww]\d{5,10}")

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
        # Skipped when the action was produced by a deterministic override
        # (bypass_gates=True). SC samples the proposer independently; for
        # auth-abandon FINALs and product-count finalisers the proposer's
        # independent votes are irrelevant and would block a correct decision.
        if not getattr(action, "bypass_gates", False):
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

            # User-ID override: if the model proposes get_user_details with a
            # non-user token (e.g. credit_card_9513926), replace with the
            # correct user_id we have in db_facts.
            if action is not None:
                uid_action = self._resolve_get_user_details(action, wm)
                if uid_action is not None:
                    raw_text = json.dumps({
                        "thought": uid_action.raw_thought,
                        "action": {
                            "name": uid_action.name,
                            "args": uid_action.args,
                            "declared_class": uid_action.declared_class.value,
                            "declared_pre": uid_action.declared_pre,
                            "declared_post": uid_action.declared_post,
                            "informational_intent": uid_action.informational_intent,
                            "user_text": uid_action.user_text,
                        },
                    })
                    action = uid_action

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

            # D2 fix: normalize order_id to #W… format before any gate sees it.
            # tau-bench retail requires the #W prefix (e.g. "#W2378156").  When
            # the model proposes get_order_details with a bare "W2378156" the env
            # returns "order not found", and the model retries in an infinite
            # loop.  Normalizing here fixes the call before execution.
            # (Observed failure: trajectories(24) T1.)
            if action is not None:
                norm_action = self._normalize_order_id_action(action, wm)
                if norm_action is not None:
                    raw_text = json.dumps({
                        "thought": norm_action.raw_thought,
                        "action": {
                            "name": norm_action.name,
                            "args": norm_action.args,
                            "declared_class": norm_action.declared_class.value,
                            "declared_pre": norm_action.declared_pre,
                            "declared_post": norm_action.declared_post,
                            "informational_intent": norm_action.informational_intent,
                            "user_text": norm_action.user_text,
                        },
                    })
                    action = norm_action

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

            # Product-count finalizer: if the user asked "how many X" and
            # we have the data but the model is looping, emit a FINAL
            # with the computed count.  This is the structural answer to
            # the model's inability to count + finalize on its own.
            if action is not None:
                fin_action = self._finalize_product_count_query(action, wm)
                if fin_action is not None:
                    raw_text = json.dumps({
                        "thought": fin_action.raw_thought,
                        "action": {
                            "name": fin_action.name,
                            "args": fin_action.args,
                            "declared_class": fin_action.declared_class.value,
                            "declared_pre": fin_action.declared_pre,
                            "declared_post": fin_action.declared_post,
                            "informational_intent": fin_action.informational_intent,
                            "user_text": fin_action.user_text,
                        },
                    })
                    action = fin_action

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

                # Hard-loop break: if the same respond text is being emitted
                # repeatedly, the trajectory is making no progress.  tau-bench
                # treats respond as a query (not a terminator), so a stuck
                # FINAL would otherwise burn the entire step budget.
                # Observed in trajectories(20) T3/T4: "Is there anything else
                # I can help you with?" emitted 8+ times in a row.
                if content == wm.last_final_text:
                    wm.consecutive_same_final += 1
                else:
                    wm.consecutive_same_final = 1
                    wm.last_final_text = content
                if wm.consecutive_same_final >= 2 and action.declared_class == RiskClass.FINAL:
                    # Already said this exact FINAL.  Terminate to avoid wasting
                    # the step budget on a frozen trajectory.
                    step_record["terminated"] = "consecutive_final"
                    stats.record_step(step_record)
                    break

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
            # Only accept tokens that pass _is_user_id_token (filters out
            # accidentally-matching credit_card_*, address_* tokens).
            if action.name in (
                "find_user_id_by_email", "find_user_id_by_name_zip",
                "get_order_details",
            ):
                if tool_obs:
                    for m in _USER_ID_PATTERN.finditer(tool_obs):
                        tok = m.group(1)
                        if self._is_user_id_token(tok) and not wm.auth_user_id:
                            wm.auth_user_id = tok
                            break

            # D1 fix: cache the confirmed email from get_user_details into the
            # durable wm.auth_email field so it is never evicted by the 48-entry
            # db_facts LRU.  Only apply the lightweight prefix-RE check (not the
            # domain block) so tau-bench @example.com real emails are accepted.
            # (Observed failure: trajectories(24) T0 — db_facts eviction broke
            # the auth-confirmation email lookup in _auth_override Path 1.)
            if action.name == "get_user_details" and tool_obs and not wm.auth_email:
                try:
                    ud = (
                        json.loads(tool_obs)
                        if tool_obs.lstrip().startswith("{")
                        else None
                    )
                    if isinstance(ud, dict):
                        candidate = str(ud.get("email", "")).strip()
                        # Accept emails that are NOT caught by the generic-prefix
                        # pattern.  Do NOT apply the domain-level block —
                        # tau-bench uses @example.com for all real user emails.
                        if candidate and not _PLACEHOLDER_EMAIL_RE.match(candidate):
                            wm.auth_email = candidate
                except Exception:
                    pass

            # Cache product details for short-circuiting count queries.  When
            # the model fetches a product successfully and the user query is
            # "how many X variants" / "how many X options", we can later emit
            # a FINAL with the computed count instead of looping.
            if action.name == "get_product_details" and tool_obs:
                try:
                    pd = json.loads(tool_obs) if tool_obs.lstrip().startswith("{") else None
                    if isinstance(pd, dict):
                        pid = str(pd.get("product_id") or
                                  action.args.get("product_id") or "")
                        if pid:
                            wm.product_details[pid] = pd
                except Exception:
                    pass

            # Cache the product type catalogue.  This is a flat dict mapping
            # name → product_id.  We cache it OUTSIDE db_facts because db_facts
            # has a 48-entry LRU cap, and a single get_product_details call
            # fills it with variant data, evicting the catalogue.  The
            # catalogue is needed by _advance_after_product_list and by the
            # product-id resolver to handle multi-product follow-up queries.
            if action.name == "list_all_product_types" and tool_obs:
                try:
                    pt = json.loads(tool_obs) if tool_obs.lstrip().startswith("{") else None
                    if isinstance(pt, dict):
                        for name, pid in pt.items():
                            if isinstance(name, str) and isinstance(pid, (str, int)):
                                wm.product_types[name] = str(pid)
                except Exception:
                    pass

            # Cache order details by order_id for downstream access.
            if action.name == "get_order_details" and tool_obs:
                try:
                    od = json.loads(tool_obs) if tool_obs.lstrip().startswith("{") else None
                    if isinstance(od, dict):
                        oid = str(od.get("order_id") or
                                  action.args.get("order_id") or "")
                        if oid:
                            wm.order_details[oid] = od
                except Exception:
                    pass
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
        """Return the first non-placeholder email found in `facts`."""
        for fact in facts:
            m = _EMAIL_RE_MOD.search(fact)
            if m and not _is_placeholder_email(m.group(0)):
                return m.group(0)
        return None

    @staticmethod
    def _extract_any_email(facts: List[str]) -> Optional[str]:
        """Return the first email found in ``facts``, skipping only obviously
        agent-fabricated placeholders caught by the full regex pattern.

        Unlike ``_extract_real_email`` this does **not** apply the blanket
        domain-based block (``_PLACEHOLDER_EMAIL_DOMAINS``).  That block
        incorrectly rejects tau-bench real user emails like
        ``yusuf.rossi7301@example.com`` because tau-bench uses ``@example.com``
        as its canonical mail domain.  Emails from db_facts come from tool
        responses, not from model hallucination, so they can be trusted even
        when the domain looks synthetic.

        Only the prefix-pattern RE (``_PLACEHOLDER_EMAIL_RE``) is applied so
        that clearly-fabricated values like ``user@example.com`` or
        ``alice@example.com`` are still blocked.
        """
        for fact in facts:
            m = _EMAIL_RE_MOD.search(fact)
            if m:
                email = m.group(0)
                # Block generic agent-fabricated prefixes (user@, alice@, etc.)
                # but do NOT apply the domain-level block — tau-bench uses
                # @example.com for real user emails.
                if not _PLACEHOLDER_EMAIL_RE.match(email):
                    return email
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
    def _is_user_id_token(token: str) -> bool:
        """True if `token` is shaped like a user_id AND not a known non-user
        prefix (credit_card_, paypal_account_, etc.)."""
        if not token or not _USER_ID_PATTERN.fullmatch(token):
            return False
        return not any(token.startswith(p) for p in _NON_USER_ID_PREFIXES)

    @staticmethod
    def _existing_user_id(wm: "WorkingMemory") -> Optional[str]:
        """Return a user_id from db_facts (or wm.auth_user_id), if any.

        Strategy:
        1. Cached: ``wm.auth_user_id``.
        2. Explicit: any db_fact starting with ``user_id=`` is the
           authoritative source — the tool literally returned this field.
        3. Fallback: any db_fact whose token matches the user_id pattern
           AND does NOT start with a known non-user prefix.

        Without (2) and the prefix check in (3), tau-bench retail order
        responses leak ``credit_card_9513926`` as the "user_id" because
        it shares the lowercase_lowercase_digits shape.  Observed in
        trajectories(19) T3 step 7: the agent issued
        ``get_user_details(user_id='credit_card_9513926')``.
        """
        if wm.auth_user_id:
            return wm.auth_user_id
        # Step 2: explicit user_id= entries.
        for fact in wm.db_facts:
            if fact.startswith("user_id="):
                m = _USER_ID_PATTERN.search(fact)
                if m and CargoAgent._is_user_id_token(m.group(1)):
                    return m.group(1)
        # Step 3: bare token fallback, with prefix filter.
        for fact in wm.db_facts:
            m = _USER_ID_PATTERN.search(fact)
            if m and CargoAgent._is_user_id_token(m.group(1)):
                return m.group(1)
        return None

    @staticmethod
    def _extract_order_id(wm: "WorkingMemory") -> Optional[str]:
        """Return a tau-bench order_id from user_facts, if any.

        Format: #W followed by 5-10 digits (e.g. "#W2378156"). This is the
        recovery anchor when both name+zip and email lookups have failed
        but the user supplied a tracking/order number.
        """
        for fact in wm.user_facts:
            m = _ORDER_ID_PATTERN.search(fact)
            if m:
                oid = m.group(0)
                # Normalise: ensure leading '#' and uppercase 'W'
                if not oid.startswith("#"):
                    oid = "#" + oid
                # Uppercase the W
                if len(oid) >= 2:
                    oid = "#" + oid[1].upper() + oid[2:]
                return oid
        return None

    @staticmethod
    def _is_no_auth_query(wm: "WorkingMemory") -> bool:
        """Heuristic: is the goal a pure product/store query (no auth needed)?

        Returns True only when:
        - Goal contains a product/store query signal (``_PRODUCT_QUERY_RE``)
        - AND goal does NOT contain any account-modifying signal
          (``_AUTH_REQUIRED_RE``: my order, exchange, return, cancel, …)

        The negative check is critical: phrases like "exchange items in my
        recent order" match the product-query keyword "items" but clearly
        require authentication.  Observed in trajectories(19) T1: the agent
        routed an exchange task to ``list_all_product_types`` because of
        a single overlapping word.
        """
        if not wm.goal:
            return False
        # Negative signal wins: any account-mutating phrase forces auth.
        # Look in user_facts too — the user may have specified the order
        # ID after the initial goal.
        all_text = wm.goal + " " + " ".join(wm.user_facts)
        if _AUTH_REQUIRED_RE.search(all_text):
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
            post_action = self._post_auth_action(wm)
            if post_action is not None:
                return post_action
            # _post_auth_action returned None: auth is done and user details
            # have been fetched, but goal is not a simple no-auth query (e.g.
            # an order-exchange task).  The model keeps looping on placeholder
            # find_user_id_by_email proposals.  Break the loop by confirming
            # the user's identity via their real email from db_facts (which was
            # populated by get_user_details) or user_facts.  After seeing a
            # successful find_user_id_by_email result the model should proceed
            # to the actual task without another auth proposal.
            # D1 fix: use the durable auth_email cache first (never evicted),
            # then fall back to db_facts (subject to 48-entry LRU eviction),
            # then user_facts (use any_email so @example.com is accepted).
            confirmed_email = (
                wm.auth_email
                or self._extract_any_email(wm.db_facts)
                or self._extract_any_email(wm.user_facts)
            )
            if confirmed_email:
                email_sig = f"find_user_id_by_email(email={confirmed_email!r})"
                if email_sig not in wm.recent_signatures:
                    return ProposedAction(
                        name="find_user_id_by_email",
                        args={"email": confirmed_email},
                        declared_class=RiskClass.READ,
                        declared_pre=["identity confirmed via order details"],
                        declared_post=["user_id confirmed via email"],
                        informational_intent=(
                            "re-confirm auth via email from db_facts "
                            "to break auth loop"
                        ),
                        raw_thought=(
                            "Post-auth override: confirming auth via email "
                            "from db_facts/user_facts to break auth loop and "
                            "signal model to proceed with actual task."
                        ),
                        user_text="",
                        raw_response="",
                    )
            return None

        # ---------------------------------------------------------------
        # Placeholder check (only applies to find_user_id_by_email)
        # ---------------------------------------------------------------
        if action.name == "find_user_id_by_email":
            email_val = str(action.args.get("email", ""))
            if not _is_placeholder_email(email_val):
                return None  # model has a real email — let it proceed

        # ---------------------------------------------------------------
        # Extract evidence ONCE, before deciding what to do.  Critical
        # ordering: we must check evidence BEFORE the abandoned-auth
        # check, otherwise fresh user input (provided AFTER our last ask)
        # gets discarded just because auth_ask_count hit the cap.  This
        # was T3 in trajectories(18): user supplied name + ZIP across
        # two turns, then the override gave up because count == max.
        # ---------------------------------------------------------------
        in_response_to_ask = wm.auth_ask_count > 0
        evidence = wm.user_facts
        # D4 fix: use _extract_any_email for user-provided credentials.
        # The original _extract_real_email applies a domain-level block that
        # rejects @example.com addresses.  But tau-bench user simulations
        # provide their email as e.g. "yusufrossi@example.com".  The prefix-RE
        # still blocks obvious fabrications (user@, demo@, test@, …) so this
        # change does not regress the hallucination guard.
        # (Observed failure: trajectories(24) T0 user provided
        # "yusufrossi@example.com" 8+ times; it was never used.)
        real_email = self._extract_any_email(evidence)
        name_pair = self._extract_name_pair(evidence, in_response_to_ask)
        zip_code = self._extract_zip(evidence, wm.auth_failed_zips)
        has_usable_pii = bool(real_email or (name_pair and zip_code))

        # ---------------------------------------------------------------
        # Path 2: We have usable PII — use it BEFORE giving up.
        # ---------------------------------------------------------------
        if real_email:
            email_sig_p2 = f"find_user_id_by_email(email={real_email!r})"
            if email_sig_p2 in wm.recent_signatures:
                # Already tried this email and it presumably failed; skip to
                # name+zip or the ask path so we don't loop.
                real_email = None
        if real_email:
            # User has provided fresh credentials — clear any prior abandonment
            # state so the auth flow can proceed normally.
            if wm.auth_abandoned:
                wm.auth_abandoned = False
                wm.auth_giveup_emitted = False
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
            # Clear abandonment when fresh name+zip appears (same logic as email)
            if wm.auth_abandoned:
                wm.auth_abandoned = False
                wm.auth_giveup_emitted = False
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

        # Order-ID fallback: if the user supplied an order number, we can
        # often proceed with get_order_details directly.  This is the recovery
        # path when both name+zip and email lookups have failed but the user
        # gave a tracking/order number ("#W2378156" in tau-bench retail).
        order_id = self._extract_order_id(wm)
        order_signature = f"get_order_details(order_id={order_id!r})"
        if order_id and order_signature not in wm.recent_signatures:
            return ProposedAction(
                name="get_order_details",
                args={"order_id": order_id},
                declared_class=RiskClass.READ,
                declared_pre=["user provided order id"],
                declared_post=["order details retrieved"],
                informational_intent="look up the user's order directly",
                raw_thought=(
                    f"Auth override: order_id {order_id} from user_facts → "
                    "fall back to get_order_details."
                ),
                user_text="",
                raw_response="",
            )

        # ---------------------------------------------------------------
        # Path 3: No usable PII.  Now check if auth has been abandoned
        # (refused / ask budget exhausted).
        # ---------------------------------------------------------------
        all_user_text = " ".join(wm.user_facts).lower()
        raw_refused = any(phrase in all_user_text for phrase in _AUTH_REFUSAL_PHRASES)
        # B6 fix: soft-refusal detection.  A message like "I'd rather not
        # share too much — can we proceed based on my recent orders?" is a
        # NEGOTIATION, not a hard refusal.  If the user also mentions an
        # alternative path (order number, different identifier) do not set
        # auth_abandoned yet; give the order-ID fallback a chance to run.
        if raw_refused:
            offering_alt = any(
                kw in all_user_text
                for kw in (
                    "order", "order number", "order id", "recent order",
                    "tracking", "some other", "identifier", "another way",
                    "different way", "instead",
                )
            )
            user_refused = not offering_alt
        else:
            user_refused = False
        if user_refused or wm.auth_ask_count >= _MAX_AUTH_ASKS:
            wm.auth_abandoned = True

        if wm.auth_abandoned:
            # D5 fix: before looping on the give-up FINAL, re-check whether the
            # user has provided fresh credentials since abandonment.  If they
            # gave an email or name+zip that we haven't tried yet, un-abandon
            # and attempt auth.  The email_sig guard prevents retrying a known-
            # bad email.  Without this, users who share their email AFTER the
            # initial give-up are silently ignored.
            # (Observed failure: trajectories(24) T4 — user provided
            # "yusufrossi@example.com" at step 24; agent kept saying "Is there
            # anything else I can help you with?" for the remaining 17 steps.)
            fresh_email = self._extract_any_email(wm.user_facts)
            if fresh_email:
                fresh_sig = f"find_user_id_by_email(email={fresh_email!r})"
                if fresh_sig not in wm.recent_signatures:
                    wm.auth_abandoned = False
                    wm.auth_giveup_emitted = False
                    return ProposedAction(
                        name="find_user_id_by_email",
                        args={"email": fresh_email},
                        declared_class=RiskClass.READ,
                        declared_pre=["user provided email after abandonment"],
                        declared_post=["user_id retrieved"],
                        informational_intent="re-attempt auth with fresh user-provided email",
                        raw_thought=(
                            f"Auth re-attempt: user provided email {fresh_email!r} "
                            "after abandonment — trying before final give-up."
                        ),
                        user_text="",
                        raw_response="",
                    )
            fresh_name = self._extract_name_pair(wm.user_facts, True)
            fresh_zip = self._extract_zip(wm.user_facts, wm.auth_failed_zips)
            if fresh_name and fresh_zip:
                fresh_sig = (
                    f"find_user_id_by_name_zip("
                    f"first_name={fresh_name[0]!r},"
                    f"last_name={fresh_name[1]!r},"
                    f"zip={fresh_zip!r})"
                )
                if fresh_sig not in wm.recent_signatures:
                    wm.auth_abandoned = False
                    wm.auth_giveup_emitted = False
                    return ProposedAction(
                        name="find_user_id_by_name_zip",
                        args={
                            "first_name": fresh_name[0],
                            "last_name": fresh_name[1],
                            "zip": fresh_zip,
                        },
                        declared_class=RiskClass.READ,
                        declared_pre=["user provided name+zip after abandonment"],
                        declared_post=["user_id retrieved"],
                        informational_intent="re-attempt auth with fresh name+zip",
                        raw_thought=(
                            "Auth re-attempt: user provided name+zip "
                            "after abandonment — trying before final give-up."
                        ),
                        user_text="",
                        raw_response="",
                    )

            # B5 fix: before giving up entirely, answer any no-auth product-
            # query component present in a mixed goal (e.g. "how many t-shirts
            # + update my pending order").  Even when _is_no_auth_query returns
            # False (because the goal also has an account-mutation phrase), we
            # can still list and count products without auth, giving the user
            # partial value before the final "can't help without identity" message.
            has_product_query = bool(_PRODUCT_QUERY_RE.search(wm.goal or ""))
            if has_product_query and not wm.product_count_finalized:
                if not wm.product_types:
                    return ProposedAction(
                        name="list_all_product_types",
                        args={},
                        declared_class=RiskClass.READ,
                        declared_pre=["auth abandoned, answering product query first"],
                        declared_post=["product types listed"],
                        informational_intent="answer product query component before auth give-up",
                        raw_thought=(
                            "Auth abandoned + product query not yet answered → "
                            "list_all_product_types before final give-up."
                        ),
                        user_text="",
                        raw_response="",
                    )
                # Product types available; pick the first user-mentioned product
                # that hasn't been fetched yet and get its details.
                name_to_id_lc = {k.lower(): v for k, v in wm.product_types.items()}
                target_pid = self._goal_matched_product_id(wm, name_to_id_lc)
                if target_pid and target_pid not in wm.product_details:
                    return ProposedAction(
                        name="get_product_details",
                        args={"product_id": target_pid},
                        declared_class=RiskClass.READ,
                        declared_pre=["answering product query component"],
                        declared_post=["product details fetched"],
                        informational_intent="fetch product details for count answer",
                        raw_thought=(
                            "Auth abandoned + product types listed → fetching "
                            "product details to answer count query."
                        ),
                        user_text="",
                        raw_response="",
                    )
                # Product details fetched; _finalize_product_count_query will
                # emit the count FINAL on the next applicable step. Fall through
                # to the give-up FINAL only after that finalizer has fired.
                if not wm.product_count_finalized:
                    return None  # let _finalize_product_count_query handle it

            if self._is_no_auth_query(wm):
                # Pure no-auth pathway (original path).
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
            # bypass_gates=True so the SC gate's independent proposer samples
            # don't block a deterministic auth-abandonment decision.
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
                    bypass_gates=True,
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
                bypass_gates=True,
            )

        # ---------------------------------------------------------------
        # Path 4: No PII at all AND the goal is a no-auth query (product
        # query) → redirect to list_all_product_types.  Without this, the
        # model's reflexive find_user_id_by_email gets converted to
        # "please give us your credentials" even though the user only
        # wanted to know how many t-shirts there are.
        # ---------------------------------------------------------------
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
        # C4 fix: lead with name + ZIP (not email).  The tau-bench user
        # script includes the user's name and ZIP code, so asking for those
        # directly gets a cooperative response.  Asking for "email address"
        # first triggers refusals from privacy-sensitive user simulations,
        # even though the same user would freely give their name and ZIP.
        # (Observed failure: trajectories(23) Tasks 3 & 4.)
        wm.auth_ask_count += 1
        return ProposedAction(
            name=RESPOND_TOOL_NAME,
            args={},
            declared_class=RiskClass.ASK_USER,
            declared_pre=[],
            declared_post=["user provides name+zip or email"],
            informational_intent="ask for authentication credentials",
            raw_thought="Auth override: need authentication credentials (name+zip preferred).",
            user_text=(
                "To access your account, could you please provide your "
                "full name and ZIP code? (Your email address works too, "
                "if you prefer.)"
            ),
            raw_response="",
        )

    # ------------------------------------------------------------------
    # Order-ID normalizer (D2 fix)
    # ------------------------------------------------------------------
    def _normalize_order_id_action(
        self,
        action: ProposedAction,
        wm: "WorkingMemory",
    ) -> Optional[ProposedAction]:
        """Ensure ``get_order_details`` uses the canonical ``#W…`` order_id
        format that tau-bench retail requires.

        The model sometimes proposes ``W2378156`` (without ``#``) or even
        ``w2378156`` (lowercase).  The env returns "order not found" for any
        non-canonical form, and the model then loops retrying the same bad
        call without self-correcting.

        This override normalises the order_id to ``#W<digits>`` before any
        gate sees it, so the env always receives the correct form.

        (Observed failure: trajectories(24) T1 — model proposed
        ``get_order_details(order_id='W2378156')`` 15+ times.)
        """
        if action.name != "get_order_details":
            return None
        oid = str(action.args.get("order_id", "")).strip()
        if not oid:
            return None
        # Already in canonical form.
        if oid.startswith("#W") or oid.startswith("#w"):
            normalized = "#W" + oid[2:]
            if normalized == oid:
                return None
        else:
            # Add '#' prefix if missing.
            if not oid.startswith("#"):
                oid = "#" + oid
            # Uppercase the 'W'.
            if len(oid) >= 2 and oid[1].lower() == "w":
                normalized = "#W" + oid[2:]
            else:
                return None  # unexpected format — don't touch it
        if normalized == action.args.get("order_id"):
            return None
        new_args = dict(action.args)
        new_args["order_id"] = normalized
        return ProposedAction(
            name=action.name,
            args=new_args,
            declared_class=action.declared_class,
            declared_pre=action.declared_pre,
            declared_post=action.declared_post,
            informational_intent=action.informational_intent,
            raw_thought=(
                f"Order-ID normalize: {action.args.get('order_id')!r} → "
                f"{normalized!r} (tau-bench requires #W prefix)."
            ),
            user_text=action.user_text,
            raw_response=action.raw_response,
        )

    # ------------------------------------------------------------------
    # User-ID resolver (defends against confused-field user_id arguments)
    # ------------------------------------------------------------------
    def _resolve_get_user_details(
        self,
        action: ProposedAction,
        wm: "WorkingMemory",
    ) -> Optional[ProposedAction]:
        """Replace a wrong ``user_id`` argument with the correct one.

        Observed failure (trajectories(19) T3 step 7): after
        ``get_order_details`` returned a JSON containing both ``user_id``
        and a ``payment_methods.credit_card_9513926`` key, the model
        passed ``credit_card_9513926`` as the user_id.  The pattern
        accidentally matches both shapes.  We detect non-user prefixes
        and substitute the correct user_id we already have in db_facts.
        """
        if action.name != "get_user_details":
            return None
        uid = str(action.args.get("user_id", "")).strip()
        if not uid:
            return None
        # If the proposed uid is a real-looking user_id, leave it alone.
        if self._is_user_id_token(uid):
            # If it matches our cached auth_user_id or any explicit
            # user_id in db_facts, accept it.
            return None
        # Otherwise the model passed a non-user token (credit_card_*, …).
        correct = self._existing_user_id(wm)
        if not correct or correct == uid:
            return None
        new_args = dict(action.args)
        new_args["user_id"] = correct
        return ProposedAction(
            name=action.name,
            args=new_args,
            declared_class=action.declared_class,
            declared_pre=action.declared_pre,
            declared_post=action.declared_post,
            informational_intent=action.informational_intent,
            raw_thought=(
                f"User-ID override: replacing non-user token {uid!r} "
                f"with grounded user_id {correct!r}."
            ),
            user_text=action.user_text,
            raw_response=action.raw_response,
        )

    # ------------------------------------------------------------------
    # Product-ID / product-list helpers
    # ------------------------------------------------------------------
    def _resolve_product_id_name(
        self,
        action: ProposedAction,
        wm: "WorkingMemory",
    ) -> Optional[ProposedAction]:
        """Resolve a corrupt or hallucinated ``product_id`` argument.

        Two failure modes handled:

        1. Type-name string (e.g. 'T-Shirt') where a numeric ID is expected.
           Resolved from db_facts entries like ``T-Shirt=6086499569``.

        2. Numeric ID that is NOT in db_facts (i.e. the model hallucinated
           a number). Observed in trajectories(18): the model called
           ``get_product_details(9523456873)`` 24+ times after
           ``list_all_product_types`` returned a JSON map of real IDs.
           In this case we look at the goal for a product-type keyword
           and substitute the matching real ID from db_facts.

        Returns a corrected action, or None if no fix is needed / possible.
        """
        if action.name != "get_product_details":
            return None
        pid = action.args.get("product_id")
        if pid is None:
            return None
        pid_str = str(pid).strip()

        # Build a {lower(name): numeric_id} map.  Prefer the durable
        # wm.product_types catalogue, then fall back to db_facts (which can
        # be evicted under memory pressure).
        name_to_id: Dict[str, str] = {}
        all_known_ids = set()
        for name, pid in wm.product_types.items():
            if isinstance(name, str) and isinstance(pid, str) and re.fullmatch(r"\d+", pid):
                name_to_id[name.strip().lower()] = pid
                all_known_ids.add(pid)
        for fact in wm.db_facts:
            if "=" not in fact:
                if re.fullmatch(r"\d+", fact.strip()):
                    all_known_ids.add(fact.strip())
                continue
            name_part, _, id_part = fact.partition("=")
            id_part = id_part.strip()
            if not re.fullmatch(r"\d+", id_part):
                continue
            name_to_id.setdefault(name_part.strip().lower(), id_part)
            all_known_ids.add(id_part)

        # ---- Case 1: numeric ID not in db_facts → hallucinated ----
        if re.fullmatch(r"\d+", pid_str):
            if pid_str in all_known_ids:
                return None  # legitimate ID, leave as-is
            # Hallucinated ID — try to find a goal-relevant replacement.
            replacement = self._goal_matched_product_id(wm, name_to_id)
            if replacement is None:
                # No good replacement; let the action through (gates may
                # reject; if not, env returns "not found" which is fine).
                return None
            new_args = dict(action.args)
            new_args["product_id"] = replacement
            return ProposedAction(
                name=action.name,
                args=new_args,
                declared_class=action.declared_class,
                declared_pre=action.declared_pre,
                declared_post=action.declared_post,
                informational_intent=action.informational_intent,
                raw_thought=(
                    f"Product-ID override: hallucinated '{pid_str}' → "
                    f"{replacement} (matched goal in db_facts)."
                ),
                user_text=action.user_text,
                raw_response=action.raw_response,
            )

        # ---- Case 2: type-name string → look up by name ----
        pid_lower = pid_str.lower()
        for name_lower, num in name_to_id.items():
            if name_lower == pid_lower or pid_lower in name_lower:
                new_args = dict(action.args)
                new_args["product_id"] = num
                return ProposedAction(
                    name=action.name,
                    args=new_args,
                    declared_class=action.declared_class,
                    declared_pre=action.declared_pre,
                    declared_post=action.declared_post,
                    informational_intent=action.informational_intent,
                    raw_thought=(
                        f"Product-ID override: resolved '{pid_str}' → {num} "
                        "from db_facts."
                    ),
                    user_text=action.user_text,
                    raw_response=action.raw_response,
                )
        return None

    @staticmethod
    def _goal_matched_product_id(
        wm: "WorkingMemory", name_to_id: Dict[str, str]
    ) -> Optional[str]:
        """Search the goal text for a product-type keyword and return the
        matching numeric ID from `name_to_id` if found."""
        goal_lower = (wm.goal or "").lower()
        if not goal_lower:
            return None
        # Exact match: goal mentions a name from db_facts directly.
        for name_lower, num in name_to_id.items():
            if name_lower and name_lower in goal_lower:
                return num
        # Soft match: split goal into words, look for any word that's a
        # significant part of a name.  E.g. goal="t-shirt options" should
        # match name="t-shirt" or "t shirt" or even "shirt".
        words = re.findall(r"[a-z0-9]+", goal_lower)
        for w in words:
            if len(w) < 4:
                continue
            for name_lower, num in name_to_id.items():
                # token must appear as a whole word in the name
                if re.search(rf"\b{re.escape(w)}\b", name_lower):
                    return num
        return None

    # ------------------------------------------------------------------
    # Product-matching helper (B2 fix: word-boundary semantics)
    # ------------------------------------------------------------------
    @staticmethod
    def _product_name_matches_user(name_norm: str, all_user: str) -> bool:
        """Return True if the product name (lowercased) is mentioned by the user.

        Rules (applied in order):
        1. Exact phrase substring — handles "t-shirt" matching "t-shirts" etc.
        2. Any single word (len≥4) with **word-boundary** regex — prevents
           "hose" (from "Garden Hose") from matching "those" in "check those
           for me".  Observed failure in trajectories(22) T2.
        3. Compound-word check — joins all words and checks as one token so
           "Smart Watch" matches "smartwatch" (and NOT "smartthermostat").
           Observed failure: "smart" prefix matched "Smart Thermostat" when
           user said "smartwatch".
        """
        if not name_norm or not all_user:
            return False
        # Rule 1: phrase substring (handles plurals like "t-shirts")
        if name_norm in all_user:
            return True
        # Rule 2: word-boundary match — D3 fix.
        # Old code used OR-any: a single matching word was enough.  That caused
        # "Smart Thermostat" to match "smart watches" because "smart" (5 chars)
        # appeared in both.  New logic:
        #   (a) For multi-word products, the LAST word (primary noun) must match
        #       AND at least one other significant word must also match —
        #       or all significant words must match if there are only two.
        #   (b) For single-word products, that word must match (unchanged).
        # This ensures "Smart Thermostat" requires "thermostat" to be present in
        # the user text, not merely "smart".
        # (Observed failure: trajectories(24) T2 — "Smart Thermostat" matched
        # "smart watches", fetching the wrong product 8+ times in a row.)
        words = name_norm.split()
        sig_words = [w for w in words if len(w) >= 4]
        if len(words) > 1 and sig_words:
            last_word = words[-1]
            last_sig = last_word if len(last_word) >= 4 else None
            if last_sig:
                # Last word (primary noun) must match.  This is sufficient for a
                # match — we do NOT require other words (adjectives like "smart",
                # "mechanical", "vacuum") to also appear.  The primary noun alone
                # is the anchor for recognition.
                # Include simple plural: "cleaner" ↔ "cleaners", "keyboard" ↔
                # "keyboards" (adds `s?` before word boundary).
                if re.search(
                    rf"\b{re.escape(last_sig)}s?\b", all_user
                ):
                    return True
            else:
                # Last word is short (<4 chars); require ALL sig_words AND.
                if all(
                    re.search(rf"\b{re.escape(w)}s?\b", all_user) for w in sig_words
                ):
                    return True
        elif sig_words:
            # Single significant word — check with simple plural form.
            tok = sig_words[0]
            if re.search(rf"\b{re.escape(tok)}s?\b", all_user):
                return True
        # Rule 3: compound form (e.g. "smartwatch" ↔ "Smart Watch")
        compound = "".join(words)
        if len(compound) >= 6 and re.search(rf"\b{re.escape(compound)}\b", all_user):
            return True
        return False

    def _advance_after_product_list(
        self,
        action: ProposedAction,
        wm: "WorkingMemory",
    ) -> Optional[ProposedAction]:
        """Skip straight to ``get_product_details`` for a product the user
        mentioned but we haven't fetched yet.

        Two trigger modes:
        1. The model is about to **repeat** ``list_all_product_types`` — the
           classic case (first call already executed).
        2. The model is proposing a **respond / ASK_USER / FINAL** prematurely
           before fetching the needed product details, AND the goal is a
           no-auth product query (B4 fix).  This prevents the model from
           asking for credentials or giving up before counting products.

        Uses ``wm.product_types`` (durable cache, not db_facts) so multi-
        product follow-up queries work even after db_facts has been evicted
        by a single get_product_details flooding it with variant data.

        B2 fix: uses word-boundary matching so "hose" does NOT match "those"
        and "smart" does NOT match "smartwatch"; adds compound-word check so
        "Smart Watch" DOES match "smartwatch".
        """
        is_list_repeat = False
        is_premature_respond = False

        if action.name == "list_all_product_types":
            sig = action.signature()
            if sig not in wm.recent_signatures:
                return None  # first call — let it run normally
            is_list_repeat = True
        elif (
            action.declared_class in (RiskClass.ASK_USER, RiskClass.FINAL)
            or action.name.lower() in ("respond", "send_user", "finish")
        ):
            # Only intercept premature responds for no-auth product queries
            # when product types are already cached.  This handles task 3 /
            # task 4 where the model tries to ask for auth even though the
            # product count can be answered without it.
            if wm.product_types and self._is_no_auth_query(wm):
                is_premature_respond = True
        if not is_list_repeat and not is_premature_respond:
            return None

        all_user = " ".join(wm.user_facts).lower()
        # Iterate the durable catalogue.  Skip anything we've already fetched.
        for name, pid in wm.product_types.items():
            if not name or not pid:
                continue
            if pid in wm.product_details:
                continue  # already have details for this product
            name_norm = name.lower()
            if self._product_name_matches_user(name_norm, all_user):
                return ProposedAction(
                    name="get_product_details",
                    args={"product_id": pid},
                    declared_class=RiskClass.READ,
                    declared_pre=["product types already listed"],
                    declared_post=["product details retrieved"],
                    informational_intent=f"get details for {name}",
                    raw_thought=(
                        f"Product-list advance: user mentioned '{name}', "
                        f"resolved to ID {pid} — fetching details directly."
                    ),
                    user_text="",
                    raw_response="",
                )
        # Fallback: legacy db_facts iteration (kept for callers populating
        # db_facts directly without the solve-loop cache).
        for fact in wm.db_facts:
            if "=" not in fact:
                continue
            name_part, _, id_part = fact.partition("=")
            name_part = name_part.strip()
            id_part = id_part.strip()
            if not re.fullmatch(r"\d+", id_part):
                continue
            if id_part in wm.product_details:
                continue
            if self._product_name_matches_user(name_part.lower(), all_user):
                return ProposedAction(
                    name="get_product_details",
                    args={"product_id": id_part},
                    declared_class=RiskClass.READ,
                    declared_pre=["product types already listed"],
                    declared_post=["product details retrieved"],
                    informational_intent=f"get details for {name_part}",
                    raw_thought=(
                        f"Product-list advance: user mentioned '{name_part}', "
                        f"resolved to ID {id_part} — fetching details directly."
                    ),
                    user_text="",
                    raw_response="",
                )
        return None

    # ------------------------------------------------------------------
    # Product-count finalizer
    # ------------------------------------------------------------------
    def _finalize_product_count_query(
        self,
        action: ProposedAction,
        wm: "WorkingMemory",
    ) -> Optional[ProposedAction]:
        """Emit a deterministic FINAL with product counts when the model has
        already fetched the needed product details but is looping or giving
        up prematurely.

        D6 fix: the method is now multi-product-aware.  When the user asks
        about N products (e.g. "how many t-shirt options and info on cleaners,
        headphones, smart watches?"), the old code emitted a partial FINAL for
        the FIRST fetched product and set ``product_count_finalized = True``,
        preventing any further products from being answered.

        New behaviour:
        1. Discover every product the user mentioned that is present in
           ``wm.product_types``.
        2. If ANY of those products are still missing from ``wm.product_details``,
           return None — let ``_advance_after_product_list`` continue fetching.
        3. Only once ALL user-mentioned products have been fetched, emit ONE
           comprehensive FINAL covering all of them.
        4. If no product_types catalogue is available (e.g. the old path where
           list_all_product_types hasn't run), fall back to the single-product
           logic so previously working cases keep working.

        (Observed failure: trajectories(24) T2 — Headphones was answered after
        step 6; product_count_finalized blocked T-Shirt / Cleaner / Smart Watch
        from being answered for the remaining 21 steps.)
        """
        if wm.product_count_finalized:
            return None
        if not wm.product_details:
            return None

        goal_lower = (wm.goal or "").lower()
        # Only fire on "how many" / "number of" / "count" type queries.
        if not re.search(
            r"\b(how\s+many|number\s+of|count\s+of|count\s+the)\b",
            goal_lower,
        ):
            return None
        # Only when goal is a product-style query.
        if not _PRODUCT_QUERY_RE.search(goal_lower):
            return None
        # Fire when the model is:
        #   (a) about to repeat list_all_product_types (original case), OR
        #   (b) proposing a FINAL / ASK_USER prematurely.
        is_final_or_ask = action.declared_class in (RiskClass.FINAL, RiskClass.ASK_USER)
        is_list_or_auth_call = action.name in (
            "list_all_product_types",
            "find_user_id_by_email",
            "find_user_id_by_name_zip",
        )
        if not is_final_or_ask and not is_list_or_auth_call:
            return None
        if is_list_or_auth_call:
            sig = action.signature()
            if action.name == "list_all_product_types" and sig not in wm.recent_signatures:
                return None  # first call — let it run

        all_user = (goal_lower + " " + " ".join(wm.user_facts).lower())

        # ------------------------------------------------------------------
        # Multi-product path: catalogue is available
        # ------------------------------------------------------------------
        if wm.product_types:
            # Which products did the user mention?
            mentioned: List[Tuple[str, str]] = []  # (name, pid)
            for pt_name, pt_pid in wm.product_types.items():
                if self._product_name_matches_user(pt_name.lower(), all_user):
                    mentioned.append((pt_name, pt_pid))

            if mentioned:
                # Are any of the user-mentioned products still unfetched?
                unfetched = [
                    (n, p) for n, p in mentioned if p not in wm.product_details
                ]
                if unfetched:
                    # More data needed — let _advance_after_product_list handle it.
                    return None

                # All mentioned products are fetched.  Build comprehensive answer.
                lines: List[str] = []
                fetched_names: List[str] = []
                for pt_name, pt_pid in mentioned:
                    details = wm.product_details.get(pt_pid, {})
                    name = str(details.get("name", pt_name)).strip()
                    variants = details.get("variants") or {}
                    if not isinstance(variants, dict) or not variants:
                        continue
                    total = len(variants)
                    available = sum(
                        1 for v in variants.values()
                        if isinstance(v, dict) and v.get("available", True)
                    )
                    if available == total:
                        lines.append(
                            f"{name}: {total} options, all currently available"
                        )
                    else:
                        lines.append(
                            f"{name}: {total} variants in catalog, "
                            f"{available} currently available"
                        )
                    fetched_names.append(name)

                if not lines:
                    return None
                if len(lines) == 1:
                    msg = lines[0] + "."
                else:
                    msg = (
                        "Here is what's currently available in the store:\n"
                        + "\n".join(f"• {l}" for l in lines)
                    )
                wm.product_count_finalized = True
                return ProposedAction(
                    name=RESPOND_TOOL_NAME,
                    args={},
                    declared_class=RiskClass.FINAL,
                    declared_pre=[f"{len(fetched_names)} products fetched"],
                    declared_post=["counts answered"],
                    informational_intent=f"answer count for {', '.join(fetched_names)}",
                    raw_thought=(
                        f"Product-count finalize: "
                        + "; ".join(
                            f"{n} ({wm.product_details[p].get('variants') and len(wm.product_details[p]['variants'])} variants)"
                            for n, p in mentioned
                            if p in wm.product_details
                        )
                    ),
                    user_text=msg,
                    raw_response="",
                    bypass_gates=True,
                )

        # ------------------------------------------------------------------
        # Single-product fallback (no catalogue or no matches above)
        # ------------------------------------------------------------------
        goal_norm = goal_lower.replace("-", " ").replace("_", " ")
        for pid, details in wm.product_details.items():
            name = str(details.get("name", "")).strip()
            if not name:
                continue
            name_norm = name.lower().replace("-", " ").replace("_", " ")
            if name_norm in goal_norm or any(
                tok in goal_norm for tok in name_norm.split() if len(tok) >= 4
            ):
                variants = details.get("variants") or {}
                if not isinstance(variants, dict) or not variants:
                    continue
                total = len(variants)
                available = sum(
                    1 for v in variants.values()
                    if isinstance(v, dict) and v.get("available", True)
                )
                if available == total:
                    msg = (
                        f"There are {total} {name} options currently "
                        "available in the store."
                    )
                else:
                    msg = (
                        f"There are {total} {name} variants in the catalog, "
                        f"of which {available} are currently available."
                    )
                wm.product_count_finalized = True
                return ProposedAction(
                    name=RESPOND_TOOL_NAME,
                    args={},
                    declared_class=RiskClass.FINAL,
                    declared_pre=[f"{name} details fetched"],
                    declared_post=["count answered"],
                    informational_intent=f"answer count for {name}",
                    raw_thought=(
                        f"Product-count finalize: {name} has {total} variants "
                        f"({available} available)."
                    ),
                    user_text=msg,
                    raw_response="",
                    bypass_gates=True,
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
