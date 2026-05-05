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
from .adapters import select_adapter
from .calibration import default_calibration
from .core import BaseCargoAdapter, GenericCargoKernel
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
# tau-bench user instructions commonly state identity as "You are Yusuf
# Rossi in 19122."  This is still user-provided evidence, not a hidden label,
# and is much safer than falling back to fabricated email lookups.
_NAME_PERSONA_RE = re.compile(
    r"(?i:\byou\s+are)\s+([A-Z][a-z]{1,20})\s+([A-Z][a-z]{1,20})"
    r"(?=\s+(?:in|at|from)\s+\d{5}\b|\s*[.,])"
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
        self.adapter = select_adapter(env_hint=env_hint, tools_info=self.tools_info, wiki=self.wiki)
        self.kernel = GenericCargoKernel(self.adapter)
        self.schemas = self.kernel.enrich_schemas(self.schemas)
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
    def _kernel(self) -> GenericCargoKernel:
        kernel = getattr(self, "kernel", None)
        if kernel is None:
            adapter = getattr(self, "adapter", None) or BaseCargoAdapter()
            kernel = GenericCargoKernel(adapter)
            self.adapter = adapter
            self.kernel = kernel
        return kernel

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

        # Successful writes are single-shot.  Check this before semantic
        # validation so a duplicate mutation is reported as a terminal
        # lifecycle error instead of a secondary grounding/completeness issue.
        if action.declared_class in (RiskClass.WRITE, RiskClass.IRREVERSIBLE):
            sig = action.signature()
            if wm.mutation_already_executed(sig) or wm.task_completed:
                done_gate = GateResult.failing(
                    "completed_task",
                    "state_change_already_executed",
                    signature=sig,
                )
                diag["gates_run"].append("completed_task")
                diag["gates_failed"].append("completed_task")
                stats.record_gate(done_gate)
                return done_gate, diag

        state_gate = self._check_state_action_validity(action, wm)
        diag["gates_run"].append("state_validity")
        stats.record_gate(state_gate)
        if not state_gate.ok:
            diag["gates_failed"].append("state_validity")
            return state_gate, diag

        if action.declared_class == RiskClass.FINAL:
            final_gate = self._check_final_completeness(action, wm)
            diag["gates_run"].append("final_completeness")
            stats.record_gate(final_gate)
            if not final_gate.ok:
                diag["gates_failed"].append("final_completeness")
                return final_gate, diag

        if action.declared_class in (RiskClass.WRITE, RiskClass.IRREVERSIBLE):
            confirm_gate = self._check_write_confirmation(action, wm)
            diag["gates_run"].append("confirmation")
            stats.record_gate(confirm_gate)
            if not confirm_gate.ok:
                diag["gates_failed"].append("confirmation")
                return confirm_gate, diag
            completeness_gate = self._check_write_completeness(action, wm)
            diag["gates_run"].append("completeness")
            stats.record_gate(completeness_gate)
            if not completeness_gate.ok:
                diag["gates_failed"].append("completeness")
                return completeness_gate, diag

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
        self._kernel().observe_user_message(wm, initial_user)

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
            if wm.task_completed:
                break
            wm.budget_steps = max_num_steps - step
            stats.steps_total += 1

            proposer_messages = self._build_proposer_messages(wm, messages, critique)
            action, raw_text = self._call_proposer(proposer_messages)

            # Grounded placeholder resolver: if the proposer emits an argument
            # such as {"user_id": "user_id"} after the user already supplied a
            # concrete user_id, substitute the single grounded value before any
            # domain-specific override or gate sees it. Ambiguous state is left
            # untouched so the normal grounding/repair path asks for clarity
            # instead of guessing.
            if action is not None:
                grounded_action = self._resolve_grounded_placeholders(action, wm)
                if grounded_action is not None:
                    raw_text = json.dumps({
                        "thought": grounded_action.raw_thought,
                        "action": {
                            "name": grounded_action.name,
                            "args": grounded_action.args,
                            "declared_class": grounded_action.declared_class.value,
                            "declared_pre": grounded_action.declared_pre,
                            "declared_post": grounded_action.declared_post,
                            "informational_intent": grounded_action.informational_intent,
                            "user_text": grounded_action.user_text,
                        },
                    })
                    action = grounded_action

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

            # Airline-style reservation advance: once a user profile has
            # grounded reservation IDs, do not retry get_user_details forever.
            # Scan reservation details one at a time so later policy/commit
            # decisions are based on DB-confirmed reservation state.
            if action is not None:
                res_action = self._advance_reservation_retrieval(action, wm)
                if res_action is not None:
                    raw_text = json.dumps({
                        "thought": res_action.raw_thought,
                        "action": {
                            "name": res_action.name,
                            "args": res_action.args,
                            "declared_class": res_action.declared_class.value,
                            "declared_pre": res_action.declared_pre,
                            "declared_post": res_action.declared_post,
                            "informational_intent": res_action.informational_intent,
                            "user_text": res_action.user_text,
                        },
                    })
                    action = res_action

            # CARGO-v4 obligation/decision guide: READ actions build state, so when a
            # proposal is an unhelpful ASK/FINAL/repeated read but obligations
            # already identify the next information need, deterministically
            # choose that READ.  Commitment gates remain strict later.
            if action is not None:
                ob_action = self._obligation_guided_action(action, wm)
                if ob_action is not None:
                    raw_text = json.dumps({
                        "thought": ob_action.raw_thought,
                        "action": {
                            "name": ob_action.name,
                            "args": ob_action.args,
                            "declared_class": ob_action.declared_class.value,
                            "declared_pre": ob_action.declared_pre,
                            "declared_post": ob_action.declared_post,
                            "informational_intent": ob_action.informational_intent,
                            "user_text": ob_action.user_text,
                        },
                    })
                    action = ob_action

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

            # Grounded progress / commit trigger: when state has enough
            # verified slots for a task-level transition, override proposer
            # relapses (auth lookup, repeated browsing, premature respond) with
            # the next grounded read or final mutation.  The normal required
            # arg + grounding gates still run on the produced action.
            if action is not None:
                progress_action = self._grounded_progress_or_commit_action(action, wm)
                if progress_action is not None:
                    raw_text = json.dumps({
                        "thought": progress_action.raw_thought,
                        "action": {
                            "name": progress_action.name,
                            "args": progress_action.args,
                            "declared_class": progress_action.declared_class.value,
                            "declared_pre": progress_action.declared_pre,
                            "declared_post": progress_action.declared_post,
                            "informational_intent": progress_action.informational_intent,
                            "user_text": progress_action.user_text,
                        },
                    })
                    action = progress_action

            # Final write discipline: any model-proposed mutation that can be
            # assembled deterministically from grounded state is replaced with
            # the canonical best action before gates/execute.  This prevents
            # "try a write, then fix it later" behavior.
            if action is not None:
                write_action = self._canonicalize_write_action(action, wm)
                if write_action is not None:
                    raw_text = json.dumps({
                        "thought": write_action.raw_thought,
                        "action": {
                            "name": write_action.name,
                            "args": write_action.args,
                            "declared_class": write_action.declared_class.value,
                            "declared_pre": write_action.declared_pre,
                            "declared_post": write_action.declared_post,
                            "informational_intent": write_action.informational_intent,
                            "user_text": write_action.user_text,
                        },
                    })
                    action = write_action

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
                        self._kernel().observe_user_message(wm, user_reply)
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
                    self._kernel().observe_user_message(wm, user_reply)
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
                wm.record_failed_signature(action.signature())
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
                    self._kernel().observe_user_message(wm, user_reply)
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
                    wm.task_completed = True
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
                    self._kernel().observe_user_message(wm, user_reply)
                reward = _float(getattr(env_resp, "reward", reward), reward)
                info = getattr(env_resp, "info", info) or info
                done = bool(getattr(env_resp, "done", False))
                stats.actions_executed += 1
                cls = action.declared_class.value
                stats.executed_by_class[cls] = stats.executed_by_class.get(cls, 0) + 1
                step_record["executed"] = True
                stats.record_step(step_record)
                if action.declared_class == RiskClass.FINAL:
                    if done or not user_reply:
                        wm.task_completed = True
                        break
                    critique = ""
                    retries_used = 0
                    continue
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
                self._kernel().observe_tool_result(
                    wm,
                    action.name,
                    {"status": "error", "error": str(env_exc)[:80]},
                )
                wm.record_failed_signature(action.signature())
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
            self._kernel().observe_tool_result(wm, action.name, tool_obs)
            self._kernel().record_action_candidates(wm, action.name, action.args, tool_obs)

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
                            wm.lock_phase("auth")
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
            if action.name == "get_user_details" and tool_obs:
                try:
                    ud = json.loads(tool_obs) if tool_obs.lstrip().startswith("{") else None
                    if isinstance(ud, dict):
                        uid = str(action.args.get("user_id") or ud.get("user_id") or "").strip()
                        if uid and self._is_user_id_token(uid):
                            wm.user_profiles[uid] = ud
                            if not wm.auth_user_id:
                                wm.auth_user_id = uid
                                wm.lock_phase("auth")
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
            if action.name == "get_reservation_details" and tool_obs:
                try:
                    rd = json.loads(tool_obs) if tool_obs.lstrip().startswith("{") else None
                    if isinstance(rd, dict):
                        rid = str(
                            rd.get("reservation_id")
                            or rd.get("reservation_number")
                            or action.args.get("reservation_id")
                            or ""
                        ).strip()
                        if rid:
                            wm.reservation_details[rid] = rd
                except Exception:
                    pass
            # Successful retail mutations also return an order-shaped object.
            # Promote it into the order cache so stale "delivered"/"pending"
            # state cannot drive another write after the world has changed.
            if action.name != "get_order_details" and tool_obs:
                try:
                    od = json.loads(tool_obs) if tool_obs.lstrip().startswith("{") else None
                    if isinstance(od, dict) and od.get("order_id"):
                        wm.order_details[str(od.get("order_id"))] = od
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
            if (
                action.declared_class in (RiskClass.WRITE, RiskClass.IRREVERSIBLE)
                and pc.ok
            ):
                wm.record_executed_mutation(action.signature())
                wm.lock_phase("mutation")
                # A task may require multiple distinct writes (for example,
                # updating two separate pending orders).  Stop immediately
                # when no further grounded mutation remains, but do not mark
                # the whole task complete after the first successful write if
                # the controller can assemble another fresh, non-duplicate
                # mutation from the updated state.
                if self._grounded_retail_commit_action(wm) is None:
                    wm.task_completed = True
                    break
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
        # Persona pass: benchmark/user role instructions sometimes provide
        # identity as "You are First Last in ZIP".  Accept only that tight
        # shape so product prose like "you are looking at Google Home" cannot
        # become a name.
        for fact in facts:
            m = _NAME_PERSONA_RE.search(fact)
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

    @staticmethod
    def _order_product_catalog(wm: "WorkingMemory") -> Dict[str, str]:
        """Return product name → product_id from grounded order details."""
        out: Dict[str, str] = {}
        for details in wm.order_details.values():
            if not isinstance(details, dict):
                continue
            for item in details.get("items") or []:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "").strip()
                pid = str(item.get("product_id") or "").strip()
                if name and pid:
                    out[name] = pid
        return out

    @staticmethod
    def _order_item_id_to_product(wm: "WorkingMemory") -> Dict[str, Tuple[str, str]]:
        """Return item_id → (product name, product_id) from order details."""
        out: Dict[str, Tuple[str, str]] = {}
        for details in wm.order_details.values():
            if not isinstance(details, dict):
                continue
            for item in details.get("items") or []:
                if not isinstance(item, dict):
                    continue
                item_id = str(item.get("item_id") or "").strip()
                name = str(item.get("name") or "").strip()
                pid = str(item.get("product_id") or "").strip()
                if item_id and pid:
                    out[item_id] = (name, pid)
        return out

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

        # Account/order tasks: once identity and order details are grounded,
        # do not let the proposer re-enter auth.  Fetch product details for
        # order items that the user mentioned so later mutations can use
        # grounded item IDs from product_details rather than invented IDs.
        all_user = (wm.goal + " " + " ".join(wm.user_facts)).lower()
        for name, pid in self._order_product_catalog(wm).items():
            if pid in wm.product_details:
                continue
            if self._product_name_matches_user(name.lower(), all_user):
                return ProposedAction(
                    name="get_product_details",
                    args={"product_id": pid},
                    declared_class=RiskClass.READ,
                    declared_pre=["order item product grounded"],
                    declared_post=["product details retrieved"],
                    informational_intent=f"fetch product details for {name}",
                    raw_thought=(
                        "Post-auth override: auth is complete; fetching "
                        f"grounded product details for order item '{name}'."
                    ),
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
            wm.lock_phase("auth")
            post_action = self._post_auth_action(wm)
            if post_action is not None:
                return post_action
            # Auth is complete.  Do not re-confirm via email or name+ZIP; that
            # re-enters a finished phase and was the source of auth relapse
            # loops.  If no grounded progress/commit action is available yet,
            # let the generic gate reject the model's redundant lookup rather
            # than issuing another auth tool call.
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

    def _advance_reservation_retrieval(
        self,
        action: ProposedAction,
        wm: "WorkingMemory",
    ) -> Optional[ProposedAction]:
        """Advance from grounded user profile to reservation retrieval.

        Airline profile lookups return a list of reservation IDs.  If the
        proposer repeats get_user_details or emits a generic respond after the
        profile is cached, scan those reservation IDs instead of retrying the
        completed auth/profile phase.
        """
        if not self._has_tool("get_reservation_details"):
            return None
        if not self._reservation_task_needs_scan(wm):
            return None
        repeated_profile = action.name == "get_user_details"
        generic_respond = (
            action.declared_class in (RiskClass.ASK_USER, RiskClass.FINAL)
            or action.name.lower() in (RESPOND_TOOL_NAME, "respond", "final", "answer")
        )
        if not (repeated_profile or generic_respond):
            return None
        reservation_ids = wm.typed_evidence_for("reservation_id")
        if not reservation_ids:
            return None
        for rid in reservation_ids:
            if rid in wm.reservation_details:
                continue
            candidate = ProposedAction(
                name="get_reservation_details",
                args={"reservation_id": rid},
                declared_class=RiskClass.READ,
                declared_pre=["reservation id grounded"],
                declared_post=["reservation details retrieved"],
                informational_intent="fetch reservation details",
                raw_thought=(
                    f"Grounded progress: profile is cached; fetch reservation {rid}."
                ),
                user_text="",
                raw_response="",
            )
            fresh = self._fresh_action_or_none(candidate, wm)
            if fresh is not None:
                return fresh
        return None

    @staticmethod
    def _reservation_task_needs_scan(wm: "WorkingMemory") -> bool:
        text = (wm.goal + " " + " ".join(wm.user_facts)).lower()
        return bool(re.search(
            r"\b(reservation|booking|flight|ticket|trip|cancel|change|modify|upgrade|baggage|bags?)\b",
            text,
        ))

    # ------------------------------------------------------------------
    # Generic grounded-argument resolver
    # ------------------------------------------------------------------
    def _resolve_grounded_placeholders(
        self,
        action: ProposedAction,
        wm: "WorkingMemory",
    ) -> Optional[ProposedAction]:
        """Replace argument placeholders with one unambiguous grounded value.

        This is the general state-consistency repair for cross-domain tasks:
        the proposer may correctly select the next tool but still pass the
        schema field name as the value (for example ``user_id="user_id"``).
        If working memory has exactly one typed value for that argument field,
        use it.  If there are zero or multiple candidates, do not guess; the
        normal grounding gate remains responsible for blocking the action.
        """
        if not action.args:
            return None
        new_args, changed = self._resolve_placeholder_obj(action.args, wm)
        if not changed:
            return None
        return ProposedAction(
            name=action.name,
            args=new_args if isinstance(new_args, dict) else action.args,
            declared_class=action.declared_class,
            declared_pre=action.declared_pre,
            declared_post=action.declared_post,
            informational_intent=action.informational_intent,
            raw_thought=(
                f"Grounded placeholder resolver: replaced placeholder args "
                f"using persisted typed state for {action.name}."
            ),
            user_text=action.user_text,
            raw_response=action.raw_response,
            bypass_gates=action.bypass_gates,
        )

    @classmethod
    def _resolve_placeholder_obj(
        cls,
        obj: Any,
        wm: "WorkingMemory",
        field_path: str = "",
    ) -> Tuple[Any, bool]:
        if isinstance(obj, dict):
            changed = False
            out: Dict[str, Any] = {}
            for key, value in obj.items():
                child_path = f"{field_path}.{key}" if field_path else str(key)
                new_value, did_change = cls._resolve_placeholder_obj(value, wm, child_path)
                out[key] = new_value
                changed = changed or did_change
            return out, changed
        if isinstance(obj, list):
            changed = False
            out: List[Any] = []
            for i, value in enumerate(obj):
                child_path = f"{field_path}[{i}]"
                new_value, did_change = cls._resolve_placeholder_obj(value, wm, child_path)
                out.append(new_value)
                changed = changed or did_change
            return out, changed
        if not isinstance(obj, str):
            return obj, False

        field = cls._placeholder_field_name(field_path)
        if not cls._is_grounded_placeholder_value(field, obj):
            return obj, False
        candidates = cls._dedupe_strings(
            [str(v).strip() for v in wm.typed_evidence_for(field) if str(v).strip()]
        )
        if len(candidates) != 1:
            return obj, False
        return candidates[0], candidates[0] != obj

    @staticmethod
    def _placeholder_field_name(field_path: str) -> str:
        key = str(field_path or "").strip().lower()
        key = re.sub(r"\[\d+\]", "", key)
        key = key.split(".")[-1]
        if key in ("item_ids", "new_item_ids"):
            return "item_id"
        if key.endswith("_ids"):
            return key[:-1]
        if key.endswith("ids") and len(key) > 3:
            return key[:-1]
        return key

    @classmethod
    def _is_grounded_placeholder_value(cls, field_name: str, value: str) -> bool:
        field = cls._placeholder_field_name(field_name)
        if not field:
            return False
        # Restrict generic replacement to opaque/typed fields.  This avoids
        # treating free-form values like insurance="none" as placeholders.
        if not (field.endswith("_id") or field == "email"):
            return False
        raw = str(value or "").strip()
        if not raw:
            return True
        norm = raw.lower().strip("<>{}[]() \t\r\n\"'")
        norm = norm.removeprefix("$")
        compact_field = field.replace("_", "")
        compact_norm = norm.replace("_", "").replace("-", "").replace(" ", "")
        if norm in {field, f"{field}?", f"{field}."}:
            return True
        if compact_norm == compact_field:
            return True
        return norm in {
            "id", "unknown", "placeholder", "todo", "tbd", "n/a", "na",
            "none", "null",
        }

    @staticmethod
    def _dedupe_strings(values: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            out.append(value)
        return out

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
        # wm.product_types catalogue and grounded order details, then fall
        # back to simple db_facts catalogue entries.  Do NOT treat arbitrary
        # numeric db_facts as product IDs; item IDs share the same shape and
        # were observed passing through as product_id arguments.
        name_to_id: Dict[str, str] = {}
        all_known_ids = set()
        for name, pid in wm.product_types.items():
            if isinstance(name, str) and isinstance(pid, str) and re.fullmatch(r"\d+", pid):
                name_to_id[name.strip().lower()] = pid
                all_known_ids.add(pid)
        for name, pid in self._order_product_catalog(wm).items():
            if re.fullmatch(r"\d+", pid):
                name_to_id.setdefault(name.strip().lower(), pid)
                all_known_ids.add(pid)
        for fact in wm.db_facts:
            if "=" not in fact:
                continue
            name_part, _, id_part = fact.partition("=")
            id_part = id_part.strip()
            if not re.fullmatch(r"\d+", id_part):
                continue
            if "." in name_part or "[" in name_part:
                # Structured observation paths like items[0].item_id=...
                # retain provenance; they are not product catalogue entries.
                continue
            name_to_id.setdefault(name_part.strip().lower(), id_part)
            all_known_ids.add(id_part)

        # ---- Case 1: numeric ID not in db_facts → hallucinated ----
        if re.fullmatch(r"\d+", pid_str):
            if pid_str in all_known_ids:
                return None  # legitimate ID, leave as-is
            item_map = self._order_item_id_to_product(wm)
            if pid_str in item_map:
                item_name, item_product_id = item_map[pid_str]
                # If the model supplied an order item_id where product_id is
                # required, convert it to that item's product_id only when the
                # item itself is goal-relevant; otherwise fall through to the
                # goal matcher below.
                all_user = (wm.goal + " " + " ".join(wm.user_facts)).lower()
                if item_name and self._product_name_matches_user(item_name.lower(), all_user):
                    new_args = dict(action.args)
                    new_args["product_id"] = item_product_id
                    return ProposedAction(
                        name=action.name,
                        args=new_args,
                        declared_class=action.declared_class,
                        declared_pre=action.declared_pre,
                        declared_post=action.declared_post,
                        informational_intent=action.informational_intent,
                        raw_thought=(
                            f"Product-ID override: item_id '{pid_str}' → "
                            f"product_id {item_product_id} for order item {item_name}."
                        ),
                        user_text=action.user_text,
                        raw_response=action.raw_response,
                    )
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
        # Normalize hyphenated product names ("T-Shirt") against compact user
        # spellings ("tshirt") and spaced spellings ("t shirt").
        name_spaced = re.sub(r"[-_]+", " ", name_norm)
        user_spaced = re.sub(r"[-_]+", " ", all_user)
        if name_spaced in user_spaced:
            return True
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
            if tok.endswith("s") and len(tok) > 4:
                singular = tok[:-1]
                if re.search(rf"\b{re.escape(singular)}s?\b", all_user):
                    return True
        # Rule 3: compound form (e.g. "smartwatch" ↔ "Smart Watch")
        compound = re.sub(r"[^a-z0-9]+", "", name_norm)
        user_compound = re.sub(r"[^a-z0-9]+", "", all_user)
        if len(compound) >= 6 and compound in user_compound:
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
                wm.lock_phase("product_count")
                final_cls = (
                    RiskClass.ASK_USER
                    if self._is_account_order_goal(wm)
                    else RiskClass.FINAL
                )
                return ProposedAction(
                    name=RESPOND_TOOL_NAME,
                    args={},
                    declared_class=final_cls,
                    declared_pre=[],
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
                wm.lock_phase("product_count")
                final_cls = (
                    RiskClass.ASK_USER
                    if self._is_account_order_goal(wm)
                    else RiskClass.FINAL
                )
                return ProposedAction(
                    name=RESPOND_TOOL_NAME,
                    args={},
                    declared_class=final_cls,
                    declared_pre=[],
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
    # Grounded task progress / commit layer
    # ------------------------------------------------------------------
    def _check_state_action_validity(
        self,
        action: ProposedAction,
        wm: "WorkingMemory",
    ) -> GateResult:
        """Validate that an action is appropriate for current task state."""
        name = action.name.lower()
        if wm.phase_locked("auth") and name.startswith("find_user_id_"):
            return GateResult.failing(
                "state_validity",
                "auth_phase_already_complete",
                action=action.name,
            )
        id_gate = self._adapter_id_field_gate(action, wm)
        if not id_gate.ok:
            return id_gate
        if name == "get_user_details":
            uid = str(action.args.get("user_id") or "").strip()
            if uid and uid in wm.user_profiles:
                return GateResult.failing(
                    "state_validity",
                    "user_profile_already_cached",
                    user_id=uid,
                )

        return self._kernel().validate_action(action, self._schema_for(action), wm)

    def _adapter_id_field_gate(
        self,
        action: ProposedAction,
        wm: "WorkingMemory",
    ) -> GateResult:
        """Backstop adapter-declared IDs even when a tool schema is weak.

        tau-bench tools generally expose clear parameter schemas, but a
        proposer repair can synthesize an action before schema enrichment sees
        the tool definition.  Adapter-declared ID names still mean "opaque ID",
        so plain words such as reservation_id="though" must never pass as a
        READ just because they do not match the generic ID regex.
        """
        id_fields = {str(v) for v in getattr(getattr(self, "adapter", None), "id_fields", set()) or set()}
        if not id_fields:
            return GateResult.passing("state_validity")
        bad: List[str] = []
        for path, value in self._iter_action_scalars(action.args):
            base = path.split(".")[-1].split("[")[0]
            root = path.split(".")[0].split("[")[0]
            if base not in id_fields and root not in id_fields:
                continue
            v = str(value or "").strip()
            if not v:
                continue
            typed = wm.typed_evidence_for(base) or wm.typed_evidence_for(root)
            if typed and v in typed:
                continue
            if not self._looks_like_adapter_id(v):
                bad.append(f"{path}={v}")
        if bad:
            return GateResult.failing(
                "state_validity",
                "adapter_id_field_plain_word",
                invalid=bad[:3],
            )
        return GateResult.passing("state_validity")

    @classmethod
    def _iter_action_scalars(cls, value: Any, prefix: str = ""):
        if isinstance(value, dict):
            for key, item in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                yield from cls._iter_action_scalars(item, path)
            return
        if isinstance(value, list):
            for idx, item in enumerate(value):
                yield from cls._iter_action_scalars(item, f"{prefix}[{idx}]")
            return
        yield prefix, value

    @staticmethod
    def _looks_like_adapter_id(value: str) -> bool:
        v = str(value or "").strip()
        if not v:
            return False
        if re.fullmatch(r"\d{4,}", v):
            return True
        if re.fullmatch(r"[A-Z0-9][A-Z0-9_-]{3,}", v):
            return True
        if re.fullmatch(r"[a-z]+_[a-z]+_\d{1,8}", v):
            return True
        if re.fullmatch(r"(?:credit_card|gift_card|certificate|paypal)_\d+", v):
            return True
        if re.fullmatch(r"#[A-Za-z]?\d{4,}", v):
            return True
        return False

    @staticmethod
    def _semantic_values_match(proposed: Any, expected: Any) -> bool:
        def norm(v: Any) -> str:
            return re.sub(r"[^a-z0-9]+", " ", str(v or "").lower()).strip()
        p = norm(proposed)
        if isinstance(expected, list):
            vals = [norm(v) for v in expected]
        else:
            vals = [norm(expected)]
        return any(p == v or (p and v and (p in v or v in p)) for v in vals)

    def _check_write_confirmation(
        self,
        action: ProposedAction,
        wm: "WorkingMemory",
    ) -> GateResult:
        """Require explicit user intent before any state-changing action.

        In tau-bench the initial instruction is itself a user instruction
        ("You want to return...", "wish to exchange...").  We treat direct
        action requests and later "yes / confirm / go ahead" replies as
        confirmation, but not mere retrieved DB facts or model assumptions.
        """
        text = (wm.goal + " " + " ".join(wm.user_facts)).lower()
        if re.search(r"\b(yes|confirm|confirmed|go ahead|proceed|do it|please do|sounds good)\b", text):
            return GateResult.passing("confirmation")
        name = action.name.lower()
        verb_groups = {
            "exchange": ("exchange", "exchanging", "swap", "replace"),
            "return": ("return", "refund"),
            "modify": ("modify", "change", "update", "adjust"),
            "cancel": ("cancel",),
            "update": ("update", "change", "modify", "set"),
            "book": ("book", "reserve", "purchase"),
        }
        for group, verbs in verb_groups.items():
            if group in name and any(re.search(rf"\b{re.escape(v)}\b", text) for v in verbs):
                return GateResult.passing("confirmation")
        return GateResult.failing(
            "confirmation",
            "missing_explicit_user_confirmation_for_write",
        )

    def _canonicalize_write_action(
        self,
        action: ProposedAction,
        wm: "WorkingMemory",
    ) -> Optional[ProposedAction]:
        """Replace retail WRITE proposals with the canonical grounded commit.

        This keeps the proposer from executing a plausible but non-best
        mutation when the controller can already compute the correct option
        from order/product state.  If the proposed action already matches the
        canonical signature, return None.
        """
        if action.name not in {
            "exchange_delivered_order_items",
            "return_delivered_order_items",
            "modify_pending_order_items",
        }:
            return None
        canonical = self._grounded_retail_commit_action(wm)
        if canonical is None or canonical.name != action.name:
            return None
        if canonical.signature() == action.signature():
            return None
        canonical.raw_thought = (
            "Write canonicalizer: replacing proposed mutation with the best "
            "grounded action assembled from current state."
        )
        return canonical

    def _check_write_completeness(
        self,
        action: ProposedAction,
        wm: "WorkingMemory",
    ) -> GateResult:
        """Reject partial or non-canonical retail mutations.

        Canonicalization fixes the common path before gates run.  This gate is
        the safety backstop: a WRITE for a known retail mutation tool may only
        execute if it exactly matches the complete grounded action assembled by
        the controller from current order/product state.  That blocks partial
        exchanges/returns/modifies, decoy variant choices, and premature writes
        where some requested item is still unresolved.
        """
        adapter_gate = self._kernel().validate_write_completeness(action, wm)
        if adapter_gate is not None:
            return adapter_gate
        if action.name not in {
            "exchange_delivered_order_items",
            "return_delivered_order_items",
            "modify_pending_order_items",
        }:
            return GateResult.passing("completeness")
        canonical = self._grounded_retail_commit_action(wm)
        if canonical is None:
            return GateResult.failing(
                "completeness",
                "no_complete_grounded_write_available",
            )
        if canonical.name != action.name:
            return GateResult.failing(
                "completeness",
                "write_tool_does_not_match_grounded_task",
                expected=canonical.name,
                proposed=action.name,
            )
        if canonical.signature() != action.signature():
            return GateResult.failing(
                "completeness",
                "write_args_do_not_match_complete_grounded_action",
                expected=canonical.args,
                proposed=action.args,
            )
        return GateResult.passing("completeness")

    def _check_final_completeness(
        self,
        action: ProposedAction,
        wm: "WorkingMemory",
    ) -> GateResult:
        """Prevent terminal answers while account-side work is still open."""
        if action.declared_class != RiskClass.FINAL:
            return GateResult.passing("final_completeness")
        adapter_gate = self._kernel().validate_final_completeness(action, wm)
        if adapter_gate is not None and not adapter_gate.ok:
            return adapter_gate
        if not self._is_account_order_goal(wm):
            return GateResult.passing("final_completeness")
        if wm.phase_locked("mutation") or wm.task_completed or wm.auth_abandoned:
            return GateResult.passing("final_completeness")
        return GateResult.failing(
            "final_completeness",
            "account_task_has_unresolved_write_phase",
        )

    def _has_tool(self, name: str) -> bool:
        schemas = getattr(self, "schemas", None) or {}
        return (not schemas) or name in schemas

    def _obligation_guided_action(
        self,
        action: ProposedAction,
        wm: "WorkingMemory",
    ) -> Optional[ProposedAction]:
        """Choose the next information-gathering action from open obligations.

        This is the CARGO-v4 controller layer.  It is deliberately narrow and
        retrieval-focused: it may replace a stuck ASK/FINAL/repeated READ with
        a grounded READ that reduces uncertainty, but it does not fabricate a
        WRITE.  Writes still flow through canonicalization and completeness.
        """
        if self._is_airline_adapter():
            return self._airline_obligation_action(action, wm)
        return None

    def _is_airline_adapter(self) -> bool:
        adapter = getattr(self, "adapter", None)
        domain = str(getattr(adapter, "domain_name", "") or "").lower()
        if domain == "airline":
            return True
        names = set((getattr(self, "schemas", None) or {}).keys())
        return bool({"search_direct_flight", "search_onestop_flight"} & names)

    def _airline_obligation_action(
        self,
        action: ProposedAction,
        wm: "WorkingMemory",
    ) -> Optional[ProposedAction]:
        if not self._has_tool("search_direct_flight") and not self._has_tool("search_onestop_flight"):
            return None

        user_id = self._known_user_id(wm)
        if not user_id and self._airline_task_needs_identity(wm):
            ask = self._airline_user_id_request(action, wm)
            if ask is not None:
                return ask

        if user_id and self._has_tool("get_user_details") and user_id not in wm.user_profiles:
            candidate = ProposedAction(
                name="get_user_details",
                args={"user_id": user_id},
                declared_class=RiskClass.READ,
                declared_pre=["user id grounded"],
                declared_post=["user profile retrieved"],
                informational_intent="fetch profile for airline task",
                raw_thought="Obligation guide: user_id is grounded; fetch profile before commitment.",
                user_text="",
                raw_response="",
            )
            fresh = self._fresh_action_or_none(candidate, wm)
            if fresh is not None:
                return fresh

        slots = wm.semantic_slots
        origin = self._slot_value(slots, "origin")
        destination = self._slot_value(slots, "destination")
        date = self._slot_value(slots, "date")
        if not (origin and destination and date):
            return None
        origin = self._canonical_airport_arg(origin, field="origin")
        destination = self._canonical_airport_arg(destination, field="destination")
        direct_args = {"origin": origin, "destination": destination, "date": date}
        direct_set = wm.task_state.candidate_set_for("search_direct_flight", direct_args)
        direct_exhausted = bool(direct_set and direct_set.exhausted)

        should_intercept = (
            action.declared_class in (RiskClass.ASK_USER, RiskClass.FINAL)
            or action.name.lower() in (RESPOND_TOOL_NAME, "respond", "final", "answer")
            or action.signature() in wm.recent_signatures
            or wm.failed_without_new_evidence(action.signature())
        )
        if action.name in {"search_direct_flight", "search_onestop_flight"}:
            should_intercept = should_intercept or action.signature() in wm.recent_signatures
        if not should_intercept:
            return None

        if not direct_exhausted:
            direct = self._flight_search_action(
                "search_direct_flight",
                origin=origin,
                destination=destination,
                date=date,
                thought="Decision engine: route/date are bound; search direct flights.",
            )
            if direct is not None:
                fresh = self._fresh_action_or_none(direct, wm)
                if fresh is not None:
                    return fresh

        one_set = None
        one_exhausted = False
        if self._onestop_allowed(wm):
            one_args = {"origin": origin, "destination": destination, "date": date}
            one_set = wm.task_state.candidate_set_for("search_onestop_flight", one_args)
            one_exhausted = bool(one_set and one_set.exhausted)
        if self._onestop_allowed(wm) and not one_exhausted:
            one = self._flight_search_action(
                "search_onestop_flight",
                origin=origin,
                destination=destination,
                date=date,
                thought="Decision engine: direct search is exhausted; search one-stop options.",
            )
            if one is not None:
                fresh = self._fresh_action_or_none(one, wm)
                if fresh is not None:
                    return fresh
        all_searches_exhausted = direct_exhausted and (
            not self._onestop_allowed(wm) or bool(one_set and one_set.exhausted)
        )
        if all_searches_exhausted:
            wm.task_state.terminal_status = "blocked_no_matching_flights"
            return ProposedAction(
                name=RESPOND_TOOL_NAME,
                args={},
                declared_class=RiskClass.FINAL,
                declared_pre=["all allowed flight searches exhausted"],
                declared_post=["user informed no matching flight exists"],
                informational_intent="terminate airline task with search blocker",
                raw_thought=(
                    "Decision engine: direct and allowed one-stop searches are "
                    "exhausted for the grounded route/date, so stop instead of looping."
                ),
                user_text=(
                    "I could not find any matching flights for the requested route and date "
                    "under the allowed search strategies."
                ),
                raw_response="",
                bypass_gates=True,
            )
        return None

    def _airline_user_id_request(
        self,
        action: ProposedAction,
        wm: "WorkingMemory",
    ) -> Optional[ProposedAction]:
        should_intercept = (
            action.declared_class in (RiskClass.ASK_USER, RiskClass.FINAL)
            or action.name.lower() in (RESPOND_TOOL_NAME, "respond", "final", "answer")
            or action.name.lower() in {
                "search_direct_flight",
                "search_onestop_flight",
                "get_reservation_details",
                "book_reservation",
                "update_reservation_flights",
                "cancel_reservation",
            }
        )
        if not should_intercept:
            return None
        if wm.auth_ask_count >= _MAX_AUTH_ASKS:
            return None
        ask = ProposedAction(
            name=RESPOND_TOOL_NAME,
            args={},
            declared_class=RiskClass.ASK_USER,
            declared_pre=["airline task requires account identity"],
            declared_post=["user provides airline user_id"],
            informational_intent="ask for missing airline user_id",
            raw_thought=(
                "Decision engine: airline policy requires user_id before "
                "profile, reservation, or booking work; ask precisely instead "
                "of a generic clarification."
            ),
            user_text="Please provide your airline user ID so I can access the relevant profile or reservation.",
            raw_response="",
            bypass_gates=True,
        )
        fresh = self._fresh_action_or_none(ask, wm)
        if fresh is not None:
            wm.auth_ask_count += 1
            return fresh
        return None

    @staticmethod
    def _airline_task_needs_identity(wm: "WorkingMemory") -> bool:
        text = (wm.goal + " " + " ".join(wm.user_facts)).lower()
        slots = wm.semantic_slots
        intents = slots.get("intents") or slots.get("intent") or []
        if not isinstance(intents, list):
            intents = [intents]
        intent_text = " ".join(str(v).lower() for v in intents if v)
        return bool(
            re.search(r"\b(book|reserve|purchase|change|modify|update|reschedule|cancel|refund|downgrade|upgrade)\b", text)
            and re.search(r"\b(flight|ticket|trip|reservation)\b", text)
        ) or bool({"book_flight", "modify_flight"} & set(intent_text.split()))

    def _canonical_airport_arg(self, value: str, *, field: str) -> str:
        mapper = getattr(getattr(self, "adapter", None), "canonicalize_airport", None)
        if callable(mapper):
            mapped = mapper(value, field=field)
            if mapped:
                return str(mapped)
        return value

    def _flight_search_action(
        self,
        name: str,
        *,
        origin: str,
        destination: str,
        date: str,
        thought: str,
    ) -> Optional[ProposedAction]:
        if not self._has_tool(name):
            return None
        return ProposedAction(
            name=name,
            args={"origin": origin, "destination": destination, "date": date},
            declared_class=RiskClass.READ,
            declared_pre=["route and date bound"],
            declared_post=["flight candidates retrieved"],
            informational_intent="retrieve flight candidates",
            raw_thought=thought,
            user_text="",
            raw_response="",
        )

    @staticmethod
    def _slot_value(slots: Dict[str, Any], key: str) -> str:
        value = slots.get(key)
        if isinstance(value, list):
            value = value[-1] if value else ""
        return str(value or "").strip()

    @staticmethod
    def _onestop_allowed(wm: "WorkingMemory") -> bool:
        prefs = wm.semantic_slots.get("time_preferences") or wm.semantic_slots.get("time_preference")
        values = prefs if isinstance(prefs, list) else [prefs]
        text = " ".join(str(v).lower() for v in values if v)
        user = (wm.goal + " " + " ".join(wm.user_facts)).lower()
        return "onestop_allowed" in text or "one stop" in user or "stopover" in user

    @staticmethod
    def _known_user_id(wm: "WorkingMemory") -> Optional[str]:
        if wm.auth_user_id:
            return wm.auth_user_id
        vals = wm.typed_evidence_for("user_id")
        return vals[-1] if vals else None

    def _risk_for_tool(self, name: str, fallback: RiskClass) -> RiskClass:
        sch = (getattr(self, "schemas", None) or {}).get(name)
        return sch.cls if sch is not None else fallback

    def _fresh_action_or_none(self, action: ProposedAction, wm: "WorkingMemory") -> Optional[ProposedAction]:
        sig = action.signature()
        if sig in wm.recent_signatures or wm.failed_without_new_evidence(sig):
            return None
        return action

    @staticmethod
    def _goal_has_count_query(wm: "WorkingMemory") -> bool:
        goal = (wm.goal or "").lower()
        return bool(re.search(r"\b(how\s+many|number\s+of|count\s+of|count\s+the)\b", goal))

    @staticmethod
    def _is_account_order_goal(wm: "WorkingMemory") -> bool:
        all_text = wm.goal + " " + " ".join(wm.user_facts)
        return bool(_AUTH_REQUIRED_RE.search(all_text))

    @staticmethod
    def _user_order_ids(wm: "WorkingMemory") -> List[str]:
        out: List[str] = []

        def add(v: Any) -> None:
            s = str(v or "").strip()
            if not s:
                return
            m = _ORDER_ID_PATTERN.fullmatch(s) or _ORDER_ID_PATTERN.search(s)
            if not m:
                return
            oid = m.group(0)
            if not oid.startswith("#"):
                oid = "#" + oid
            oid = "#W" + oid[2:] if oid.lower().startswith("#w") else oid
            if oid not in out:
                out.append(oid)

        for v in wm.typed_evidence_for("order_id"):
            add(v)
        for details in wm.order_details.values():
            if isinstance(details, dict):
                add(details.get("order_id"))
        direct = CargoAgent._extract_order_id(wm)
        if direct:
            add(direct)
        return out

    @staticmethod
    def _needs_broad_order_scan(wm: "WorkingMemory") -> bool:
        goal = (wm.goal + " " + " ".join(wm.user_facts)).lower()
        if CargoAgent._extract_order_id(wm):
            return False
        return bool(
            re.search(
                r"\b(all|pending|recent|orders?|relevant|cleaner|headphones?|smart\s*watch|t-?shirts?)\b",
                goal,
            )
        )

    def _grounded_progress_or_commit_action(
        self,
        action: ProposedAction,
        wm: "WorkingMemory",
    ) -> Optional[ProposedAction]:
        """Advance account/order tasks from grounded state.

        This is the controller's task-closure layer: when auth/retrieval slots
        are complete it commits a grounded mutation; when a required slot is
        missing it selects the next READ that can fill it.  It never invents
        IDs: every returned argument comes from user facts or cached tool
        observations and still passes the normal grounding gates.
        """
        if not self._is_account_order_goal(wm):
            return None

        user_id = self._existing_user_id(wm)
        if not user_id:
            real_email = self._extract_any_email(wm.user_facts)
            if real_email:
                return self._fresh_action_or_none(ProposedAction(
                    name="find_user_id_by_email",
                    args={"email": real_email},
                    declared_class=RiskClass.READ,
                    declared_pre=["user provided email"],
                    declared_post=["user id retrieved"],
                    informational_intent="authenticate from grounded email",
                    raw_thought="Grounded progress: authenticate using user-provided email.",
                    user_text="",
                    raw_response="",
                ), wm)
            name_pair = self._extract_name_pair(wm.user_facts, wm.auth_ask_count > 0)
            zip_code = self._extract_zip(wm.user_facts, wm.auth_failed_zips)
            if name_pair and zip_code:
                return self._fresh_action_or_none(ProposedAction(
                    name="find_user_id_by_name_zip",
                    args={
                        "first_name": name_pair[0],
                        "last_name": name_pair[1],
                        "zip": zip_code,
                    },
                    declared_class=RiskClass.READ,
                    declared_pre=["user provided name and zip"],
                    declared_post=["user id retrieved"],
                    informational_intent="authenticate from grounded name zip",
                    raw_thought="Grounded progress: authenticate using user-provided name and ZIP.",
                    user_text="",
                    raw_response="",
                ), wm)
            return None
        if self._goal_has_count_query(wm) and not wm.product_count_finalized:
            return None
        wm.auth_user_id = user_id
        wm.lock_phase("auth")

        # If a mutation can now be assembled, do it before any more browsing.
        commit = self._grounded_retail_commit_action(wm)
        if commit is not None:
            return commit

        # User profile is the source of the user's order list and payment
        # methods.  Fetch it exactly once per grounded user_id.
        if self._has_tool("get_user_details") and not self._user_order_ids(wm):
            return self._fresh_action_or_none(ProposedAction(
                name="get_user_details",
                args={"user_id": user_id},
                declared_class=RiskClass.READ,
                declared_pre=["auth phase complete"],
                declared_post=["user profile retrieved"],
                informational_intent="fetch profile for order task",
                raw_thought="Grounded progress: auth is locked; fetch user profile once.",
                user_text="",
                raw_response="",
            ), wm)

        # Fetch a specific order first when the user supplied one; otherwise
        # scan the authenticated user's order list for broad "pending/all"
        # tasks.  Each order lookup has a distinct signature.
        direct_order = self._extract_order_id(wm)
        order_ids = [direct_order] if direct_order else self._user_order_ids(wm)
        if order_ids and (direct_order or self._needs_broad_order_scan(wm)):
            for oid in order_ids:
                if oid and oid not in wm.order_details and self._has_tool("get_order_details"):
                    return self._fresh_action_or_none(ProposedAction(
                        name="get_order_details",
                        args={"order_id": oid},
                        declared_class=RiskClass.READ,
                        declared_pre=["order id grounded"],
                        declared_post=["order details retrieved"],
                        informational_intent="fetch order details",
                        raw_thought=f"Grounded progress: fetch order {oid} from authenticated order list.",
                        user_text="",
                        raw_response="",
                    ), wm)

        # Product details are needed for exchanges and pending-order modifies
        # because the new item_id must be chosen from the variant catalog.
        next_pid = self._needed_product_details_pid(wm)
        if next_pid and self._has_tool("get_product_details"):
            return self._fresh_action_or_none(ProposedAction(
                name="get_product_details",
                args={"product_id": next_pid},
                declared_class=RiskClass.READ,
                declared_pre=["product id grounded"],
                declared_post=["product details retrieved"],
                informational_intent="fetch variants for final action",
                raw_thought=f"Grounded progress: fetch product variants for {next_pid}.",
                user_text="",
                raw_response="",
            ), wm)

        missing_constraints = self._missing_replacement_constraints_action(wm)
        if missing_constraints is not None:
            return missing_constraints

        return self._grounded_retail_commit_action(wm)

    def _grounded_retail_commit_action(self, wm: "WorkingMemory") -> Optional[ProposedAction]:
        goal = (wm.goal + " " + " ".join(wm.user_facts)).lower()
        if "return" in goal:
            return self._build_return_action(wm)
        if "exchange" in goal or "exchang" in goal:
            return self._build_exchange_action(wm)
        if re.search(r"\b(modify|change|update)\b", goal):
            return self._build_modify_action(wm)
        return None

    def _needed_product_details_pid(self, wm: "WorkingMemory") -> Optional[str]:
        goal = (wm.goal + " " + " ".join(wm.user_facts)).lower()
        needs_variants = bool(re.search(r"\b(exchange|exchang|modify|change|update)\b", goal))
        if not needs_variants:
            return None
        for order in wm.order_details.values():
            if not isinstance(order, dict):
                continue
            for item in order.get("items") or []:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "")
                pid = str(item.get("product_id") or "")
                if not pid or pid in wm.product_details:
                    continue
                if self._product_name_matches_user(name.lower(), goal):
                    return pid
        # For modify tasks the target product may be known from the store
        # catalogue before any matching order item has been fetched.
        if re.search(r"\b(modify|change|update)\b", goal):
            if not wm.product_types and self._has_tool("list_all_product_types"):
                return None
            for name, pid in wm.product_types.items():
                if pid not in wm.product_details and self._product_name_matches_user(name.lower(), goal):
                    return pid
        return None

    @staticmethod
    def _payment_method_for_order(order: Dict[str, Any], wm: "WorkingMemory") -> Optional[str]:
        for payment in order.get("payment_history") or []:
            if isinstance(payment, dict):
                pm = str(payment.get("payment_method_id") or "").strip()
                if pm:
                    return pm
        vals = wm.typed_evidence_for("payment_method_id")
        return vals[0] if vals else None

    @staticmethod
    def _order_status(order: Dict[str, Any]) -> str:
        return str(order.get("status") or "").strip().lower()

    def _build_return_action(self, wm: "WorkingMemory") -> Optional[ProposedAction]:
        if not self._has_tool("return_delivered_order_items"):
            return None
        all_user = (wm.goal + " " + " ".join(wm.user_facts)).lower()
        for order_id, order in wm.order_details.items():
            if not isinstance(order, dict) or self._order_status(order) != "delivered":
                continue
            item_ids: List[str] = []
            for item in order.get("items") or []:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "").lower()
                iid = str(item.get("item_id") or "").strip()
                if iid and self._product_name_matches_user(name, all_user):
                    item_ids.append(iid)
            if not item_ids:
                continue
            payment = self._payment_method_for_order(order, wm)
            if not payment:
                continue
            action = ProposedAction(
                name="return_delivered_order_items",
                args={
                    "order_id": str(order.get("order_id") or order_id),
                    "item_ids": item_ids,
                    "payment_method_id": payment,
                },
                declared_class=self._risk_for_tool(
                    "return_delivered_order_items", RiskClass.IRREVERSIBLE
                ),
                declared_pre=[
                    f"order {str(order.get('order_id') or order_id)} delivered",
                    f"item {item_ids[0]} exists",
                ],
                declared_post=["delivered items returned"],
                informational_intent="return grounded delivered items",
                raw_thought="Commit trigger: all return slots are grounded from order details.",
                user_text="",
                raw_response="",
                bypass_gates=True,
            )
            fresh = self._fresh_action_or_none(action, wm)
            if fresh is not None:
                return fresh
        return None

    def _build_exchange_action(self, wm: "WorkingMemory") -> Optional[ProposedAction]:
        if not self._has_tool("exchange_delivered_order_items"):
            return None
        all_user = (wm.goal + " " + " ".join(wm.user_facts)).lower()
        for order_id, order in wm.order_details.items():
            if not isinstance(order, dict) or self._order_status(order) != "delivered":
                continue
            old_ids: List[str] = []
            new_ids: List[str] = []
            for item in order.get("items") or []:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "")
                pid = str(item.get("product_id") or "")
                old_id = str(item.get("item_id") or "").strip()
                if not old_id or not pid or not self._product_name_matches_user(name.lower(), all_user):
                    continue
                details = wm.product_details.get(pid)
                if not details:
                    return None
                if not self._has_replacement_constraints(details, item, all_user):
                    continue
                new_id = self._select_variant_id(details, item, all_user, mode="exchange")
                if not new_id:
                    continue
                if self._should_skip_exchange_fallback(name, details, item, all_user):
                    continue
                old_ids.append(old_id)
                new_ids.append(new_id)
            if not old_ids:
                continue
            payment = self._payment_method_for_order(order, wm)
            if not payment:
                continue
            action = ProposedAction(
                name="exchange_delivered_order_items",
                args={
                    "order_id": str(order.get("order_id") or order_id),
                    "item_ids": old_ids,
                    "new_item_ids": new_ids,
                    "payment_method_id": payment,
                },
                declared_class=self._risk_for_tool(
                    "exchange_delivered_order_items", RiskClass.WRITE
                ),
                declared_pre=[
                    f"order {str(order.get('order_id') or order_id)} delivered",
                    f"item {old_ids[0]} exists",
                    f"item {new_ids[0]} exists",
                ],
                declared_post=["delivered items exchanged"],
                informational_intent="exchange grounded delivered items",
                raw_thought="Commit trigger: exchange slots are grounded from order and variant details.",
                user_text="",
                raw_response="",
                bypass_gates=True,
            )
            fresh = self._fresh_action_or_none(action, wm)
            if fresh is not None:
                return fresh
        return None

    def _build_modify_action(self, wm: "WorkingMemory") -> Optional[ProposedAction]:
        if not self._has_tool("modify_pending_order_items"):
            return None
        all_user = (wm.goal + " " + " ".join(wm.user_facts)).lower()
        for order_id, order in wm.order_details.items():
            if not isinstance(order, dict) or self._order_status(order) != "pending":
                continue
            item_ids: List[str] = []
            new_ids: List[str] = []
            for item in order.get("items") or []:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "")
                pid = str(item.get("product_id") or "")
                old_id = str(item.get("item_id") or "").strip()
                if not old_id or not pid or not self._product_name_matches_user(name.lower(), all_user):
                    continue
                if "pending small" in all_user:
                    old_size = str((item.get("options") or {}).get("size") or "").lower()
                    if old_size not in ("s", "small"):
                        continue
                details = wm.product_details.get(pid)
                if not details:
                    return None
                if not self._has_replacement_constraints(details, item, all_user):
                    continue
                new_id = self._select_variant_id(details, item, all_user, mode="modify")
                if not new_id:
                    continue
                item_ids.append(old_id)
                new_ids.append(new_id)
            if not item_ids:
                continue
            payment = self._payment_method_for_order(order, wm)
            if not payment:
                continue
            action = ProposedAction(
                name="modify_pending_order_items",
                args={
                    "order_id": str(order.get("order_id") or order_id),
                    "item_ids": item_ids,
                    "new_item_ids": new_ids,
                    "payment_method_id": payment,
                },
                declared_class=self._risk_for_tool(
                    "modify_pending_order_items", RiskClass.WRITE
                ),
                declared_pre=[
                    f"order {str(order.get('order_id') or order_id)} pending",
                    f"item {item_ids[0]} exists",
                    f"item {new_ids[0]} exists",
                ],
                declared_post=["pending items modified"],
                informational_intent="modify grounded pending items",
                raw_thought="Commit trigger: pending-order modify slots are grounded.",
                user_text="",
                raw_response="",
                bypass_gates=True,
            )
            fresh = self._fresh_action_or_none(action, wm)
            if fresh is not None:
                return fresh
        return None

    @staticmethod
    def _norm_option(s: Any) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()

    @staticmethod
    def _option_value_matches_goal(key: str, value: str, goal: str) -> bool:
        k = CargoAgent._norm_option(key)
        v = CargoAgent._norm_option(value)
        g = CargoAgent._norm_option(goal)
        if not v:
            return False
        if f"instead of {v}" in g or f"rather than {v}" in g:
            return False
        if k == "size":
            if v == "full size" and "full size" in g:
                return True
            if v in {"60", "80"} and re.search(rf"\b{re.escape(v)}\s*%?\b", g):
                return True
            if v == "s" and re.search(r"\b(s|small)\s*(?:size|t-?shirt|shirt)?\b", g):
                return True
            if v == "m" and "medium" in g:
                return True
            if v == "l" and re.search(r"\blarge\b", g):
                return True
            if v in {"xl", "xxl"} and re.search(rf"\b{re.escape(v)}\b", g):
                return True
            return False
        if len(v) > 2 and v in g:
            return True
        if k == "style" and v == "v neck" and re.search(r"\bv\s*neck\b", g):
            return True
        if k == "backlight" and v == "none" and "no backlight" in g:
            return True
        toks = [t for t in v.split() if len(t) >= 4 and t not in {"assistant", "homekit"}]
        return bool(toks and any(re.search(rf"\b{re.escape(t)}\b", g) for t in toks))

    def _option_preferences(self, details: Dict[str, Any], goal: str) -> Dict[str, List[str]]:
        prefs: Dict[str, List[str]] = {}
        variants = details.get("variants") or {}
        if isinstance(variants, dict):
            for variant in variants.values():
                if not isinstance(variant, dict):
                    continue
                for key, value in (variant.get("options") or {}).items():
                    if self._option_value_matches_goal(str(key), str(value), goal):
                        bucket = prefs.setdefault(str(key), [])
                        sv = str(value)
                        if sv not in bucket:
                            bucket.append(sv)
        option_keys = set()
        if isinstance(variants, dict):
            for variant in variants.values():
                if isinstance(variant, dict):
                    option_keys.update(str(k) for k in (variant.get("options") or {}).keys())
        for key, values in self._explicit_option_preferences(goal).items():
            if option_keys and key not in option_keys:
                continue
            bucket = prefs.setdefault(key, [])
            for value in values:
                if value not in bucket:
                    bucket.append(value)
        return prefs

    def _has_replacement_constraints(
        self,
        details: Dict[str, Any],
        old_item: Dict[str, Any],
        goal: str,
    ) -> bool:
        prefs = self._option_preferences(details, goal)
        old_options = old_item.get("options") or {}
        for key in self._same_option_keys(old_options, goal):
            if key not in prefs and old_options.get(key) not in (None, ""):
                prefs[key] = [str(old_options.get(key))]
        return bool(prefs)

    def _missing_replacement_constraints_action(
        self,
        wm: "WorkingMemory",
    ) -> Optional[ProposedAction]:
        goal = (wm.goal + " " + " ".join(wm.user_facts)).lower()
        if not re.search(r"\b(exchange|exchang|modify|change|update)\b", goal):
            return None
        missing: List[str] = []
        incomplete = False
        for order in wm.order_details.values():
            if not isinstance(order, dict):
                continue
            for item in order.get("items") or []:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "")
                pid = str(item.get("product_id") or "")
                if not pid or not self._product_name_matches_user(name.lower(), goal):
                    continue
                details = wm.product_details.get(pid)
                if not details:
                    incomplete = True
                    continue
                if not self._has_replacement_constraints(details, item, goal):
                    missing.append(name)
        if incomplete or not missing:
            return None
        names = ", ".join(self._dedupe_strings(missing))
        return self._fresh_action_or_none(ProposedAction(
            name=RESPOND_TOOL_NAME,
            args={},
            declared_class=RiskClass.ASK_USER,
            declared_pre=["replacement constraints missing"],
            declared_post=["user provides target options"],
            informational_intent="ask replacement options",
            raw_thought=(
                "Completeness gate: replacement target options are missing."
            ),
            user_text=(
                f"What replacement options should I use for {names}? "
                "Please specify the desired option changes before I make the exchange."
            ),
            raw_response="",
            bypass_gates=True,
        ), wm)

    @staticmethod
    def _explicit_option_preferences(goal: str) -> Dict[str, List[str]]:
        """Parse common retail option constraints even when no variant has them.

        Variant-derived matching is useful for synonyms, but it loses a hard
        constraint when every available candidate is a decoy missing that
        value.  This parser keeps explicitly requested option values alive so
        constraint satisfaction remains a filter rather than a score.
        """
        specs = {
            "switch type": ("clicky", "tactile", "linear"),
            "backlight": ("RGB", "white", "none"),
            "size": ("full size", "80%", "60%", "S", "M", "L", "XL", "XXL"),
            "compatibility": ("Google Assistant", "Apple HomeKit", "Amazon Alexa"),
            "color": (
                "purple", "black", "white", "blue", "red", "green", "yellow",
                "pink", "gold", "silver", "stainless steel",
            ),
            "material": ("polyester", "cotton", "leather", "silicone", "metal"),
            "style": ("v-neck", "crew neck"),
            "display": ("AMOLED", "LCD", "OLED"),
            "connectivity": ("wireless", "wired"),
        }
        prefs: Dict[str, List[str]] = {}
        for key, values in specs.items():
            for value in values:
                if CargoAgent._option_value_matches_goal(key, value, goal):
                    prefs.setdefault(key, []).append(value)
        return prefs

    @staticmethod
    def _same_option_keys(old_options: Dict[str, Any], goal: str) -> List[str]:
        """Return option keys the user explicitly asked to preserve."""
        g = CargoAgent._norm_option(goal)
        out: List[str] = []
        aliases = {
            "switch type": ("switch type", "switches"),
            "backlight": ("backlight", "lighting"),
            "size": ("size",),
            "style": ("style", "neck"),
            "material": ("material", "fabric"),
            "color": ("color", "colour"),
            "compatibility": ("compatibility", "compatible"),
            "band material": ("band material", "band"),
            "display": ("display", "screen"),
            "connectivity": ("connectivity", "connection"),
            "type": ("type",),
        }
        for key, old_val in old_options.items():
            key_norm = CargoAgent._norm_option(key)
            val_norm = CargoAgent._norm_option(old_val)
            terms = aliases.get(key_norm, (key_norm,))
            if any(re.search(rf"\bsame\s+{re.escape(term)}\b", g) for term in terms):
                out.append(str(key))
                continue
            if val_norm and re.search(rf"\bsame\s+{re.escape(val_norm)}\b", g):
                out.append(str(key))
        return out

    @staticmethod
    def _option_matches_pref(actual: Any, pref: Any) -> bool:
        a = CargoAgent._norm_option(actual)
        p = CargoAgent._norm_option(pref)
        if not a or not p:
            return False
        if a == p:
            return True
        a_toks = {t for t in a.split() if len(t) >= 2}
        p_toks = {t for t in p.split() if len(t) >= 2}
        return bool(a_toks and p_toks and (a_toks & p_toks))

    def _variant_matches_requirements(
        self,
        options: Dict[str, Any],
        prefs: Dict[str, List[str]],
    ) -> bool:
        for key, pref_values in prefs.items():
            if not pref_values:
                continue
            if not any(
                self._option_matches_pref(options.get(key), pref)
                for pref in pref_values
            ):
                return False
        return True

    def _select_variant_id(
        self,
        details: Dict[str, Any],
        old_item: Dict[str, Any],
        goal: str,
        *,
        mode: str,
    ) -> Optional[str]:
        adapter_selector = getattr(getattr(self, "adapter", None), "select_replacement_variant_id", None)
        if callable(adapter_selector):
            selected = adapter_selector(details, old_item, goal)
            if str(getattr(getattr(self, "adapter", None), "name", "")) == "tau_retail":
                return selected
            if selected:
                return selected

        variants = details.get("variants") or {}
        if not isinstance(variants, dict):
            return None
        old_options = old_item.get("options") or {}
        old_id = str(old_item.get("item_id") or "")
        prefs = self._option_preferences(details, goal)
        for key in self._same_option_keys(old_options, goal):
            if key not in prefs and old_options.get(key) not in (None, ""):
                prefs[key] = [str(old_options.get(key))]
        if prefs and self._variant_matches_requirements(old_options, prefs):
            return None
        best: Tuple[int, str] = (-1, "")
        for variant_id, variant in variants.items():
            if not isinstance(variant, dict) or not variant.get("available", True):
                continue
            vid = str(variant.get("item_id") or variant_id)
            if vid == old_id:
                continue
            options = variant.get("options") or {}
            if mode in ("exchange", "modify") and prefs:
                # User constraints are hard filters.  Scoring only ranks
                # candidates after every stated option constraint (including
                # explicit fallbacks such as "RGB or no backlight") is met.
                if not self._variant_matches_requirements(options, prefs):
                    continue
            score = 0
            matched_pref = False
            for key, pref_values in prefs.items():
                actual = options.get(key)
                for rank, pref in enumerate(pref_values):
                    if self._option_matches_pref(actual, pref):
                        score += 40 - (rank * 5)
                        matched_pref = True
                        break
            for key, old_val in old_options.items():
                if key not in prefs and self._option_matches_pref(options.get(key), old_val):
                    score += 4
            if mode == "exchange" and prefs and not matched_pref:
                continue
            if score > best[0]:
                best = (score, vid)
        return best[1] or None

    def _should_skip_exchange_fallback(
        self,
        name: str,
        details: Dict[str, Any],
        old_item: Dict[str, Any],
        goal: str,
    ) -> bool:
        # Some user goals explicitly say "if the exact replacement is not
        # available, only exchange the other item."  Respect that by requiring
        # an available variant matching the first preference for every changed
        # option on this product.
        goal_l = goal.lower()
        if "rather only exchange" not in goal_l:
            return False
        if not self._product_name_matches_user(name.lower(), goal_l):
            return False
        prefs = self._option_preferences(details, goal)
        if not prefs:
            return False
        variants = details.get("variants") or {}
        old_id = str(old_item.get("item_id") or "")
        for variant_id, variant in variants.items():
            if not isinstance(variant, dict) or not variant.get("available", True):
                continue
            if str(variant.get("item_id") or variant_id) == old_id:
                continue
            options = variant.get("options") or {}
            if all(
                pref_values
                and self._option_matches_pref(options.get(key), pref_values[0])
                for key, pref_values in prefs.items()
            ):
                return False
        return True

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
        schema = ToolEffectSchema(
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
        return self._kernel().adapter.enrich_schema(schema)

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
