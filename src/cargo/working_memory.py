"""Typed working memory for the CARGO agent.

The agent never edits memory through the LLM; the proposer reads from it,
and *deterministic* code (tool-result parsing, gate diagnostics) writes to
it. This is the DST-aware design: facts revealed by the user are kept
separate from facts confirmed by a tool/DB observation, and only
DB-confirmed facts are authoritative for irreversible actions.
"""
from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional


# A short rolling window of recent action signatures (for repeat detection).
_RECENT_WINDOW = 5


@dataclass
class WorkingMemory:
    goal: str = ""
    user_facts: List[str] = field(default_factory=list)
    db_facts: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    pending_obligations: List[str] = field(default_factory=list)
    last_obs: Any = ""
    last_error: str = ""
    budget_steps: int = 30
    recent_signatures: Deque[str] = field(default_factory=lambda: deque(maxlen=_RECENT_WINDOW))
    failed_signatures: Dict[str, int] = field(default_factory=dict)
    evidence_version: int = 0
    typed_values: Dict[str, List[str]] = field(default_factory=dict)
    # CARGO-v2 semantic task state.  ``typed_values`` stores opaque IDs;
    # semantic_slots stores ordinary task facts such as dates, routes, cabin,
    # baggage count, insurance choice, payment preferences, and operations.
    # DB-confirmed slots outrank later user claims.
    semantic_slots: Dict[str, Any] = field(default_factory=dict)
    db_confirmed_slots: Dict[str, bool] = field(default_factory=dict)
    # Auth-loop guard: how many times we've already asked the user for identity
    # credentials.  The override stops asking after 2 attempts to avoid an
    # infinite ASK_USER bounce.
    auth_ask_count: int = 0
    # ZIPs that have already been tried with find_user_id_by_name_zip and
    # returned a not-found / error.  The override skips them on retry.
    auth_failed_zips: List[str] = field(default_factory=list)
    # Set once the user has refused to share credentials OR the ask budget is
    # exhausted.  When true, the auth override stops triggering and the agent
    # falls back to no-auth pathways (e.g. product-only queries) or emits a
    # single FINAL apology.  This breaks the infinite ASK→FINAL→ASK loop.
    auth_abandoned: bool = False
    # Set once a user_id has been confirmed by a successful find_user_id_*
    # call.  Used to cleanly suppress further authentication proposals.
    auth_user_id: str = ""
    # Durable phase locks.  These are deterministic controller state, not LLM
    # memory: once a phase completes (auth, product count, etc.), future
    # proposals must advance from that state instead of re-entering it.
    phase_locks: Dict[str, bool] = field(default_factory=dict)
    # Signatures of successful state-changing actions.  Unlike
    # recent_signatures, this is not a short rolling window: a write that has
    # already succeeded must not be retried later in the same task.
    executed_mutations: Dict[str, int] = field(default_factory=dict)
    # Set after a successful state change or terminal final answer.  The
    # controller checks this before each proposer call so no more tools run
    # after the task is complete.
    task_completed: bool = False
    # Set once we've already emitted a "give up on auth" FINAL.  Prevents the
    # exact same FINAL from being emitted twice in a row.
    auth_giveup_emitted: bool = False
    # Confirmed email from a successful get_user_details / find_user_id_by_email
    # call.  Stored here so it is NEVER evicted by the 48-entry db_facts LRU
    # and so render_compact() can surface it directly to the proposer.
    # Without this, get_user_details floods db_facts with variant/payment data
    # that pushes out the email entry, breaking the auth-confirmation loop in
    # Path 1 of _auth_override.  (Observed: trajectories(24) T0.)
    auth_email: str = ""
    # Cache of product details keyed by product_id.  Populated by the agent's
    # solve loop when ``get_product_details`` returns successfully.  Used to
    # short-circuit "how many X variants are available?" queries that the
    # model fails to finalize on its own.
    product_details: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Set once we've emitted a deterministic answer for a product-count
    # query.  Prevents re-emitting the same FINAL on each subsequent step.
    product_count_finalized: bool = False
    # Catalog of product types — populated by the solve loop when
    # ``list_all_product_types`` returns.  Maps product name → product_id.
    # Lives outside ``db_facts`` so it survives the LRU cap (48 entries),
    # which would otherwise evict product types as soon as a single
    # ``get_product_details`` response floods db_facts with variant info.
    # Observed eviction failure: trajectories(20) T1/T2/T3 lost the
    # Headphones/Cleaner/Smartwatch entries after T-Shirt details came in.
    product_types: Dict[str, str] = field(default_factory=dict)
    # Cache of order details keyed by order_id.  Populated by the solve
    # loop when ``get_order_details`` returns.  Used by downstream tooling
    # to look up items in the order without re-fetching.
    order_details: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Durable profile / reservation caches for airline-style domains.  These
    # prevent state loss after large observations evict prompt-facing db_facts.
    user_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    reservation_details: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Track the last respond/FINAL message we emitted so the controller can
    # detect a "same FINAL on every step" infinite loop and break out.
    last_final_text: str = ""
    consecutive_same_final: int = 0

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------
    def absorb_user_message(self, text: str) -> None:
        """Record facts revealed by the user in NL.

        The user can lie or be mistaken, so these facts are NOT yet
        authoritative for irreversible actions; they unlock READ tools and
        seed argument-grounding candidates.
        """
        if not text:
            return
        clean = text.strip()
        if not clean:
            return
        changed = False
        if clean not in self.user_facts:
            self.user_facts.append(clean)
            changed = True
        # Extract atomic candidate values (proper nouns, IDs, emails) as
        # individual fact lines so arg-grounding has fine-grained anchors.
        for tok in _extract_id_tokens(clean):
            if tok not in self.user_facts:
                self.user_facts.append(tok)
                changed = True
            for key in _typed_keys_for_token(tok):
                before = list(self.typed_values.get(key, []))
                self._add_typed_value(key, tok)
                if self.typed_values.get(key, []) != before:
                    changed = True
        for key, value in _extract_labeled_values(clean):
            if value not in self.user_facts:
                self.user_facts.append(value)
                changed = True
            before = list(self.typed_values.get(_normalize_typed_key(key), []))
            self._add_typed_value(key, value)
            if self.typed_values.get(_normalize_typed_key(key), []) != before:
                changed = True
        if self._bind_user_semantics(clean):
            changed = True
        if changed:
            self.evidence_version += 1

    def absorb_observation(self, obs: Any) -> None:
        """Promote scalar key/value pairs in a tool observation to db_facts."""
        if obs is None or obs == "":
            return
        self.last_obs = obs
        struct = _coerce_struct(obs)
        if isinstance(struct, dict):
            self._absorb_dict(struct, prefix="")
        elif isinstance(struct, list):
            for i, item in enumerate(struct):
                if isinstance(item, dict):
                    self._absorb_dict(item, prefix=f"[{i}]")
        else:
            # Plain string observation: extract IDs as bare facts.
            for tok in _extract_id_tokens(str(struct)):
                self._add_db_fact(tok)

    def _absorb_dict(self, d: Dict[str, Any], prefix: str) -> None:
        for k, v in d.items():
            kpath = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (str, int, float)) and v not in (None, ""):
                self._add_typed_value(kpath, v)
                self._add_semantic_slot(kpath, v, confirmed=True)
                self._add_db_fact(f"{kpath}={v}")
                # Also store the value alone (helps arg-grounding substring).
                if isinstance(v, str) and v.strip():
                    self._add_db_fact(v.strip())
            elif isinstance(v, dict):
                self._absorb_dict(v, prefix=kpath)
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        self._absorb_dict(item, prefix=f"{kpath}[{i}]")
                    elif isinstance(item, (str, int, float)) and item not in (None, ""):
                        self._add_typed_value(kpath, item)
                        self._add_semantic_slot(kpath, item, confirmed=True)
                        self._add_db_fact(f"{kpath}[{i}]={item}")

    def _bind_user_semantics(self, text: str) -> bool:
        changed = False
        for key, value in _extract_semantic_slots(text):
            if self._add_semantic_slot(key, value, confirmed=False):
                changed = True
        return changed

    def _add_semantic_slot(self, key_path: str, value: Any, *, confirmed: bool) -> bool:
        key = _normalize_semantic_key(key_path)
        if not key:
            return False
        if key == "intent":
            return self._add_semantic_list_value("intents", value, confirmed=confirmed)
        if key == "payment_preference":
            return self._add_semantic_list_value("payment_preferences", value, confirmed=confirmed)
        val = str(value).strip() if not isinstance(value, (int, float)) else value
        if val in (None, ""):
            return False
        if self.db_confirmed_slots.get(key) and not confirmed:
            return False
        cur = self.semantic_slots.get(key)
        if cur == val:
            if confirmed and not self.db_confirmed_slots.get(key):
                self.db_confirmed_slots[key] = True
                return True
            return False
        self.semantic_slots[key] = val
        if confirmed:
            self.db_confirmed_slots[key] = True
        return True

    def _add_semantic_list_value(self, key: str, value: Any, *, confirmed: bool) -> bool:
        val = str(value).strip()
        if not val:
            return False
        cur = self.semantic_slots.setdefault(key, [])
        if not isinstance(cur, list):
            cur = [str(cur)]
            self.semantic_slots[key] = cur
        if val in cur:
            return False
        cur.append(val)
        if confirmed:
            self.db_confirmed_slots[key] = True
        return True

    def _add_db_fact(self, fact: str) -> None:
        if not fact:
            return
        fact = fact[:100]  # tip 5: short canonical form
        if fact in self.db_facts:
            return
        self.db_facts.append(fact)
        self.evidence_version += 1
        # Hard cap: keep only the 48 most-recent facts.  Older facts are the
        # ones least likely to be needed for the current gate or proposer call.
        if len(self.db_facts) > 48:
            self.db_facts = self.db_facts[-48:]

    def _add_typed_value(self, key_path: str, value: Any) -> None:
        val = str(value).strip()
        if not val:
            return
        keys = _typed_keys_for_path(key_path, val)
        for key in keys:
            bucket = self.typed_values.setdefault(key, [])
            if val not in bucket:
                bucket.append(val)
                if len(bucket) > 64:
                    del bucket[:-64]

    def record_action_signature(self, sig: str) -> None:
        if sig:
            self.recent_signatures.append(sig)

    def record_failed_signature(self, sig: str) -> None:
        """Remember a failed proposal until new evidence arrives.

        A retry is useful only after the state changes or the action changes.
        Tying failed signatures to ``evidence_version`` prevents loops on the
        exact same rejected action while still allowing the same action after a
        new tool observation or user reply grounds it.
        """
        if sig:
            self.failed_signatures[sig] = self.evidence_version
            if len(self.failed_signatures) > 64:
                stale = list(self.failed_signatures.keys())[:-64]
                for k in stale:
                    self.failed_signatures.pop(k, None)

    def failed_without_new_evidence(self, sig: str) -> bool:
        return bool(sig and self.failed_signatures.get(sig) == self.evidence_version)

    def record_executed_mutation(self, sig: str) -> None:
        if sig:
            self.executed_mutations[sig] = self.evidence_version

    def mutation_already_executed(self, sig: str) -> bool:
        return bool(sig and sig in self.executed_mutations)

    def lock_phase(self, phase: str) -> None:
        key = str(phase or "").strip().lower()
        if key:
            self.phase_locks[key] = True

    def phase_locked(self, phase: str) -> bool:
        return bool(self.phase_locks.get(str(phase or "").strip().lower()))

    # ------------------------------------------------------------------
    # Queries used by gates
    # ------------------------------------------------------------------
    def all_evidence(self) -> str:
        """Concatenated evidence string used for substring grounding.

        Includes ``product_types`` values so that the arg-grounding gate
        accepts product IDs resolved from the durable catalogue even after
        the original ``list_all_product_types`` JSON response has been evicted
        from ``db_facts`` by a subsequent large tool response (e.g.
        get_product_details flooding db_facts with variant data).

        Also includes ``auth_user_id`` and ``auth_email`` so that confirmed
        identity tokens pass arg_grounding even after the corresponding
        db_facts entries are evicted by the LRU cap.
        (Observed failure: trajectories(24) T0 — email evicted after
        get_user_details flooded db_facts, breaking auth-confirmation loop.)

        The durable structured caches are included too.  ``db_facts`` is a
        short prompt-facing LRU, but gates are correctness machinery; they must
        see all grounded order item IDs, payment IDs, product variant IDs, and
        option values that came from tool observations.  Without this, a large
        ``get_product_details`` response can evict the very item IDs needed for
        a later WRITE precondition, causing useless ASK_USER churn even though
        the controller already has grounded state.
        """
        base = "\n".join(self.user_facts + self.db_facts)
        extras: list = []
        if self.product_types:
            extras.append("\n".join(self.product_types.keys()))
            extras.append("\n".join(self.product_types.values()))
        if self.auth_user_id:
            extras.append(self.auth_user_id)
        if self.auth_email:
            extras.append(self.auth_email)
        for key, value in self.semantic_slots.items():
            if isinstance(value, list):
                extras.extend(str(v) for v in value)
            else:
                extras.append(f"{key}={value}")
                extras.append(str(value))
        for order in self.order_details.values():
            extras.extend(_flatten_scalar_facts(order))
        for profile in self.user_profiles.values():
            extras.extend(_flatten_scalar_facts(profile))
        for reservation in self.reservation_details.values():
            extras.extend(_flatten_scalar_facts(reservation))
        for details in self.product_details.values():
            extras.extend(_flatten_scalar_facts(details))
        if extras:
            return base + "\n" + "\n".join(extras)
        return base

    def typed_evidence_for(self, field_name: str) -> List[str]:
        """Return grounded values known for a specific ID-like argument field."""
        key = _normalize_typed_key(field_name)
        vals: List[str] = []

        def add(v: Any) -> None:
            sv = str(v).strip()
            if sv and sv not in vals:
                vals.append(sv)

        for v in self.typed_values.get(key, []):
            add(v)
        if key == "product_id":
            for v in self.product_types.values():
                add(v)
            for details in self.order_details.values():
                if isinstance(details, dict):
                    for item in details.get("items") or []:
                        if isinstance(item, dict):
                            add(item.get("product_id"))
        elif key == "item_id":
            for details in self.product_details.values():
                if isinstance(details, dict):
                    variants = details.get("variants") or {}
                    if isinstance(variants, dict):
                        for variant_id, variant in variants.items():
                            add(variant_id)
                            if isinstance(variant, dict):
                                add(variant.get("item_id"))
            for details in self.order_details.values():
                if isinstance(details, dict):
                    for item in details.get("items") or []:
                        if isinstance(item, dict):
                            add(item.get("item_id"))
        elif key == "user_id":
            add(self.auth_user_id)
        elif key == "email":
            add(self.auth_email)
        elif key == "order_id":
            for details in self.order_details.values():
                if isinstance(details, dict):
                    add(details.get("order_id"))
        elif key == "reservation_id":
            for profile in self.user_profiles.values():
                if isinstance(profile, dict):
                    for rid in profile.get("reservations") or []:
                        add(rid)
            for details in self.reservation_details.values():
                if isinstance(details, dict):
                    add(details.get("reservation_id"))
                    add(details.get("reservation_number"))
        return vals

    def render_compact(self, max_chars: int = 800) -> str:
        """Compact NL render used in the proposer prompt.

        Design choices (context-optimization tips):
        - No ``last_obs`` dump: its key fields are already in db_confirmed_facts
          (absorbed by absorb_observation), so including the raw JSON is pure
          duplication (tip 6).
        - Small item windows: only the most-recent facts matter for the current
          step; older ones are already captured in db_facts from prior turns.
        - Hard char cap (800) keeps STATE ≈ 200 tokens (tip 10).
        """
        def trim(items: List[str], cap: int) -> List[str]:
            return items[-cap:]

        parts = [
            f"goal: {self.goal[:150]}" if self.goal else "",
        ]
        # Surface confirmed identity explicitly so the proposer always sees
        # them even when db_facts has been evicted (48-entry LRU cap).
        # Without these lines the model keeps proposing find_user_id_* after
        # auth is complete because it can't see the user_id in truncated STATE.
        # (Observed failure: trajectories(24) T0.)
        if self.auth_user_id:
            parts.append(f"confirmed_user_id: {self.auth_user_id}")
            parts.append("phase_locked: auth")
        if self.auth_email:
            parts.append(f"confirmed_email: {self.auth_email}")
        if not self.auth_user_id:
            user_ids = self.typed_evidence_for("user_id")
            if user_ids:
                parts.append(f"known_user_id: {user_ids[-1]}")
        if self.product_count_finalized:
            parts.append("phase_locked: product_count")
        if self.phase_locked("mutation"):
            parts.append("phase_locked: mutation")
        if self.task_completed:
            parts.append("task_completed: true")
        if self.order_details:
            parts.append(f"orders_cached: {len(self.order_details)}")
        if self.user_profiles:
            parts.append(f"user_profiles_cached: {len(self.user_profiles)}")
        if self.reservation_details:
            parts.append(f"reservations_cached: {len(self.reservation_details)}")
        if self.semantic_slots:
            rendered = []
            for key in sorted(self.semantic_slots.keys()):
                val = self.semantic_slots[key]
                if isinstance(val, list):
                    val_s = ",".join(str(v) for v in val[-4:])
                else:
                    val_s = str(val)
                rendered.append(f"{key}={val_s}")
            parts.append("task_slots: " + "; ".join(rendered)[:220])
        parts += [
            "user_facts:",
            *(f"  {f[:90]}" for f in trim(self.user_facts, 5)),
            "db_facts:",
            *(f"  {f[:90]}" for f in trim(self.db_facts, 8)),
        ]
        if self.assumptions:
            parts += ["assumptions:",
                      *(f"  {a[:80]}" for a in trim(self.assumptions, 3))]
        if self.pending_obligations:
            parts += ["obligations:",
                      *(f"  {o[:80]}" for o in trim(self.pending_obligations, 2))]
        # last_error: single short line; empty string suppresses the field (tip 8).
        if self.last_error:
            parts.append(f"last_error: {self.last_error[:80]}")
        parts.append(f"steps_left: {self.budget_steps}")
        out = "\n".join(p for p in parts if p)
        return out[:max_chars]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]{3,}")
_USER_ID_RE = re.compile(r"\b[a-z]+_[a-z]+_\d{1,8}\b")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
# tau-bench retail order IDs: "#W" + digits, with or without the leading #.
# We capture both forms so arg_grounding accepts either.
_ORDER_ID_LOOSE_RE = re.compile(r"#?[Ww](\d{5,10})")
_RESERVATION_ID_RE = re.compile(r"\b[A-Z0-9]{6}\b")

_TYPED_ID_KEYS = {
    "account_id", "booking_id", "customer_id", "email", "flight_id",
    "item_id", "order_id", "payment_id", "payment_method_id",
    "product_id", "reservation_id", "session_id", "ticket_id", "user_id",
}


def _normalize_typed_key(field_name: str) -> str:
    key = str(field_name or "").strip().lower()
    key = re.sub(r"\[\d+\]", "", key)
    key = key.split(".")[-1]
    if key in ("item_ids", "new_item_ids"):
        return "item_id"
    if key.endswith("_ids"):
        return key[:-1]
    if key.endswith("ids") and len(key) > 3:
        return key[:-1]
    return key


def _typed_keys_for_path(key_path: str, value: str) -> List[str]:
    key = _normalize_typed_key(key_path)
    out: List[str] = []
    if key in _TYPED_ID_KEYS:
        out.append(key)
    # tau retail stores a user's order list under "orders"; those values are
    # order IDs even though the key is plural and not an argument name.
    if key == "orders" and _ORDER_ID_LOOSE_RE.fullmatch(value):
        out.append("order_id")
    # tau airline stores a user's reservation list under "reservations".
    if key in ("reservations", "reservation_numbers") and _RESERVATION_ID_RE.fullmatch(value):
        out.append("reservation_id")
    if key in ("payment_methods", "payment_history") and value:
        out.append("payment_method_id")
    # Preserve alternate order-id spellings for grounding after normalization.
    if "order_id" in out:
        m = _ORDER_ID_LOOSE_RE.fullmatch(value)
        if m:
            out.extend(["order_id"])
    dedup: List[str] = []
    for item in out:
        if item not in dedup:
            dedup.append(item)
    return dedup


def _normalize_semantic_key(key_path: str) -> str:
    key = str(key_path or "").strip().lower()
    key = re.sub(r"\[\d+\]", "", key)
    key = key.split(".")[-1]
    key = key.replace("-", "_")
    aliases = {
        "departure_date": "date",
        "arrival_date": "date",
        "scheduled_departure_date": "date",
        "scheduled_arrival_date": "date",
        "origin_airport": "origin",
        "destination_airport": "destination",
        "cabin_class": "cabin",
        "class": "cabin",
        "trip": "trip_type",
        "insurance": "travel_insurance",
        "total_bags": "baggage_count",
        "total_baggages": "baggage_count",
        "checked_bags": "baggage_count",
        "checked_baggages": "baggage_count",
    }
    key = aliases.get(key, key)
    semantic_keys = {
        "date", "origin", "destination", "cabin", "trip_type",
        "baggage_count", "travel_insurance", "intent",
        "payment_preference", "time_preference",
    }
    return key if key in semantic_keys else ""


_MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
}


def _extract_semantic_slots(text: str) -> List[tuple]:
    s = str(text or "")
    if not s:
        return []
    out: List[tuple] = []
    seen: set = set()

    def push(key: str, value: Any) -> None:
        if value in (None, ""):
            return
        pair = (key, str(value))
        if pair in seen:
            return
        seen.add(pair)
        out.append((key, value))

    low = s.lower()
    if re.search(r"\b(book|reserve|purchase)\b", low):
        push("intent", "book")
    if re.search(r"\b(cancel)\b", low):
        push("intent", "cancel")
    if re.search(r"\b(change|modify|upgrade|downgrade|update)\b", low):
        push("intent", "modify")
    if re.search(r"\b(exchange|swap|replace)\b", low):
        push("intent", "exchange")
    if re.search(r"\b(return|refund)\b", low):
        push("intent", "return")

    for m in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", s):
        push("date", m.group(0))
    for m in re.finditer(
        r"\b("
        + "|".join(sorted(_MONTHS, key=len, reverse=True))
        + r")\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(\d{4}))?\b",
        low,
    ):
        year = int(m.group(3) or 2024)
        month = _MONTHS[m.group(1)]
        day = int(m.group(2))
        if 1 <= day <= 31:
            push("date", f"{year:04d}-{month:02d}-{day:02d}")

    route = re.search(r"\bfrom\s+([A-Za-z][A-Za-z .'-]{1,40}?)\s+to\s+([A-Za-z][A-Za-z .'-]{1,40}?)(?:[.,;]| on | in | at |$)", s, re.I)
    if route:
        push("origin", _clean_semantic_phrase(route.group(1)))
        push("destination", _clean_semantic_phrase(route.group(2)))
    depart = re.search(r"\b(?:departing|leaving|flying out)\s+from\s+([A-Z]{3}|[A-Za-z][A-Za-z .'-]{1,40})(?:[.,;]| on | in | at |$)", s, re.I)
    if depart:
        push("origin", _clean_semantic_phrase(depart.group(1)))

    if "basic economy" in low:
        push("cabin", "basic economy")
    elif "business class" in low or re.search(r"\bbusiness\b", low):
        push("cabin", "business")
    elif re.search(r"\beconomy\b", low):
        push("cabin", "economy")

    if re.search(r"\bround[- ]trip\b", low):
        push("trip_type", "round trip")
    if re.search(r"\bone[- ]way\b", low):
        push("trip_type", "one way")
    if re.search(r"\breturn (?:flight|trip)\b", low):
        push("trip_type", "round trip")

    bag_match = re.search(r"\b(\d+|zero|one|two|three|four|five|six|seven|eight|nine)\s+checked\s+bags?\b", low)
    if bag_match:
        raw = bag_match.group(1)
        push("baggage_count", int(raw) if raw.isdigit() else _NUMBER_WORDS.get(raw))

    if re.search(r"\b(no|without|do not want|don't want)\s+(?:travel\s+)?insurance\b", low):
        push("travel_insurance", "no")
    elif re.search(r"\b(?:buy|get|add|want|need|using my)\s+(?:travel\s+)?insurance\b", low):
        push("travel_insurance", "yes")

    if "certificate" in low:
        push("payment_preference", "certificate")
    if "gift card" in low or "gift_card" in low:
        push("payment_preference", "gift_card")
    if "credit card" in low or "card" in low:
        push("payment_preference", "credit_card")

    if re.search(r"\bcheapest\b", low):
        push("time_preference", "cheapest")
    if re.search(r"\bdirect\b", low):
        push("time_preference", "direct_preferred")
    if re.search(r"\bone[- ]?stop|stopover\b", low):
        push("time_preference", "onestop_allowed")
    if re.search(r"\bafter\s+\d{1,2}\s*(?:am|pm)?\b", low):
        push("time_preference", re.search(r"\bafter\s+\d{1,2}\s*(?:am|pm)?\b", low).group(0))
    return out


def _clean_semantic_phrase(value: str) -> str:
    val = re.sub(r"\s+", " ", str(value or "")).strip(" .,;:!?")
    return val


def _typed_keys_for_token(token: str) -> List[str]:
    tok = str(token or "").strip()
    if not tok:
        return []
    out: List[str] = []
    if _USER_ID_RE.fullmatch(tok):
        out.append("user_id")
    if _EMAIL_RE.fullmatch(tok):
        out.append("email")
    if _ORDER_ID_LOOSE_RE.fullmatch(tok):
        out.append("order_id")
    if _RESERVATION_ID_RE.fullmatch(tok):
        out.append("reservation_id")
    if tok.startswith(("credit_card_", "gift_card_", "certificate_")):
        out.extend(["payment_id", "payment_method_id"])
    dedup: List[str] = []
    for item in out:
        if item not in dedup:
            dedup.append(item)
    return dedup


_LABEL_ALIASES = {
    "user_id": ("user id", "user_id", "userid"),
    "reservation_id": ("reservation id", "reservation_id", "booking id", "booking_id"),
    "order_id": ("order id", "order number", "order_id"),
    "payment_id": ("payment id", "payment_id", "payment method", "payment_method_id"),
}


def _extract_labeled_values(text: str) -> List[tuple]:
    """Extract field-labelled values from user text.

    This is intentionally conservative: it binds values only when the user
    labels the field directly ("my user ID is ...", "reservation_id: ...").
    Free-floating opaque tokens still remain evidence but are not assigned to
    arbitrary argument fields unless their shape is unambiguous.
    """
    out: List[tuple] = []
    seen: set = set()

    def push(key: str, value: str) -> None:
        value = value.strip().strip(".,;:!?()[]{}\"'")
        if not value:
            return
        pair = (key, value)
        if pair in seen:
            return
        seen.add(pair)
        out.append(pair)

    for key, aliases in _LABEL_ALIASES.items():
        for alias in aliases:
            alias_pat = re.escape(alias).replace(r"\ ", r"\s+")
            pattern = re.compile(
                rf"\b{alias_pat}\b"
                r"(?:\s*(?:is|=|:|#|number)?\s*)"
                r"([A-Za-z0-9][A-Za-z0-9_\-]{2,80})",
                re.IGNORECASE,
            )
            for m in pattern.finditer(text):
                candidate = m.group(1)
                if candidate.lower() in {"not", "unknown", "missing", "forgot", "remembered"}:
                    continue
                push(key, candidate)
    return out


def _extract_id_tokens(s: str) -> List[str]:
    if not s:
        return []
    out: List[str] = []
    seen: set = set()

    def _push(tok: str) -> None:
        if not tok or tok in seen:
            return
        seen.add(tok)
        out.append(tok)

    for m in _USER_ID_RE.finditer(s):
        _push(m.group(0))
    for m in _EMAIL_RE.finditer(s):
        _push(m.group(0))
    for m in _RESERVATION_ID_RE.finditer(s):
        _push(m.group(0))
    # Order IDs: emit BOTH the bare ("W2378156") and the canonical ("#W2378156")
    # form so arg_grounding accepts either when the action layer normalises
    # to one or the other.  Without this, the user saying "order number
    # W2378156" doesn't ground the agent's "#W2378156" and the call loops.
    for m in _ORDER_ID_LOOSE_RE.finditer(s):
        digits = m.group(1)
        _push(f"W{digits}")
        _push(f"#W{digits}")
    # Capital-prefixed IDs like O123, R4567, B89.
    for m in re.finditer(r"\b[A-Z]\d{2,}\b", s):
        _push(m.group(0))
    # Bare numeric IDs of length >= 4.
    for m in re.finditer(r"\b\d{4,}\b", s):
        _push(m.group(0))
    return out


def _coerce_struct(obs: Any) -> Any:
    """If `obs` is a JSON-serialized string, parse it; else return as-is."""
    if isinstance(obs, str):
        s = obs.strip()
        if not s:
            return s
        if s[0] in "[{":
            try:
                return json.loads(s)
            except Exception:
                return s
        return s
    return obs


def _flatten_scalar_facts(value: Any, prefix: str = "") -> List[str]:
    """Return compact scalar facts from a durable structured cache."""
    out: List[str] = []

    def add(text: Any) -> None:
        s = str(text).strip()
        if s:
            out.append(s[:100])

    if isinstance(value, dict):
        for k, v in value.items():
            kpath = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (str, int, float)) and v not in (None, ""):
                add(f"{kpath}={v}")
                add(v)
            elif isinstance(v, (dict, list)):
                out.extend(_flatten_scalar_facts(v, kpath))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            out.extend(_flatten_scalar_facts(item, f"{prefix}[{i}]"))
    return out


__all__ = ["WorkingMemory"]
