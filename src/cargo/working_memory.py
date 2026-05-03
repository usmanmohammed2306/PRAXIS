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
    # Set once we've already emitted a "give up on auth" FINAL.  Prevents the
    # exact same FINAL from being emitted twice in a row.
    auth_giveup_emitted: bool = False
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
        if clean not in self.user_facts:
            self.user_facts.append(clean)
        # Extract atomic candidate values (proper nouns, IDs, emails) as
        # individual fact lines so arg-grounding has fine-grained anchors.
        for tok in _extract_id_tokens(clean):
            if tok not in self.user_facts:
                self.user_facts.append(tok)

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
                        self._add_db_fact(f"{kpath}[{i}]={item}")

    def _add_db_fact(self, fact: str) -> None:
        if not fact:
            return
        fact = fact[:100]  # tip 5: short canonical form
        if fact in self.db_facts:
            return
        self.db_facts.append(fact)
        # Hard cap: keep only the 48 most-recent facts.  Older facts are the
        # ones least likely to be needed for the current gate or proposer call.
        if len(self.db_facts) > 48:
            self.db_facts = self.db_facts[-48:]

    def record_action_signature(self, sig: str) -> None:
        if sig:
            self.recent_signatures.append(sig)

    # ------------------------------------------------------------------
    # Queries used by gates
    # ------------------------------------------------------------------
    def all_evidence(self) -> str:
        """Concatenated evidence string used for substring grounding."""
        return "\n".join(self.user_facts + self.db_facts)

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


__all__ = ["WorkingMemory"]
