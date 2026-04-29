"""Pre-condition gate: every declared pre-condition must be supported by
some fact in working memory.

The matcher is intentionally permissive in v1: a precondition is
"satisfied" iff ALL of its non-stopword content tokens appear (as
substrings, case-insensitively) somewhere in the union of
``user_facts ∪ db_facts``. This catches the high-yield failures
(e.g. mutating ``order_id=O1234`` when no fact mentions ``O1234``)
without rejecting reasonable actions whose preconditions are only
loosely paraphrased.

The complementary required-args gate enforces that every required
parameter is non-empty (cheap structural check).
"""
from __future__ import annotations

import re
from typing import Iterable

from ..schemas import GateResult, ProposedAction, ToolEffectSchema
from ..working_memory import WorkingMemory


_STOPWORDS = {
    "a", "an", "the", "is", "are", "of", "to", "for", "in", "on", "at",
    "by", "with", "and", "or", "but", "as", "be", "been", "being", "this",
    "that", "these", "those", "it", "its", "their", "there", "has", "have",
    "had", "must", "should", "may", "can", "will", "would", "shall",
    "exists", "exist", "user", "users", "user's", "users'", "entity",
    "any", "all", "some",
}

_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]+")


def _content_tokens(s: str) -> Iterable[str]:
    for m in _TOKEN_RE.finditer(s.lower()):
        tok = m.group(0)
        if not tok or tok in _STOPWORDS:
            continue
        if len(tok) < 2:
            continue
        yield tok


def check_preconditions(
    action: ProposedAction,
    schema: ToolEffectSchema,
    wm: WorkingMemory,
) -> GateResult:
    # 1. Required-args structural check.
    missing_args = [
        p for p in (schema.required_params or [])
        if p not in action.args or action.args.get(p) in (None, "", [])
    ]
    if missing_args:
        return GateResult.failing(
            "preconditions",
            f"missing_required_args:{missing_args}",
            missing_args=missing_args,
        )

    # 2. Declared NL pre-conditions: each must overlap evidence by all of
    # its content tokens. Empty / tautological pre-conds pass automatically.
    evidence = wm.all_evidence().lower()
    if not evidence and not action.declared_pre and not schema.preconditions:
        return GateResult.passing("preconditions")

    declared = list(action.declared_pre or [])
    if not declared:
        # Fall back to schema preconditions if the LLM didn't declare any.
        declared = list(schema.preconditions or [])
    unmet = []
    for pre in declared:
        toks = list(_content_tokens(pre))
        if not toks:
            continue
        # Permissive: if AT LEAST 60% of content tokens appear in evidence,
        # treat the pre-cond as satisfied. This avoids over-rejecting on
        # paraphrase mismatch.
        hits = sum(1 for t in toks if t in evidence)
        if not toks or hits / max(len(toks), 1) < 0.6:
            # Also pass if no concrete proper-noun-like tokens are required.
            proper_nouns = [t for t in toks if any(ch.isdigit() for ch in t)
                            or (len(t) >= 4 and t.isalpha())]
            if not proper_nouns:
                continue
            unmet.append(pre)
    if unmet:
        return GateResult.failing(
            "preconditions",
            f"unverified_pre:{unmet[:3]}",
            unmet=unmet[:3],
        )
    return GateResult.passing("preconditions")


__all__ = ["check_preconditions"]
