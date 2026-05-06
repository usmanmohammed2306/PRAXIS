"""Deterministic PII / ID / benchmark-leak sanitization for procedural memory.

The runtime memory bank is read by every future run, including held-out
test runs. Anything that survives sanitization here will show up in
prompts at evaluation time, so this module is treated as a security
boundary, not a convenience helper.

Rules:

* Every textual field that flows into a card passes through ``sanitize_text``.
* The output is whitespace-normalized so that identical strings dedup
  identically across runs.
* The output never reintroduces PII even if the upstream data did. That is
  enforced by ``contains_unredacted_sensitive`` in the audit step.

Backwards compatibility: ``redact_text`` and ``contains_unredacted_sensitive``
are also exported by ``experience.py``; this module is the single source of
truth and ``experience.py`` re-uses these patterns.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable, List, Mapping, Set, Tuple


# ---------------------------------------------------------------------------
# Sensitive patterns — order matters: more-specific matches go first.
#
# ``SENSITIVE_PATTERNS`` is the redact list (replace with placeholder).
# ``UNREDACTED_PATTERNS`` is the *audit* list — if any of these still match
# after redaction, the audit pass fails the card.
# ---------------------------------------------------------------------------
SENSITIVE_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    # Emails
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "<EMAIL>"),
    # Tau-bench retail order ids, e.g. "#W1234567"
    (re.compile(r"#W\d+\b"), "<ORDER_ID>"),
    # Tau-bench payment ids, e.g. "credit_card_4423124"
    (re.compile(r"\b(?:credit_card|gift_card|paypal|certificate)_\d+\b"), "<PAYMENT_ID>"),
    # Tau-bench user ids, e.g. "noah_brown_5678"
    (re.compile(r"\b[a-z]+_[a-z]+_\d{3,5}\b"), "<USER_ID>"),
    # Tau-bench airline reservation ids — six uppercase alnum w/ at least one
    # letter and at least one digit (e.g. "X1Y2Z3"). The lookaheads guard
    # against matching pure-digit sequences (which the next rule handles).
    (re.compile(r"\b(?=[A-Z0-9]*[A-Z])(?=[A-Z0-9]*\d)[A-Z0-9]{6}\b"), "<RESERVATION_ID>"),
    # Generic 9–10 digit ids (item / product / phone-shaped)
    (re.compile(r"\b\d{9,10}\b"), "<ITEM_OR_PRODUCT_ID>"),
    # Phone-style numbers (loose) — will rarely match in benchmarks but is
    # important for ACE schemas that include a sample contact.
    (re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}\b"), "<PHONE_NUMBER>"),
    # Credit-card-like 13–19 digit groups separated by spaces or dashes.
    (re.compile(r"\b(?:\d{4}[\s-]?){3,4}\d{1,4}\b"), "<CARD_NUMBER>"),
    # SSN-shape (US) 3-2-4
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "<SSN_LIKE>"),
    # Long uppercase alnum strings (likely tokens / API keys >= 24 chars)
    (re.compile(r"\b[A-Z0-9]{24,}\b"), "<TOKEN>"),
]

UNREDACTED_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"#W\d+\b"),
    re.compile(r"\b(?:credit_card|gift_card|paypal|certificate)_\d+\b"),
    re.compile(r"\b[a-z]+_[a-z]+_\d{3,5}\b"),
    re.compile(r"\b\d{9,10}\b"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
]


_WHITESPACE_RE = re.compile(r"\s+")


def sanitize_text(text: Any, *, limit: int = 0) -> str:
    """Redact PII / IDs and normalize whitespace.

    ``limit`` (when > 0) clips the output to that many characters.
    """
    s = str(text or "")
    for pattern, repl in SENSITIVE_PATTERNS:
        s = pattern.sub(repl, s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    if limit and len(s) > limit:
        s = s[: max(0, limit - 3)].rstrip() + "..."
    return s


# Alias kept for backwards compatibility with ``experience.redact_text``.
redact_text = sanitize_text


def contains_unredacted_sensitive(text: Any) -> bool:
    """Return True if ``text`` still contains a pattern the audit forbids."""
    s = str(text or "")
    return any(p.search(s) for p in UNREDACTED_PATTERNS)


def raw_sensitive_tokens(text: Any) -> Set[str]:
    """Return the literal substrings that look sensitive (used to seed
    forbidden-token sets across benchmark splits)."""
    out: Set[str] = set()
    s = str(text or "")
    for pattern, _ in SENSITIVE_PATTERNS:
        for m in pattern.finditer(s):
            tok = m.group(0)
            if tok:
                out.add(tok)
    return out


def sanitize_dict(payload: Mapping[str, Any], *, str_limit: int = 0) -> dict:
    """Recursively sanitize every string leaf in a JSON-shaped payload."""
    return _sanitize_value(payload, str_limit)


def _sanitize_value(value: Any, str_limit: int) -> Any:
    if isinstance(value, str):
        return sanitize_text(value, limit=str_limit)
    if isinstance(value, Mapping):
        return {k: _sanitize_value(v, str_limit) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        out = [_sanitize_value(v, str_limit) for v in value]
        return out if isinstance(value, list) else type(value)(out)
    return value


def compress_repeated_phrases(lines: Iterable[str], *, max_repeats: int = 1) -> List[str]:
    """Deduplicate identical / near-identical lines while preserving order.

    Two lines that differ only by leading/trailing whitespace or surrounding
    punctuation are treated as duplicates. The first occurrence wins.
    """
    seen: dict[str, int] = {}
    out: List[str] = []
    for raw in lines:
        norm = re.sub(r"[^a-z0-9 ]+", " ", str(raw or "").lower())
        norm = _WHITESPACE_RE.sub(" ", norm).strip()
        if not norm:
            continue
        seen[norm] = seen.get(norm, 0) + 1
        if seen[norm] <= max_repeats:
            out.append(str(raw).strip())
    return out


def stable_hash(*parts: str) -> str:
    """Deterministic 12-hex-char hash used for ids that must survive across runs."""
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p or "").encode("utf-8", "ignore"))
        h.update(b"\x1f")
    return h.hexdigest()[:12]


def audit_text(
    text: Any,
    *,
    forbidden_tokens: Iterable[str] = (),
) -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for a redaction audit on ``text``."""
    s = str(text or "")
    if contains_unredacted_sensitive(s):
        return False, "unredacted_sensitive_token"
    for tok in forbidden_tokens or ():
        if tok and tok in s:
            return False, f"forbidden_test_token:{tok}"
    return True, ""


__all__ = [
    "SENSITIVE_PATTERNS",
    "UNREDACTED_PATTERNS",
    "sanitize_text",
    "redact_text",
    "contains_unredacted_sensitive",
    "raw_sensitive_tokens",
    "sanitize_dict",
    "compress_repeated_phrases",
    "stable_hash",
    "audit_text",
]
