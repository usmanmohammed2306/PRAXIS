"""Quality control for the procedural-memory bank.

This module is the consolidation/aging layer. It does *not* run on the
hot path — it is invoked periodically by the runners (after each run) to
keep the memory bank healthy.

Responsibilities:

  * Detect duplicates by content signature.
  * Detect contradictions (cards with the same task category but
    contradictory advice — e.g. one says "verify before mutation" and
    another says "skip verification"). Contradictory cards have their
    confidence damped equally so the retriever weighs them down.
  * Apply age decay so old, rarely-used cards drop in score.
  * Recompute ``ProcessMemoryCard.confidence`` based on usage/success.
  * Provide ``consolidation_report`` for diagnostics.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .config import RexConfig, default_config
from .memory_types import ProcessMemoryCard


# ---------------------------------------------------------------------------
# Quality computation
# ---------------------------------------------------------------------------
def _normalize_phrase(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).strip()


def _bag_of_words(text: str) -> Set[str]:
    norm = _normalize_phrase(text)
    return {t for t in norm.split() if len(t) > 2}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def detect_duplicates(
    cards: Sequence[ProcessMemoryCard],
    *,
    config: Optional[RexConfig] = None,
) -> List[Tuple[int, int, float]]:
    """Return triples ``(i, j, similarity)`` for likely-duplicate card pairs.

    ``similarity`` is Jaccard over the union of ``successful_patterns`` +
    ``failure_patterns`` + ``recommended_next_tools``. We only emit pairs
    with similarity >= 0.85 to keep the report short.
    """
    cfg = config or default_config()  # noqa: F841 — kept for symmetry; reserved for future tuning
    sigs: List[Set[str]] = []
    for c in cards:
        text = " ".join(c.successful_patterns + c.failure_patterns + c.recommended_next_tools)
        sigs.append(_bag_of_words(text))
    out: List[Tuple[int, int, float]] = []
    for i in range(len(cards)):
        for j in range(i + 1, len(cards)):
            sim = jaccard(sigs[i], sigs[j])
            if sim >= 0.85:
                out.append((i, j, sim))
    return out


def detect_contradictions(cards: Sequence[ProcessMemoryCard]) -> List[Tuple[int, int]]:
    """Pairs of cards with the same ``task_category`` and contradictory hints.

    Contradiction heuristic: same env + same intent (category prefix
    before ``__``) but different ``confirmation`` polarity (one says
    "ask the user", the other says "no confirmation needed").
    """
    out: List[Tuple[int, int]] = []
    by_intent: Dict[str, List[int]] = {}
    for i, c in enumerate(cards):
        intent = (c.task_category or "").split("__", 1)[0]
        by_intent.setdefault(f"{c.environment}|{intent}", []).append(i)
    for indices in by_intent.values():
        for i in indices:
            for j in indices:
                if j <= i:
                    continue
                a = cards[i]
                b = cards[j]
                if _polar_disagreement(a.confirmation, b.confirmation):
                    out.append((i, j))
    return out


def _polar_disagreement(a: str, b: str) -> bool:
    al = (a or "").lower()
    bl = (b or "").lower()
    a_ask = "ask" in al or "confirm" in al or "stop" in al
    b_ask = "ask" in bl or "confirm" in bl or "stop" in bl
    a_skip = "no" in al and ("confirm" in al or "ask" in al)
    b_skip = "no" in bl and ("confirm" in bl or "ask" in bl)
    return (a_ask and b_skip) or (a_skip and b_ask)


def apply_age_decay(
    cards: Sequence[ProcessMemoryCard],
    *,
    now_ms: Optional[int] = None,
    config: Optional[RexConfig] = None,
) -> int:
    """Decay confidence for old, rarely-used cards.

    Returns the number of cards whose confidence was modified.
    """
    cfg = config or default_config()
    if not cards or cfg.quality_decay_half_life_days <= 0:
        return 0
    now = int(now_ms if now_ms is not None else time.time() * 1000)
    half_life_ms = cfg.quality_decay_half_life_days * 24 * 3600 * 1000
    modified = 0
    for c in cards:
        if c.usage_count >= cfg.quality_min_usage_for_decay:
            # Frequently-used cards are exempt from decay.
            continue
        age_ms = max(0, now - int(c.timestamp_ms))
        if age_ms <= 0:
            continue
        factor = math.pow(0.5, age_ms / half_life_ms) if half_life_ms > 0 else 1.0
        new_conf = max(0.0, min(1.0, c.confidence * factor))
        if abs(new_conf - c.confidence) > 1e-6:
            c.confidence = new_conf
            modified += 1
    return modified


def damp_contradictions(
    cards: Sequence[ProcessMemoryCard],
    contradictions: Sequence[Tuple[int, int]],
    *,
    factor: float = 0.7,
) -> int:
    """Lower confidence on each card that participates in a contradiction."""
    if not contradictions:
        return 0
    touched: Set[int] = set()
    for i, j in contradictions:
        touched.add(i)
        touched.add(j)
    modified = 0
    for idx in touched:
        if 0 <= idx < len(cards):
            c = cards[idx]
            new_conf = max(0.0, min(1.0, c.confidence * factor))
            if abs(new_conf - c.confidence) > 1e-6:
                c.confidence = new_conf
                modified += 1
    return modified


def recompute_confidence_from_usage(cards: Sequence[ProcessMemoryCard]) -> int:
    """Refresh ``confidence`` from observed usage outcomes.

    This is conservative: cards with no usage data keep their original
    confidence; cards with usage have their confidence pulled toward
    success_rate proportional to ``min(usage_count, 10) / 10``.
    """
    modified = 0
    for c in cards:
        if c.usage_count <= 0:
            continue
        success_rate = c.success_count / c.usage_count
        weight = min(1.0, c.usage_count / 10.0)
        new_conf = (1.0 - weight) * c.confidence + weight * success_rate
        new_conf = max(0.0, min(1.0, new_conf))
        if abs(new_conf - c.confidence) > 1e-6:
            c.confidence = new_conf
            modified += 1
    return modified


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
@dataclass
class ConsolidationReport:
    duplicates: List[Tuple[str, str, float]] = field(default_factory=list)
    contradictions: List[Tuple[str, str]] = field(default_factory=list)
    decayed: int = 0
    contradiction_damped: int = 0
    confidence_recomputed: int = 0
    total_cards: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_cards": self.total_cards,
            "duplicates": [{"a": a, "b": b, "similarity": sim} for (a, b, sim) in self.duplicates],
            "contradictions": [{"a": a, "b": b} for (a, b) in self.contradictions],
            "decayed": self.decayed,
            "contradiction_damped": self.contradiction_damped,
            "confidence_recomputed": self.confidence_recomputed,
        }


def consolidate(
    cards: Sequence[ProcessMemoryCard],
    *,
    config: Optional[RexConfig] = None,
) -> ConsolidationReport:
    """Run a full quality pass over ``cards`` and return a report.

    Mutates the cards in-place (confidence updates) but does not modify
    storage. Callers persist the updated cards back to disk if desired.
    """
    cfg = config or default_config()
    duplicates = detect_duplicates(cards, config=cfg)
    contradictions = detect_contradictions(cards)
    decayed = apply_age_decay(cards, config=cfg)
    damped = damp_contradictions(cards, contradictions)
    refreshed = recompute_confidence_from_usage(cards)
    return ConsolidationReport(
        duplicates=[(cards[i].card_id, cards[j].card_id, sim) for (i, j, sim) in duplicates],
        contradictions=[(cards[i].card_id, cards[j].card_id) for (i, j) in contradictions],
        decayed=decayed,
        contradiction_damped=damped,
        confidence_recomputed=refreshed,
        total_cards=len(cards),
    )


__all__ = [
    "ConsolidationReport",
    "consolidate",
    "detect_duplicates",
    "detect_contradictions",
    "apply_age_decay",
    "damp_contradictions",
    "recompute_confidence_from_usage",
    "jaccard",
]
