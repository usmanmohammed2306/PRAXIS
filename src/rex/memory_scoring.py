"""Dynamic memory usefulness scoring with runtime reinforcement.

Memory cards gain or lose confidence based on:
  * successful retrieval outcomes
  * successful recoveries
  * failed recommendations
  * retrieval usefulness
  * runtime success after retrieval
  * repeated operational utility

The scoring system provides:
  1. Usefulness scoring
  2. Runtime reinforcement
  3. Confidence decay
  4. Retrieval impact tracking
  5. Memory ranking evolution
  6. Operational utility scoring
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time


@dataclass
class CardScore:
    """Score metrics for a single memory card."""

    card_id: str
    confidence: float = 1.0  # [0, 1] — current trust level
    usage_count: int = 0  # How many times retrieved
    success_count: int = 0  # How many times led to success
    failure_count: int = 0  # How many times led to failure
    retrieval_last_ms: int = 0  # Last time retrieved (wall-clock ms)
    success_last_ms: int = 0  # Last time led to success

    def success_rate(self) -> float:
        """Success rate when card has been used."""
        total = self.usage_count
        if total == 0:
            return 0.0
        return self.success_count / total

    def recency_boost(self, current_ms: int, decay_half_life_ms: int = 3600000) -> float:
        """Compute recency boost (1.0 for fresh, decays toward 0).

        Args:
            current_ms: Current time in milliseconds
            decay_half_life_ms: Half-life for decay (default 1 hour)

        Returns:
            Recency boost in [0, 1]
        """
        if self.retrieval_last_ms == 0:
            return 0.5  # Neutral boost for never-used

        age_ms = max(0, current_ms - self.retrieval_last_ms)
        # Half-life decay: e^(-age / half_life)
        import math
        return max(0.1, math.exp(-(age_ms / decay_half_life_ms)))


@dataclass
class MemoryScoreSnapshot:
    """Snapshot of all card scores at a point in time."""

    timestamp_ms: int
    scores: Dict[str, CardScore] = field(default_factory=dict)

    def get_ranked_cards(self) -> List[Tuple[str, float]]:
        """Get card IDs ranked by current score (highest first)."""
        ranked = [
            (card_id, score.confidence)
            for card_id, score in self.scores.items()
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked


class MemoryScoreManager:
    """Manages dynamic scoring of memory cards across time."""

    def __init__(self):
        """Initialize score manager."""
        self.scores: Dict[str, CardScore] = {}
        self.history: List[MemoryScoreSnapshot] = []
        self._last_snapshot_ms = 0

    def initialize_card(self, card_id: str, initial_confidence: float = 1.0) -> None:
        """Register a card with initial confidence."""
        if card_id not in self.scores:
            self.scores[card_id] = CardScore(
                card_id=card_id,
                confidence=initial_confidence,
                usage_count=0,
                success_count=0,
                failure_count=0,
            )

    def record_retrieval(self, card_id: str) -> None:
        """Record that a card was retrieved."""
        if card_id not in self.scores:
            self.initialize_card(card_id)

        card = self.scores[card_id]
        card.usage_count += 1
        card.retrieval_last_ms = int(time.time() * 1000)

    def record_success(
        self,
        card_id: str,
        success_magnitude: float = 1.0,
    ) -> None:
        """Record that card led to success, update confidence.

        Args:
            card_id: Card ID
            success_magnitude: How strong the success was [0, 1]
        """
        if card_id not in self.scores:
            self.initialize_card(card_id)

        card = self.scores[card_id]
        card.success_count += 1
        card.success_last_ms = int(time.time() * 1000)

        # Boost confidence on success
        success_boost = 0.05 * success_magnitude  # 0-5% boost
        card.confidence = min(1.0, card.confidence + success_boost)

    def record_failure(
        self,
        card_id: str,
        failure_severity: float = 1.0,
    ) -> None:
        """Record that card led to failure, reduce confidence.

        Args:
            card_id: Card ID
            failure_severity: How severe the failure was [0, 1]
        """
        if card_id not in self.scores:
            self.initialize_card(card_id)

        card = self.scores[card_id]
        card.failure_count += 1

        # Penalize confidence on failure
        failure_penalty = 0.1 * failure_severity  # 0-10% penalty
        card.confidence = max(0.0, card.confidence - failure_penalty)

    def apply_age_decay(
        self,
        decay_half_life_ms: int = 3600000,  # 1 hour
        min_confidence: float = 0.2,
    ) -> None:
        """Apply time-based decay to card confidences.

        Older cards gradually lose confidence unless reinforced.

        Args:
            decay_half_life_ms: Decay half-life in milliseconds
            min_confidence: Floor for confidence (won't decay below this)
        """
        current_ms = int(time.time() * 1000)

        for card in self.scores.values():
            if card.retrieval_last_ms == 0:
                # Never used cards decay quickly
                card.confidence *= 0.95  # 5% decay per call
            else:
                age_ms = current_ms - card.retrieval_last_ms
                import math
                decay_factor = math.exp(-(age_ms / decay_half_life_ms))
                card.confidence = max(
                    min_confidence,
                    card.confidence * decay_factor
                )

    def compute_operational_utility(
        self,
        card_id: str,
        phase_match: bool = False,
        failure_pattern_match: bool = False,
        recovery_heuristic_match: bool = False,
        tool_transition_match: bool = False,
    ) -> float:
        """Compute operational utility score for a card.

        Considers structural alignment with current operational state.

        Args:
            card_id: Card ID
            phase_match: Whether card matches current phase
            failure_pattern_match: Whether card covers current failure pattern
            recovery_heuristic_match: Whether card has recovery guidance
            tool_transition_match: Whether card matches tool transitions

        Returns:
            Utility score [0, 1]
        """
        score = 0.0

        # Base score from confidence
        if card_id in self.scores:
            score += self.scores[card_id].confidence * 0.4

        # Operational alignment bonuses
        if phase_match:
            score += 0.2
        if failure_pattern_match:
            score += 0.15
        if recovery_heuristic_match:
            score += 0.15
        if tool_transition_match:
            score += 0.1

        return min(1.0, score)

    def rank_by_confidence(
        self,
        card_ids: List[str],
        current_ms: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Rank cards by confidence (with optional recency boost).

        Args:
            card_ids: List of card IDs to rank
            current_ms: Current time for recency calc (defaults to now)

        Returns:
            List of (card_id, score) tuples, sorted by score descending
        """
        if current_ms is None:
            current_ms = int(time.time() * 1000)

        ranked = []
        for card_id in card_ids:
            if card_id in self.scores:
                card = self.scores[card_id]
                score = card.confidence * card.recency_boost(current_ms)
                ranked.append((card_id, score))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def snapshot(self) -> MemoryScoreSnapshot:
        """Create a snapshot of current scores for historical tracking."""
        snapshot = MemoryScoreSnapshot(
            timestamp_ms=int(time.time() * 1000),
            scores={
                card_id: CardScore(
                    card_id=score.card_id,
                    confidence=score.confidence,
                    usage_count=score.usage_count,
                    success_count=score.success_count,
                    failure_count=score.failure_count,
                    retrieval_last_ms=score.retrieval_last_ms,
                    success_last_ms=score.success_last_ms,
                )
                for card_id, score in self.scores.items()
            }
        )
        self.history.append(snapshot)
        return snapshot

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall memory scoring statistics."""
        if not self.scores:
            return {
                "total_cards": 0,
                "avg_confidence": 0.0,
                "avg_success_rate": 0.0,
                "cards_high_confidence": 0,
                "cards_medium_confidence": 0,
                "cards_low_confidence": 0,
            }

        scores = list(self.scores.values())
        confidences = [s.confidence for s in scores]
        success_rates = [s.success_rate() for s in scores if s.usage_count > 0]

        high_conf = sum(1 for s in scores if s.confidence > 0.7)
        med_conf = sum(1 for s in scores if 0.3 <= s.confidence <= 0.7)
        low_conf = sum(1 for s in scores if s.confidence < 0.3)

        return {
            "total_cards": len(scores),
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
            "avg_success_rate": sum(success_rates) / len(success_rates) if success_rates else 0.0,
            "total_retrievals": sum(s.usage_count for s in scores),
            "total_successes": sum(s.success_count for s in scores),
            "total_failures": sum(s.failure_count for s in scores),
            "cards_high_confidence": high_conf,
            "cards_medium_confidence": med_conf,
            "cards_low_confidence": low_conf,
        }


__all__ = [
    "MemoryScoreManager",
    "CardScore",
    "MemoryScoreSnapshot",
]
