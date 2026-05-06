"""Configuration system for the REx-RPE continual procedural-memory architecture.

All runtime parameters are centralized here. Resolution order is:

  1. Constructor argument (when used as a library)
  2. Environment variable (uppercase, ``REX_`` prefixed)
  3. Built-in default

The configuration is *deterministic*: identical environment + identical
arguments yield identical behavior. This is important for benchmark
reproducibility.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_str(key: str, default: str) -> str:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    return raw


def _env_path(key: str, default: Path) -> Path:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    return Path(raw)


@dataclass(frozen=True)
class RexConfig:
    """All tunable runtime parameters for the REx-RPE memory architecture.

    Attributes that affect benchmark safety (test_split blocking, redaction
    aggressiveness) are deliberately stricter than performance parameters by
    default. Tests may relax them, but production runs should not.
    """

    # ------------------------------------------------------------------
    # Filesystem locations
    # ------------------------------------------------------------------
    bank_dir: Path = field(default_factory=lambda: REPO_ROOT / "outputs" / "experience_bank")
    runtime_dir: Path = field(default_factory=lambda: REPO_ROOT / "outputs" / "experience_runtime")
    retrieval_log_dir: Path = field(default_factory=lambda: REPO_ROOT / "outputs" / "retrieval_logs")
    embedding_cache_dir: Path = field(default_factory=lambda: REPO_ROOT / "outputs" / "embedding_cache")

    # ------------------------------------------------------------------
    # Retrieval knobs
    # ------------------------------------------------------------------
    top_k: int = 3
    retrieval_refresh_every: int = 2
    retrieval_score_threshold: float = 0.0  # accept any positive score
    diversity_intent_cap: int = 2
    hybrid_lexical_weight: float = 0.6
    hybrid_embedding_weight: float = 0.4

    # ------------------------------------------------------------------
    # Embedding backend
    # ------------------------------------------------------------------
    embedding_backend: str = "tfidf"  # one of: tfidf | sentence_transformers
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_cache_enabled: bool = True
    embedding_dim: int = 256  # used by hashed TF-IDF

    # ------------------------------------------------------------------
    # Memory promotion / quality
    # ------------------------------------------------------------------
    promotion_min_messages: int = 2
    promotion_min_confidence: float = 0.0  # 0 = accept all eligible
    promotion_max_per_run: int = 64
    promotion_block_test_split: bool = True
    quality_decay_half_life_days: float = 30.0
    quality_min_usage_for_decay: int = 3
    dedup_signature_chars: int = 220
    max_runtime_cards_per_domain: int = 4096

    # ------------------------------------------------------------------
    # Playbook / prompt budgets
    # ------------------------------------------------------------------
    playbook_max_chars: int = 1800
    playbook_max_lines: int = 28
    playbook_section_chars: int = 360
    startup_analysis: bool = True
    startup_analysis_max_words: int = 120
    mutation_reflection: bool = True

    # ------------------------------------------------------------------
    # Benchmark safety
    # ------------------------------------------------------------------
    sanitize_aggressively: bool = True
    forbidden_token_max_per_card: int = 0  # zero tolerance after audit

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    retrieval_log_enabled: bool = True
    retrieval_log_max_records: int = 5000

    @classmethod
    def from_env(cls, **overrides: Any) -> "RexConfig":
        """Construct a config by reading every supported env var.

        ``overrides`` take precedence over env vars and built-in defaults.
        """
        defaults = cls()
        values: Dict[str, Any] = {
            # Paths
            "bank_dir": _env_path("REX_EXPERIENCE_DIR", defaults.bank_dir),
            "runtime_dir": _env_path("REX_RUNTIME_DIR", defaults.runtime_dir),
            "retrieval_log_dir": _env_path(
                "REX_RETRIEVAL_LOG_DIR", defaults.retrieval_log_dir
            ),
            "embedding_cache_dir": _env_path(
                "REX_EMBEDDING_CACHE_DIR", defaults.embedding_cache_dir
            ),
            # Retrieval
            "top_k": _env_int("REX_TOP_K", defaults.top_k),
            "retrieval_refresh_every": _env_int(
                "REX_RETRIEVAL_REFRESH_EVERY", defaults.retrieval_refresh_every
            ),
            "retrieval_score_threshold": _env_float(
                "REX_RETRIEVAL_SCORE_THRESHOLD", defaults.retrieval_score_threshold
            ),
            "diversity_intent_cap": _env_int(
                "REX_DIVERSITY_INTENT_CAP", defaults.diversity_intent_cap
            ),
            "hybrid_lexical_weight": _env_float(
                "REX_HYBRID_LEXICAL_WEIGHT", defaults.hybrid_lexical_weight
            ),
            "hybrid_embedding_weight": _env_float(
                "REX_HYBRID_EMBEDDING_WEIGHT", defaults.hybrid_embedding_weight
            ),
            # Embeddings
            "embedding_backend": _env_str(
                "REX_EMBEDDING_BACKEND", defaults.embedding_backend
            ).lower(),
            "embedding_model": _env_str(
                "REX_EMBEDDING_MODEL", defaults.embedding_model
            ),
            "embedding_cache_enabled": _env_bool(
                "REX_EMBEDDING_CACHE_ENABLED", defaults.embedding_cache_enabled
            ),
            "embedding_dim": _env_int(
                "REX_EMBEDDING_DIM", defaults.embedding_dim
            ),
            # Promotion / quality
            "promotion_min_messages": _env_int(
                "REX_PROMOTION_MIN_MESSAGES", defaults.promotion_min_messages
            ),
            "promotion_min_confidence": _env_float(
                "REX_PROMOTION_MIN_CONFIDENCE", defaults.promotion_min_confidence
            ),
            "promotion_max_per_run": _env_int(
                "REX_PROMOTION_MAX_PER_RUN", defaults.promotion_max_per_run
            ),
            "promotion_block_test_split": _env_bool(
                "REX_PROMOTION_BLOCK_TEST_SPLIT", defaults.promotion_block_test_split
            ),
            "quality_decay_half_life_days": _env_float(
                "REX_QUALITY_DECAY_HALF_LIFE_DAYS",
                defaults.quality_decay_half_life_days,
            ),
            "quality_min_usage_for_decay": _env_int(
                "REX_QUALITY_MIN_USAGE_FOR_DECAY",
                defaults.quality_min_usage_for_decay,
            ),
            "dedup_signature_chars": _env_int(
                "REX_DEDUP_SIGNATURE_CHARS", defaults.dedup_signature_chars
            ),
            "max_runtime_cards_per_domain": _env_int(
                "REX_MAX_RUNTIME_CARDS_PER_DOMAIN",
                defaults.max_runtime_cards_per_domain,
            ),
            # Playbook
            "playbook_max_chars": _env_int(
                "REX_PLAYBOOK_MAX_CHARS", defaults.playbook_max_chars
            ),
            "playbook_max_lines": _env_int(
                "REX_PLAYBOOK_MAX_LINES", defaults.playbook_max_lines
            ),
            "playbook_section_chars": _env_int(
                "REX_PLAYBOOK_SECTION_CHARS", defaults.playbook_section_chars
            ),
            "startup_analysis": _env_bool(
                "REX_STARTUP_ANALYSIS", defaults.startup_analysis
            ),
            "startup_analysis_max_words": _env_int(
                "REX_STARTUP_ANALYSIS_MAX_WORDS",
                defaults.startup_analysis_max_words,
            ),
            "mutation_reflection": _env_bool(
                "REX_MUTATION_REFLECTION", defaults.mutation_reflection
            ),
            # Benchmark safety
            "sanitize_aggressively": _env_bool(
                "REX_SANITIZE_AGGRESSIVELY", defaults.sanitize_aggressively
            ),
            "forbidden_token_max_per_card": _env_int(
                "REX_FORBIDDEN_TOKEN_MAX_PER_CARD",
                defaults.forbidden_token_max_per_card,
            ),
            # Diagnostics
            "retrieval_log_enabled": _env_bool(
                "REX_RETRIEVAL_LOG_ENABLED", defaults.retrieval_log_enabled
            ),
            "retrieval_log_max_records": _env_int(
                "REX_RETRIEVAL_LOG_MAX_RECORDS",
                defaults.retrieval_log_max_records,
            ),
        }
        values.update(overrides)
        # Filter out keys that aren't dataclass fields (defensive against typos).
        valid = {f.name for f in fields(cls)}
        values = {k: v for k, v in values.items() if k in valid}
        return cls(**values)

    def with_overrides(self, **kwargs: Any) -> "RexConfig":
        """Return a copy of this config with ``kwargs`` overridden."""
        merged = self.to_dict()
        merged.update({k: v for k, v in kwargs.items() if k in merged})
        return self.__class__(**merged)

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self.__class__)}


# Module-level singleton — used by code paths that don't take an explicit
# config arg. Tests that need a different config should construct a new
# RexConfig and pass it explicitly.
_DEFAULT_CONFIG: Optional[RexConfig] = None


def default_config() -> RexConfig:
    """Return the lazily-initialized module-level default config."""
    global _DEFAULT_CONFIG
    if _DEFAULT_CONFIG is None:
        _DEFAULT_CONFIG = RexConfig.from_env()
    return _DEFAULT_CONFIG


def reset_default_config() -> None:
    """Drop the cached default config so the next call to ``default_config`` re-reads env."""
    global _DEFAULT_CONFIG
    _DEFAULT_CONFIG = None


__all__ = [
    "RexConfig",
    "default_config",
    "reset_default_config",
]
