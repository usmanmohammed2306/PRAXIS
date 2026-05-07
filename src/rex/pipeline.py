"""High-level pipeline that ties the procedural-memory architecture together.

Most callers should use this module rather than constructing the pieces
themselves. It composes:

    trajectory_parser → distill → memory_store → retrieval / playbook

into:

  * :func:`promote_records` — distills a batch of records and persists
    them. Used by the runners after a benchmark run completes.
  * :func:`load_corpus_for_domain` — loads seed + runtime cards (deduped).
  * :func:`build_retriever` — wraps :class:`HybridRetriever` with a
    domain-specific corpus.
  * :func:`refresh_playbook` — given the current trajectory, return the
    playbook + retrieval result. Used by the agent loops at every refresh.

Every function takes an optional :class:`RexConfig`; when omitted it
falls back to the module-level default.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .config import RexConfig, default_config
from .distill import distill_record_to_card
from .embeddings import embedder_for_config
from .memory_store import MemoryStore
from .memory_types import (
    ProcessMemoryCard,
    RetrievalQuery,
    RetrievalResult,
    TacticalPlaybook,
    from_experience_card,
)
from .playbook import synthesize_playbook
from .retrieval import HybridRetriever, build_query
from .retrieval_logging import RetrievalLogger
from .working_state import WorkingState, working_state_for_messages


# ---------------------------------------------------------------------------
# Promotion — runs after a benchmark to grow the runtime store
# ---------------------------------------------------------------------------
def promote_records(
    records: Sequence[Dict[str, Any]],
    *,
    domain: str,
    config: Optional[RexConfig] = None,
    runtime_dir: Optional[Path] = None,
    seed_dir: Optional[Path] = None,
    allow_test_split: bool = False,
    forbidden_tokens: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Distill ``records`` into the runtime store for ``domain``.

    Returns a small manifest describing how many cards were promoted /
    rejected and what the runtime size now is.
    """
    import sys
    cfg = config or default_config()
    store = MemoryStore(
        seed_dir=Path(seed_dir) if seed_dir else cfg.bank_dir,
        runtime_dir=Path(runtime_dir) if runtime_dir else cfg.runtime_dir,
        config=cfg,
    )
    promoted: List[ProcessMemoryCard] = []
    rejected: List[Dict[str, Any]] = []
    blocked: List[Dict[str, Any]] = []

    for idx, record in enumerate(records or []):
        if not isinstance(record, dict):
            continue
        if not allow_test_split and _is_test_split_record(record):
            blocked.append({"task_id": record.get("task_id"), "reason": "test_split_blocked"})
            continue
        result = distill_record_to_card(
            record,
            config=cfg,
            forbidden_tokens=forbidden_tokens,
        )
        if result.card is None:
            if result.rejected_reason and result.rejected_reason not in {"ineligible", "no_tool_calls"}:
                rejected.append({
                    "task_id": record.get("task_id"),
                    "reason": result.rejected_reason,
                })
            continue
        _assert_not_raw_card(result.card)
        promoted.append(result.card)
        if cfg.promotion_max_per_run and len(promoted) >= cfg.promotion_max_per_run:
            break

    written = store.bulk_add_cards(promoted, domain=domain)
    store.save_memory_index()
    runtime_count = sum(1 for _ in store.iter_runtime(domain))

    # Log distillation summary
    print(
        f"[distill] domain={domain}  distilled={len(promoted)}  "
        f"written={len(written)}  rejected={len(rejected)}  blocked={len(blocked)}",
        file=sys.stderr,
    )

    return {
        "domain": domain,
        "runtime_dir": str(store.runtime_dir),
        "promoted": len(written),
        "duplicates_skipped": len(promoted) - len(written),
        "rejected": rejected[:25],
        "blocked": blocked[:25],
        "total_runtime_cards_after": runtime_count,
    }


def _is_test_split_record(record: Dict[str, Any]) -> bool:
    info = record.get("info") if isinstance(record, dict) else None
    if isinstance(info, dict):
        split = (info.get("task_split") or "").strip().lower()
        if split == "test":
            return True
    return False


def _assert_not_raw_card(card: ProcessMemoryCard) -> None:
    """Benchmark safety guard: a ProcessMemoryCard must never carry raw
    trajectory content — message lists, observation text, or gold actions.

    This is a defence-in-depth check called before writing to the store.
    If this fires, the distiller produced something unexpected.
    """
    forbidden_fields = ("messages", "gold_actions", "observations", "raw_trajectory")
    for f in forbidden_fields:
        if hasattr(card, f) and getattr(card, f):
            raise ValueError(
                f"SAFETY: ProcessMemoryCard field '{f}' contains raw trajectory data; "
                "only distilled procedural experience may enter the memory bank."
            )


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------
def load_corpus_for_domain(
    domain: str,
    *,
    config: Optional[RexConfig] = None,
    runtime_dir: Optional[Path] = None,
    seed_dir: Optional[Path] = None,
) -> List[ProcessMemoryCard]:
    """Return the merged seed+runtime card list for ``domain``.

    The seed is read from the legacy ``ExperienceCard`` JSONL (lifted to
    :class:`ProcessMemoryCard`); the runtime is read directly. Duplicate
    signatures are dropped.
    """
    cfg = config or default_config()
    seed = _load_seed_corpus(domain, cfg=cfg, override_dir=seed_dir)
    store = MemoryStore(
        seed_dir=Path(seed_dir) if seed_dir else cfg.bank_dir,
        runtime_dir=Path(runtime_dir) if runtime_dir else cfg.runtime_dir,
        config=cfg,
    )
    runtime_cards = store.load_runtime(domain)
    return store.deduplicate_cards(seed + runtime_cards)


def _load_seed_corpus(
    domain: str,
    *,
    cfg: RexConfig,
    override_dir: Optional[Path] = None,
) -> List[ProcessMemoryCard]:
    """Read the seed bank and return :class:`ProcessMemoryCard`-shaped cards.

    The seed file may be in either schema:
      * **v2** ``ProcessMemoryCard`` JSONL (produced by build_experience_memory.py)
      * **Legacy** ``ExperienceCard`` JSONL (lifted automatically)

    If the seed file is absent, returns an empty list — no cards are
    auto-generated from benchmark training data or hardcoded policies.
    REx starts with empty memory on the first run; permanent experience is
    built explicitly via build_experience_memory.py.
    """
    import json as _json
    from .experience import _read_cards_jsonl

    bank_dir = Path(override_dir) if override_dir else cfg.bank_dir
    bank_path = bank_dir / f"{domain}.jsonl"
    if not bank_path.exists():
        return []

    cards: List[ProcessMemoryCard] = []
    with bank_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            if "tool_ordering_hints" in data or "task_category" in data:
                cards.append(ProcessMemoryCard.from_dict(data))
            else:
                # Legacy ExperienceCard schema — lift the whole file and return
                legacy_pmcs = _read_cards_jsonl(bank_path)
                return [from_experience_card(c) for c in legacy_pmcs]
    return cards


# ---------------------------------------------------------------------------
# Retriever construction
# ---------------------------------------------------------------------------
def build_retriever(
    domain: str,
    *,
    config: Optional[RexConfig] = None,
    runtime_dir: Optional[Path] = None,
    seed_dir: Optional[Path] = None,
) -> HybridRetriever:
    """Construct a hybrid retriever pre-fit on the corpus for ``domain``."""
    cfg = config or default_config()
    cards = load_corpus_for_domain(
        domain, config=cfg, runtime_dir=runtime_dir, seed_dir=seed_dir
    )
    embedder = embedder_for_config(cfg, documents=[c.retrieval_text for c in cards])
    return HybridRetriever(cards, config=cfg, embedder=embedder)


# ---------------------------------------------------------------------------
# Runtime refresh — used by the agent loops
# ---------------------------------------------------------------------------
def refresh_playbook(
    *,
    retriever: HybridRetriever,
    initial_user: str,
    messages: Sequence[Dict[str, Any]],
    environment: str,
    benchmark: str = "tau-bench",
    controller: str = "rex",
    step_index: int = 0,
    config: Optional[RexConfig] = None,
    logger: Optional[RetrievalLogger] = None,
    task_id: Any = "",
    trial: int = 0,
) -> Tuple[TacticalPlaybook, RetrievalResult, WorkingState]:
    """Rebuild the working state, retrieve cards, synthesize playbook.

    Returns a tuple of (playbook, retrieval_result, working_state). The
    caller injects ``playbook.body`` into the system prompt.
    """
    cfg = config or default_config()
    state = working_state_for_messages(messages, initial_user=initial_user)
    query = build_query(
        initial_user=initial_user,
        messages=messages,
        environment=environment,
        benchmark=benchmark,
        controller=controller,
        step_index=step_index,
        top_k=cfg.top_k,
    )
    # Inject working-state summary into the query text so failure / retry
    # context biases retrieval toward recovery cards.
    aux = state.summary_for_query()
    if aux:
        query.text = (query.text + " " + aux).strip()
    result = retriever.search(query)
    playbook = synthesize_playbook(result.cards, config=cfg)
    if logger is not None:
        logger.log(
            benchmark=benchmark,
            environment=environment,
            controller=controller,
            task_id=task_id,
            trial=trial,
            step_index=step_index,
            query=query,
            result=result,
            playbook=playbook,
        )
    return playbook, result, state


__all__ = [
    "promote_records",
    "load_corpus_for_domain",
    "build_retriever",
    "refresh_playbook",
]
