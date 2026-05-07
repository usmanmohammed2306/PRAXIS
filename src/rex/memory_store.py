"""Filesystem-backed, append-only persistent procedural memory store.

The store is intentionally simple:

  * One JSONL file per (memory_kind, domain), e.g.
    ``runtime/retail.jsonl`` and ``seed/retail.jsonl``.
  * Append-only writes, atomic via OS-level ``fcntl`` locking on POSIX.
  * Deduplication by content signature (see ``ProcessMemoryCard.signature``).
  * Cap on max cards per domain to keep retrieval fast (oldest cards
    pruned when over budget; uses ``quality_score`` as the keep heuristic).
  * Index file (``index.json``) summarizing card counts and last update.

It does *not* depend on a database. ``$REX_RUNTIME_DIR`` is the only
state that matters across runs.
"""
from __future__ import annotations

import errno
import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from .config import RexConfig, default_config
from .memory_types import ProcessMemoryCard


# ---------------------------------------------------------------------------
# Cross-process file lock
# ---------------------------------------------------------------------------
@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    """Acquire an OS-level exclusive lock on ``path.lock``.

    Used to make append + dedup safe across concurrent runners. Uses
    ``fcntl.flock`` on POSIX; falls back to a best-effort lock-file on
    platforms without ``fcntl`` (e.g. Windows).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    fh = open(lock_path, "a+")
    try:
        try:
            import fcntl  # type: ignore[import-not-found]

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except ImportError:
            # Best-effort cooperative lock for non-POSIX hosts.
            for _ in range(50):
                try:
                    os.rename(lock_path, lock_path)  # noop, but raises if held
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EBUSY):
                        break
                    time.sleep(0.05)
            yield
    finally:
        fh.close()


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------
class MemoryStore:
    """Persistent procedural memory store.

    Two memory kinds are recognized:

      * ``"seed"`` — built once per checkout from allowed support data
      * ``"runtime"`` — accumulated across runs by the distiller

    Most callers want :meth:`load_all` which returns ``seed + runtime``
    deduplicated by signature.
    """

    SEED_KIND = "seed"
    RUNTIME_KIND = "runtime"

    def __init__(
        self,
        *,
        seed_dir: Path,
        runtime_dir: Path,
        config: Optional[RexConfig] = None,
    ) -> None:
        self.config = config or default_config()
        self.seed_dir = Path(seed_dir)
        self.runtime_dir = Path(runtime_dir)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def path_for(self, kind: str, domain: str) -> Path:
        if kind == self.SEED_KIND:
            return self.seed_dir / f"{domain}.jsonl"
        if kind == self.RUNTIME_KIND:
            return self.runtime_dir / f"{domain}.jsonl"
        raise ValueError(f"Unknown memory kind: {kind!r}")

    def index_path(self) -> Path:
        return self.runtime_dir / "index.json"

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def _read_cards(self, path: Path) -> List[ProcessMemoryCard]:
        cards: List[ProcessMemoryCard] = []
        if not path.exists():
            return cards
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(data, dict):
                    continue
                # If the file is in the legacy ExperienceCard schema, lift it.
                if _is_legacy_experience_card(data):
                    pmc = _lift_legacy_dict(data)
                else:
                    pmc = ProcessMemoryCard.from_dict(data)
                cards.append(pmc)
        return cards

    def load_seed(self, domain: str) -> List[ProcessMemoryCard]:
        return self._read_cards(self.path_for(self.SEED_KIND, domain))

    def load_runtime(self, domain: str) -> List[ProcessMemoryCard]:
        return self._read_cards(self.path_for(self.RUNTIME_KIND, domain))

    def load_all(self, domain: str) -> List[ProcessMemoryCard]:
        """Seed + runtime, deduplicated by content signature."""
        return self.deduplicate_cards(
            self.load_seed(domain) + self.load_runtime(domain)
        )

    def iter_runtime(self, domain: str) -> Iterator[ProcessMemoryCard]:
        """Streaming variant of :meth:`load_runtime` for large stores."""
        path = self.path_for(self.RUNTIME_KIND, domain)
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(data, dict):
                    continue
                if _is_legacy_experience_card(data):
                    yield _lift_legacy_dict(data)
                else:
                    yield ProcessMemoryCard.from_dict(data)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def add_card(self, card: ProcessMemoryCard, *, domain: Optional[str] = None) -> bool:
        return bool(self.bulk_add_cards([card], domain=domain or card.environment))

    def bulk_add_cards(
        self,
        cards: Sequence[ProcessMemoryCard],
        *,
        domain: str,
    ) -> List[ProcessMemoryCard]:
        """Append new (deduplicated) cards to the runtime store for ``domain``.

        Only distilled :class:`ProcessMemoryCard` objects are accepted.
        Raw trajectory data (messages, observations, gold actions) must never
        be written to the runtime store — it is a permanent memory bank, not
        a replay buffer.  The distillation pipeline (``pipeline.promote_records``)
        is the only sanctioned writer.

        Returns the list of cards actually written.
        """
        if not cards:
            return []
        path = self.path_for(self.RUNTIME_KIND, domain)
        with _exclusive_lock(path):
            existing = self._read_cards(path)
            existing_sigs = {
                c.signature(self.config.dedup_signature_chars) for c in existing
            }
            written: List[ProcessMemoryCard] = []
            with path.open("a", encoding="utf-8") as f:
                for card in cards:
                    sig = card.signature(self.config.dedup_signature_chars)
                    if sig in existing_sigs:
                        continue
                    existing_sigs.add(sig)
                    f.write(json.dumps(card.to_dict(), ensure_ascii=False, sort_keys=True, default=str) + "\n")
                    written.append(card)
            self._maybe_prune(domain, existing + written)
            self._update_index(domain, after_added=len(written))
        return written

    def deduplicate_cards(self, cards: Sequence[ProcessMemoryCard]) -> List[ProcessMemoryCard]:
        """Drop later cards whose signature matches an earlier one."""
        seen: set[str] = set()
        out: List[ProcessMemoryCard] = []
        for c in cards:
            sig = c.signature(self.config.dedup_signature_chars)
            if sig in seen:
                continue
            seen.add(sig)
            out.append(c)
        return out

    def memory_pruning(self, domain: str) -> int:
        """Force a prune pass; returns the number of cards removed."""
        path = self.path_for(self.RUNTIME_KIND, domain)
        with _exclusive_lock(path):
            cards = self._read_cards(path)
            cap = max(0, int(self.config.max_runtime_cards_per_domain))
            if not cap or len(cards) <= cap:
                return 0
            ranked = sorted(cards, key=lambda c: (c.quality_score(), c.timestamp_ms), reverse=True)
            kept = ranked[:cap]
            removed = len(cards) - len(kept)
            tmp = path.with_suffix(path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for c in kept:
                    f.write(json.dumps(c.to_dict(), ensure_ascii=False, sort_keys=True, default=str) + "\n")
            tmp.replace(path)
            self._update_index(domain, after_added=0, after_pruned=removed)
        return removed

    def _maybe_prune(self, domain: str, cards: Sequence[ProcessMemoryCard]) -> None:
        """Internal: prune to cap if exceeded. Caller already holds the lock."""
        cap = max(0, int(self.config.max_runtime_cards_per_domain))
        if not cap or len(cards) <= cap:
            return
        path = self.path_for(self.RUNTIME_KIND, domain)
        ranked = sorted(cards, key=lambda c: (c.quality_score(), c.timestamp_ms), reverse=True)
        kept = ranked[:cap]
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for c in kept:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False, sort_keys=True, default=str) + "\n")
        tmp.replace(path)

    # ------------------------------------------------------------------
    # Statistics & index
    # ------------------------------------------------------------------
    def memory_statistics(self) -> Dict[str, Any]:
        """Return a snapshot of card counts per domain across both stores."""
        out: Dict[str, Any] = {
            "seed_dir": str(self.seed_dir),
            "runtime_dir": str(self.runtime_dir),
            "by_domain": {},
        }
        domains = self._known_domains()
        for d in sorted(domains):
            seed = self.load_seed(d)
            runtime = self.load_runtime(d)
            combined = self.deduplicate_cards(seed + runtime)
            out["by_domain"][d] = {
                "seed": len(seed),
                "runtime": len(runtime),
                "combined_dedup": len(combined),
            }
        return out

    def retrieval_statistics(self) -> Dict[str, Any]:
        """Aggregate usage / success counts (read-only)."""
        out: Dict[str, Any] = {"by_domain": {}}
        for d in sorted(self._known_domains()):
            cards = self.load_all(d)
            out["by_domain"][d] = {
                "total_cards": len(cards),
                "total_uses": sum(c.usage_count for c in cards),
                "total_successes": sum(c.success_count for c in cards),
                "total_failures": sum(c.failure_count for c in cards),
                "avg_confidence": (
                    sum(c.confidence for c in cards) / len(cards) if cards else 0.0
                ),
            }
        return out

    def save_memory_index(self) -> Path:
        """Write the index file and return its path."""
        path = self.index_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        snapshot = {
            "updated_ms": int(time.time() * 1000),
            "memory_statistics": self.memory_statistics(),
            "retrieval_statistics": self.retrieval_statistics(),
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2, default=str)
        tmp.replace(path)
        return path

    def _update_index(self, domain: str, *, after_added: int, after_pruned: int = 0) -> None:
        path = self.index_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}
        events = data.setdefault("events", [])
        events.append({
            "ts_ms": int(time.time() * 1000),
            "domain": domain,
            "added": int(after_added),
            "pruned": int(after_pruned),
        })
        # Cap the events list at 200 entries to keep the file small.
        data["events"] = events[-200:]
        try:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        except OSError:
            return

    def _known_domains(self) -> List[str]:
        domains: set[str] = set()
        for d in (self.seed_dir, self.runtime_dir):
            if not d.exists():
                continue
            for child in d.iterdir():
                if child.is_file() and child.suffix == ".jsonl":
                    domains.add(child.stem)
        return list(domains)


# ---------------------------------------------------------------------------
# Legacy schema bridge
# ---------------------------------------------------------------------------
def _is_legacy_experience_card(data: Dict[str, Any]) -> bool:
    """Heuristic: legacy cards have only the small ExperienceCard fields."""
    if not isinstance(data, dict):
        return False
    legacy_keys = {"card_id", "domain", "intent", "tool_sequence"}
    new_keys = {"environment", "task_category", "tool_ordering_hints", "retrieval_text"}
    return legacy_keys.issubset(data.keys()) and not (new_keys & data.keys())


def _lift_legacy_dict(data: Dict[str, Any]) -> ProcessMemoryCard:
    from .experience import ExperienceCard
    from .memory_types import from_experience_card

    safe = {
        k: data.get(k)
        for k in ("card_id", "domain", "source", "intent", "instruction_template",
                  "needed_evidence", "tool_sequence", "confirmation", "common_trap")
    }
    safe["needed_evidence"] = list(safe.get("needed_evidence") or [])
    safe["tool_sequence"] = list(safe.get("tool_sequence") or [])
    safe["instruction_template"] = str(safe.get("instruction_template") or "")
    safe["confirmation"] = str(safe.get("confirmation") or "")
    safe["common_trap"] = str(safe.get("common_trap") or "")
    safe["intent"] = str(safe.get("intent") or "")
    safe["domain"] = str(safe.get("domain") or "generic")
    safe["source"] = str(safe.get("source") or "seed")
    safe["card_id"] = str(safe.get("card_id") or "seed-card")
    legacy = ExperienceCard(**safe)
    return from_experience_card(legacy)


__all__ = ["MemoryStore"]
