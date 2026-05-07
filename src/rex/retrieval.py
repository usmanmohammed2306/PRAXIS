"""Hybrid procedural-memory retrieval engine.

Combines:

  * **Lexical** — BM25 over each card's ``retrieval_text``.
  * **Semantic** — cosine similarity over deterministic local embeddings.
  * **Diversity** — caps how many cards may share an intent label so the
    retriever doesn't return five copies of the same lesson.

Stateful query construction lives in ``RetrievalQuery`` (see
``memory_types``); this module is the **engine** that ranks against those
queries.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import RexConfig, default_config
from .embeddings import CachedEmbedder, cosine_similarity, embedder_for_config
from .memory_types import (
    ProcessMemoryCard,
    RetrievalQuery,
    RetrievalResult,
)


_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if len(t) > 1]


# ---------------------------------------------------------------------------
# BM25 lexical scorer
# ---------------------------------------------------------------------------
class _BM25Scorer:
    """Plain-Python BM25 scorer over a fixed corpus."""

    def __init__(self, documents: Sequence[List[str]], *, k1: float = 1.4, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs = list(documents)
        df: Dict[str, int] = {}
        for d in self.docs:
            for t in set(d):
                df[t] = df.get(t, 0) + 1
        n = max(1, len(self.docs))
        self.idf: Dict[str, float] = {
            t: math.log(1 + (n - cnt + 0.5) / (cnt + 0.5))
            for t, cnt in df.items()
        }
        total_len = sum(len(d) for d in self.docs)
        self.avgdl = (total_len / max(1, len(self.docs))) if self.docs else 1.0
        # Pre-compute term frequencies.
        self.tfs: List[Dict[str, int]] = []
        for d in self.docs:
            tf: Dict[str, int] = {}
            for t in d:
                tf[t] = tf.get(t, 0) + 1
            self.tfs.append(tf)

    def score(self, query_tokens: Sequence[str]) -> List[float]:
        if not self.docs:
            return []
        if not query_tokens:
            return [0.0] * len(self.docs)
        qset = set(query_tokens)
        scores: List[float] = [0.0] * len(self.docs)
        for i, tf in enumerate(self.tfs):
            dl = max(1, len(self.docs[i]))
            length_norm = 1.0 - self.b + self.b * (dl / max(1.0, self.avgdl))
            s = 0.0
            for t in qset:
                f = tf.get(t)
                if not f:
                    continue
                idf = self.idf.get(t, 0.0)
                s += idf * (f * (self.k1 + 1.0)) / (f + self.k1 * length_norm)
            scores[i] = s
        return scores


# ---------------------------------------------------------------------------
# Score normalization
# ---------------------------------------------------------------------------
def _minmax_normalize(values: Sequence[float]) -> List[float]:
    """Rescale ``values`` to [0, 1]; constant input maps to all-zero."""
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return [0.0] * len(values)
    span = hi - lo
    return [(float(v) - lo) / span for v in values]


# ---------------------------------------------------------------------------
# Main retriever
# ---------------------------------------------------------------------------
class HybridRetriever:
    """Hybrid BM25 + embedding cosine retriever with diversity filtering.

    Construction precomputes:

      * tokenized retrieval text for each card → BM25 scorer
      * card embeddings via the configured embedder

    ``search`` is therefore O(N) per query where N is the corpus size.
    """

    def __init__(
        self,
        cards: Sequence[ProcessMemoryCard],
        *,
        config: Optional[RexConfig] = None,
        embedder: Optional[CachedEmbedder] = None,
    ) -> None:
        self.config = config or default_config()
        self.cards: List[ProcessMemoryCard] = list(cards)

        docs = [_tokenize(c.retrieval_text) for c in self.cards]
        self._bm25 = _BM25Scorer(docs)

        # Embedder: fit on the corpus so IDF is meaningful.
        retrieval_texts = [c.retrieval_text for c in self.cards]
        self.embedder = embedder or embedder_for_config(self.config, documents=retrieval_texts)
        if not self.embedder.is_fitted():
            self.embedder.fit(retrieval_texts)
        # Cache card vectors so we don't re-encode on every query.
        self._card_vectors: List[List[float]] = (
            self.embedder.encode_batch(retrieval_texts) if self.cards else []
        )

    # ---- Diagnostics --------------------------------------------------
    @property
    def num_cards(self) -> int:
        return len(self.cards)

    # ---- Query construction helpers ----------------------------------
    @staticmethod
    def build_query_text(query: RetrievalQuery) -> str:
        """Build the text passed to lexical / semantic scorers."""
        parts: List[str] = []
        if query.text:
            parts.append(query.text)
        if query.environment:
            parts.append(query.environment)
        if query.last_tool:
            parts.append(query.last_tool)
        if query.last_observation:
            parts.append(query.last_observation[:240])
        for t in query.failed_tools[-3:]:
            parts.append(f"failed:{t}")
        for t in query.attempted_tools[-3:]:
            parts.append(f"tried:{t}")
        return " ".join(p for p in parts if p)

    # ---- Search -------------------------------------------------------
    def search(self, query: RetrievalQuery) -> RetrievalResult:
        if not self.cards:
            return RetrievalResult(cards=[], scores=[], diagnostic={"reason": "empty_corpus"})

        text = self.build_query_text(query)
        q_tokens = _tokenize(text)

        # 1) Lexical
        bm25_raw = self._bm25.score(q_tokens)
        # 2) Semantic
        if text and self.embedder.is_fitted():
            q_vec = self.embedder.encode(text)
            emb_raw = [cosine_similarity(q_vec, v) for v in self._card_vectors]
        else:
            emb_raw = [0.0] * len(self.cards)

        # Normalize both so the weighted sum is well-conditioned.
        bm25_n = _minmax_normalize(bm25_raw)
        emb_n = _minmax_normalize(emb_raw)
        w_lex = float(self.config.hybrid_lexical_weight)
        w_emb = float(self.config.hybrid_embedding_weight)
        denom = max(1e-6, w_lex + w_emb)
        combined: List[Tuple[float, int]] = []
        for i, card in enumerate(self.cards):
            score = (w_lex * bm25_n[i] + w_emb * emb_n[i]) / denom
            # Apply environment match boost so domain-correct cards win ties.
            if query.environment and card.environment == query.environment:
                score += 0.05
            # Slight boost for cards with high quality score.
            score += 0.05 * card.quality_score()
            combined.append((score, i))

        combined.sort(key=lambda t: (t[0], -t[1]), reverse=True)

        # Threshold + diversity + dedup pass.
        threshold = float(self.config.retrieval_score_threshold)
        diversity_cap = max(1, int(self.config.diversity_intent_cap))
        k = max(1, int(query.top_k or self.config.top_k))

        chosen: List[int] = []
        chosen_scores: List[float] = []
        seen_intents: Dict[str, int] = {}
        # Phase 1: high-quality, environment-matched cards first.
        for score, idx in combined:
            if score < threshold and chosen:
                break
            card = self.cards[idx]
            if seen_intents.get(card.task_category, 0) >= diversity_cap:
                continue
            chosen.append(idx)
            chosen_scores.append(score)
            seen_intents[card.task_category] = seen_intents.get(card.task_category, 0) + 1
            if len(chosen) >= k:
                break
        # Phase 2: backfill if we didn't reach k due to threshold.
        if len(chosen) < k:
            for score, idx in combined:
                if idx in chosen:
                    continue
                chosen.append(idx)
                chosen_scores.append(score)
                if len(chosen) >= k:
                    break

        result_cards = [self.cards[i] for i in chosen]
        # Classify memory source for retrieval logging.
        # Cards whose card_id starts with "seed-" come from the seed (permanent)
        # bank; others were distilled at runtime.  "hybrid" when both are present.
        _sources = set()
        for c in result_cards:
            _sources.add("seed" if str(c.card_id).startswith("seed-") else "runtime")
        _memory_source = (
            "seed" if _sources == {"seed"}
            else "runtime" if _sources == {"runtime"}
            else "hybrid" if _sources
            else "unknown"
        )

        diagnostic: Dict[str, Any] = {
            "query_text": text,
            "query_environment": query.environment,
            "query_step": query.step_index,
            "query_last_tool": query.last_tool,
            "num_corpus": len(self.cards),
            "top_lex_max": max(bm25_raw) if bm25_raw else 0.0,
            "top_emb_max": max(emb_raw) if emb_raw else 0.0,
            "selected_card_ids": [c.card_id for c in result_cards],
            "selected_intents": [c.task_category for c in result_cards],
            "memory_source": _memory_source,
        }
        # Bump usage counters so quality scoring evolves.
        for c in result_cards:
            c.update_usage()
        return RetrievalResult(cards=result_cards, scores=chosen_scores, diagnostic=diagnostic)


# ---------------------------------------------------------------------------
# Convenience builder used by the agent loops
# ---------------------------------------------------------------------------
def build_query(
    *,
    initial_user: str,
    messages: Sequence[Dict[str, Any]],
    environment: str = "generic",
    benchmark: str = "unknown",
    controller: str = "rex",
    step_index: int = 0,
    top_k: int = 3,
) -> RetrievalQuery:
    """Build a ``RetrievalQuery`` from the conversation state.

    The latest tool call/result and the latest user reply are included in
    the search query so retrieval evolves with the trajectory.
    """
    last_user = ""
    last_tool = ""
    last_obs = ""
    attempted: List[str] = []
    failed: List[str] = []
    last_was_tool_call: Optional[str] = None
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "user":
            content = m.get("content") or ""
            if content:
                last_user = str(content)
        elif role == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") if isinstance(tc, dict) else None
                if isinstance(fn, dict) and fn.get("name"):
                    last_was_tool_call = str(fn["name"])
                    attempted.append(last_was_tool_call)
        elif role == "tool":
            last_tool = str(m.get("name") or "")
            last_obs = str(m.get("content") or "")
            # Heuristic: tool observation containing "error" or "not found"
            # is treated as a failure that should bias retrieval toward
            # recovery cards.
            obs_lower = last_obs.lower()
            if any(k in obs_lower for k in ("error", "not found", "invalid", "failed")):
                if last_was_tool_call:
                    failed.append(last_was_tool_call)

    text_parts = [initial_user or ""]
    if last_user and last_user != initial_user:
        text_parts.append(last_user)
    return RetrievalQuery(
        text=" ".join(p for p in text_parts if p),
        environment=environment,
        benchmark=benchmark,
        last_tool=last_tool,
        last_observation=last_obs,
        attempted_tools=attempted,
        failed_tools=failed,
        controller=controller,
        step_index=step_index,
        top_k=top_k,
    )


__all__ = [
    "HybridRetriever",
    "build_query",
]
