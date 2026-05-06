"""Lightweight, local, deterministic embeddings for retrieval.

The default backend is a hashed-feature TF-IDF / cosine model:

  * No external services, no GPU, no network.
  * Pure-Python, deterministic across runs (no randomness).
  * Cheap enough to run on every retrieval refresh.

A ``sentence_transformers`` backend is supported when the library is
installed and ``REX_EMBEDDING_BACKEND=sentence_transformers``. We never
crash if the library is missing — we fall back to TF-IDF.

Embeddings are cached to disk under ``$REX_EMBEDDING_CACHE_DIR`` so re-runs
don't recompute. The cache is keyed by (backend, model, sha256(text)).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .config import RexConfig, default_config


_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if len(t) > 1]


def _sha256(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8", "ignore")).hexdigest()


# ---------------------------------------------------------------------------
# Hashed-feature TF-IDF backend (default)
# ---------------------------------------------------------------------------
class HashedTfidfEmbedder:
    """Deterministic, dependency-free embedder.

    A document is mapped to a fixed-size dense vector in ``R^dim`` by hashing
    each token to a bucket and accumulating its IDF-weighted TF. The vector
    is L2-normalized so cosine similarity reduces to a dot product.

    Determinism: the hash function is Python's built-in ``hashlib.sha256``
    truncated, which is identical across processes and Python versions.
    """

    name = "tfidf"

    def __init__(self, *, dim: int = 256) -> None:
        self.dim = max(16, int(dim))
        self._idf: Dict[str, float] = {}
        self._fitted: bool = False

    # ---- Fit / refit --------------------------------------------------
    def fit(self, documents: Iterable[str]) -> "HashedTfidfEmbedder":
        df: Dict[str, int] = {}
        n = 0
        for doc in documents:
            tokens = set(_tokenize(doc))
            n += 1
            for t in tokens:
                df[t] = df.get(t, 0) + 1
        n = max(1, n)
        # Smoothed IDF (Lucene-style) — works even when a token only appears once.
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self._fitted = True
        return self

    def is_fitted(self) -> bool:
        return self._fitted

    # ---- Encode -------------------------------------------------------
    def encode(self, text: str) -> List[float]:
        tokens = _tokenize(text)
        if not tokens:
            return [0.0] * self.dim
        tf: Dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        vec = [0.0] * self.dim
        for tok, count in tf.items():
            idf = self._idf.get(tok, 1.0)  # unseen tokens still contribute
            bucket = self._bucket(tok)
            vec[bucket] += float(count) * float(idf)
        # L2 normalize so dot product == cosine.
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            vec = [v / norm for v in vec]
        return vec

    def encode_batch(self, texts: Sequence[str]) -> List[List[float]]:
        return [self.encode(t) for t in texts]

    def _bucket(self, token: str) -> int:
        h = hashlib.blake2b(token.encode("utf-8", "ignore"), digest_size=8).digest()
        return int.from_bytes(h, "big") % self.dim


# ---------------------------------------------------------------------------
# Optional sentence-transformers backend
# ---------------------------------------------------------------------------
class SentenceTransformerEmbedder:
    """Wrap ``sentence_transformers.SentenceTransformer`` if available.

    Loads lazily so importing this module never fails when the library is
    absent. If the model can't be loaded for any reason, callers should
    fall back to the TF-IDF backend (``embedder_for_config`` does this).
    """

    name = "sentence_transformers"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: Optional[Any] = None
        self.dim = 0

    def _load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer  # type: ignore

        self._model = SentenceTransformer(self.model_name)
        try:
            self.dim = int(self._model.get_sentence_embedding_dimension())
        except Exception:
            self.dim = 384

    def fit(self, documents: Iterable[str]) -> "SentenceTransformerEmbedder":
        # Pre-trained models don't need fitting; we just lazy-load.
        # Consume the iterator so callers using generators don't deadlock.
        for _ in documents:
            pass
        self._load()
        return self

    def is_fitted(self) -> bool:
        return self._model is not None

    def encode(self, text: str) -> List[float]:
        self._load()
        vec = self._model.encode(text, normalize_embeddings=True)  # type: ignore[attr-defined]
        return [float(v) for v in vec]

    def encode_batch(self, texts: Sequence[str]) -> List[List[float]]:
        self._load()
        vecs = self._model.encode(list(texts), normalize_embeddings=True, batch_size=32)  # type: ignore[attr-defined]
        return [[float(v) for v in row] for row in vecs]


# ---------------------------------------------------------------------------
# Cache wrapper
# ---------------------------------------------------------------------------
@dataclass
class EmbeddingCache:
    """JSONL-backed embedding cache keyed by (backend, model, sha256(text)).

    Entries are loaded lazily into memory and written out on close.
    """

    cache_dir: Path
    backend_name: str
    model_name: str
    enabled: bool = True

    def __post_init__(self) -> None:
        self._dirty: bool = False
        self._mem: Dict[str, List[float]] = {}
        self._loaded: bool = False

    def _path(self) -> Path:
        safe_model = re.sub(r"[^a-zA-Z0-9._-]+", "_", self.model_name or "default")
        return self.cache_dir / f"emb_{self.backend_name}_{safe_model}.jsonl"

    def _ensure_loaded(self) -> None:
        if self._loaded or not self.enabled:
            return
        self._loaded = True
        path = self._path()
        if not path.exists():
            return
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = rec.get("k")
                    vec = rec.get("v")
                    if isinstance(key, str) and isinstance(vec, list):
                        self._mem[key] = [float(x) for x in vec]
        except OSError:
            return

    def get(self, text: str) -> Optional[List[float]]:
        if not self.enabled:
            return None
        self._ensure_loaded()
        return self._mem.get(_sha256(text))

    def put(self, text: str, vec: List[float]) -> None:
        if not self.enabled:
            return
        self._ensure_loaded()
        self._mem[_sha256(text)] = list(vec)
        self._dirty = True

    def flush(self) -> None:
        if not self.enabled or not self._dirty:
            return
        path = self._path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for key, vec in sorted(self._mem.items()):
                f.write(json.dumps({"k": key, "v": vec}) + "\n")
        tmp.replace(path)
        self._dirty = False


# ---------------------------------------------------------------------------
# High-level facade
# ---------------------------------------------------------------------------
class CachedEmbedder:
    """Embedder + cache facade used by the retrieval system.

    The facade owns a backend (TF-IDF or sentence_transformers) and a cache.
    ``encode`` reads from the cache first, falls back to the backend, then
    writes the result back. ``encode_batch`` does the same in bulk.
    """

    def __init__(
        self,
        *,
        backend: Any,
        cache: Optional[EmbeddingCache] = None,
    ) -> None:
        self.backend = backend
        self.cache = cache
        self.dim = getattr(backend, "dim", 0) or 0

    def fit(self, documents: Iterable[str]) -> "CachedEmbedder":
        self.backend.fit(documents)
        self.dim = getattr(self.backend, "dim", self.dim) or self.dim
        return self

    def is_fitted(self) -> bool:
        return bool(getattr(self.backend, "is_fitted", lambda: True)())

    def encode(self, text: str) -> List[float]:
        if self.cache is not None:
            cached = self.cache.get(text)
            if cached is not None:
                return cached
        vec = self.backend.encode(text)
        if self.cache is not None:
            self.cache.put(text, vec)
        return vec

    def encode_batch(self, texts: Sequence[str]) -> List[List[float]]:
        if self.cache is None:
            return self.backend.encode_batch(list(texts))
        out: List[List[float]] = []
        missing_idx: List[int] = []
        missing_texts: List[str] = []
        for i, t in enumerate(texts):
            cached = self.cache.get(t)
            if cached is None:
                out.append([])
                missing_idx.append(i)
                missing_texts.append(t)
            else:
                out.append(cached)
        if missing_idx:
            fresh = self.backend.encode_batch(missing_texts)
            for j, idx in enumerate(missing_idx):
                vec = fresh[j]
                out[idx] = vec
                self.cache.put(missing_texts[j], vec)
        return out

    def flush(self) -> None:
        if self.cache is not None:
            self.cache.flush()

    def invalidate_cache(self) -> None:
        if self.cache is not None:
            self.cache._mem.clear()
            self.cache._dirty = True


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------
def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity for already-normalized or arbitrary vectors."""
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        x = float(a[i])
        y = float(b[i])
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def embedder_for_config(
    config: Optional[RexConfig] = None,
    *,
    documents: Optional[Sequence[str]] = None,
) -> CachedEmbedder:
    """Build the configured embedder, falling back to TF-IDF on any error.

    If ``documents`` is provided, the backend is fit immediately. Otherwise
    callers are expected to call ``fit()`` themselves.
    """
    cfg = config or default_config()
    backend: Any
    backend_name = cfg.embedding_backend
    if backend_name == "sentence_transformers":
        try:
            backend = SentenceTransformerEmbedder(cfg.embedding_model)
            # Trigger lazy load to validate availability.
            backend._load()
        except Exception:
            backend = HashedTfidfEmbedder(dim=cfg.embedding_dim)
            backend_name = "tfidf"
    else:
        backend = HashedTfidfEmbedder(dim=cfg.embedding_dim)
        backend_name = "tfidf"

    cache: Optional[EmbeddingCache] = None
    if cfg.embedding_cache_enabled:
        cache = EmbeddingCache(
            cache_dir=cfg.embedding_cache_dir,
            backend_name=backend_name,
            model_name=cfg.embedding_model if backend_name == "sentence_transformers" else f"hashed_{cfg.embedding_dim}",
            enabled=True,
        )
    embedder = CachedEmbedder(backend=backend, cache=cache)
    if documents is not None:
        embedder.fit(documents)
    return embedder


__all__ = [
    "HashedTfidfEmbedder",
    "SentenceTransformerEmbedder",
    "EmbeddingCache",
    "CachedEmbedder",
    "cosine_similarity",
    "embedder_for_config",
]
