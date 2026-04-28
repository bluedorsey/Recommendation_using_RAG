"""
Embeddings wrapper around sentence-transformers.

We use `paraphrase-multilingual-MiniLM-L12-v2` because:
  - it speaks both English and Arabic (the spec calls for both),
  - it's ~120MB, downloads once, runs CPU-only in a few hundred ms,
  - quality is good enough for a 20-item catalog where retrieval is mostly
    used as a *filter* before the LLM does the real reasoning.

Embeddings for the catalog are computed once and cached to disk under
`.cache/`. Subsequent runs load from disk and start in <1s.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Lazy import: sentence-transformers pulls in torch and is heavy. We don't
# want `from rag import embeddings` to slow down evals that don't actually
# need it.
_MODEL = None
_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
_CACHE_DIR.mkdir(exist_ok=True)


def _get_model():
    global _MODEL
    if _MODEL is None:#check for the sentance encoder 
        from sentence_transformers import SentenceTransformer  
        _MODEL = SentenceTransformer(_MODEL_NAME)
    return _MODEL


def _product_to_text(p: dict) -> str:
    """
    Compose a single string per product for embedding.

    Tags and description carry more semantic signal than the name alone, so
    we include them explicitly. Including category and recipient gives the
    retriever a fighting chance on queries like "self-care gift for a new mom".
    """
    parts = [
        p["name"],
        f"category: {p['category']}",
        f"recipient: {p['for_recipient']}",
        f"tags: {', '.join(p.get('tags', []))}",
        p["description"],
    ]
    if p.get("age_min_months") is not None and p.get("age_max_months") is not None:
        parts.append(f"age range: {p['age_min_months']}-{p['age_max_months']} months")
    return ". ".join(parts)


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of texts. Returns L2-normalized vectors, shape (N, D)."""
    model = _get_model()
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)


def build_or_load_index(products: List[dict]) -> Tuple[np.ndarray, List[dict]]:
    """
    Build (or load from cache) an embedding matrix for the catalog.

    Cache key includes the model name and the catalog's product IDs, so a
    catalog edit invalidates the cache automatically.
    """
    cache_key = (
        _MODEL_NAME + "::" + ",".join(p["id"] for p in products)
    )
    _CACHE_DIR.mkdir(exist_ok=True)
    cache_file = _CACHE_DIR / f"emb_{hashlib.md5(cache_key.encode()).hexdigest()}.pkl"

    if cache_file.exists():
        with cache_file.open("rb") as f:
            return pickle.load(f)

    texts = [_product_to_text(p) for p in products]
    vecs = embed_texts(texts)
    payload = (vecs, products)
    with cache_file.open("wb") as f:
        pickle.dump(payload, f)
    return payload
