"""
Retriever: vector similarity + deterministic constraint filters.

The retrieval pipeline is:

    parsed_intent + query
            │
            ▼
    semantic similarity  (multilingual MiniLM)
            │
            ▼
    hard filters         (budget, age compatibility)
            │
            ▼
    soft re-rank         (recipient match bonus)
            │
            ▼
    top-K candidates  →  passed to LLM

Hard filters live here, NOT in the LLM, because deterministic Python is the
only thing that reliably refuses to suggest a ₹5000 stroller when the user
said "under ₹2000".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from utils.intent import ParsedIntent
from rag.embeddings import build_or_load_index, embed_texts

_DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "products.json"


@dataclass
class Candidate:
    product: dict
    similarity: float        # cosine similarity in [-1, 1], typically [0, 1]
    final_score: float       # similarity + soft bonuses


def load_products() -> List[dict]:
    with _DATA_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def _passes_budget(product: dict, intent: ParsedIntent) -> bool:
    if intent.budget is None:
        return True
    if intent.currency == "INR":
        return product["price_inr"] <= intent.budget
    if intent.currency == "AED":
        return product["price_aed"] <= intent.budget
    return True


def _passes_age(product: dict, intent: ParsedIntent) -> bool:
    if intent.baby_age_months is None:
        return True
    a_min = product.get("age_min_months")
    a_max = product.get("age_max_months")
    # Mom-only items (no age range) always pass age filter.
    if a_min is None or a_max is None:
        return True
    # Allow ±2 months of slack so a 6-month-old still gets a 7-12mo toy.
    return (a_min - 2) <= intent.baby_age_months <= (a_max + 2)


def _recipient_bonus(product: dict, intent: ParsedIntent) -> float:
    """Soft re-rank: products whose recipient matches the query get a small lift."""
    if intent.recipient == "unknown":
        return 0.0
    pr = product["for_recipient"]  # mom / baby / mom_and_baby
    if intent.recipient == "mom" and pr in ("mom", "mom_and_baby"):
        return 0.05
    if intent.recipient == "baby" and pr in ("baby", "mom_and_baby"):
        return 0.05
    if intent.recipient == "both" and pr == "mom_and_baby":
        return 0.08
    return 0.0


class Retriever:
    """In-memory vector search with hard filters."""

    def __init__(self) -> None:
        self.products = load_products()
        self.matrix, _ = build_or_load_index(self.products)

    def retrieve(
        self,
        query: str,
        intent: ParsedIntent,
        top_k: int = 8,
        min_similarity: float = 0.10,
    ) -> List[Candidate]:
        """
        Return up to `top_k` candidates that pass all hard filters.

        `min_similarity` is a sanity floor: queries that match nothing in the
        catalog (e.g. completely off-domain) won't sneak in low-relevance
        items just because the budget filter passed.
        """
        q_vec = embed_texts([query])[0]                 # (D,)
        sims = self.matrix @ q_vec                      # cosine, since both L2-norm
        order = np.argsort(-sims)                       # descending

        out: List[Candidate] = []
        for idx in order:
            sim = float(sims[idx])
            if sim < min_similarity:
                break
            p = self.products[idx]
            if not _passes_budget(p, intent):
                continue
            if not _passes_age(p, intent):
                continue
            score = sim + _recipient_bonus(p, intent)
            out.append(Candidate(product=p, similarity=sim, final_score=score))

        out.sort(key=lambda c: c.final_score, reverse=True)
        return out[:top_k]
