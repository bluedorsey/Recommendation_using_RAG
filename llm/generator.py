"""
Top-level generator: query in, validated GiftFinderResponse out.

Pipeline:

    parse_intent  ──►  refuse if out of scope / empty
        │
        ▼
    retrieve      ──►  fallback if 0 candidates after filtering
        │
        ▼
    LLM (EN)      ──►  parse JSON  ──►  Pydantic validate
        │                                  │
        │                              retry once on failure
        │                                  │
        │                              still fail  ──►  structured error
        ▼
    LLM (AR)      ──►  same loop

A *post-LLM grounding check* verifies that every recommended product_name
appears in the candidate list. If the model hallucinated, we drop those
items; if too few survive, we fallback.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional
from pydantic import ValidationError
from llm.client import call_llm, LLMError
from llm.prompts import build_en_prompt, build_ar_prompt
from rag.retriever import Candidate, Retriever
from utils.intent import ParsedIntent, parse_intent
from utils.language import detect_language
from utils.schema import GiftFinderResponse, Recommendation


@dataclass
class GenerationResult:
    """Bundle returned to the UI / evals."""
    en: GiftFinderResponse
    ar: GiftFinderResponse
    intent: ParsedIntent
    candidates: List[Candidate]
    debug: dict        # latencies, retries, validation errors, etc.


#  helpers 

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _strip_to_json(raw: str) -> str:
    """
    Pull the JSON object out of a model response.

    Handles three cases:
      1. Pure JSON  →  return as-is
      2. ```json ... ```  →  strip fences
      3. JSON embedded in prose  →  take from first '{' to last '}'
    """
    s = raw.strip()
    m = _JSON_FENCE.search(s)
    if m:
        return m.group(1).strip()
    # Take the largest balanced-looking {...} blob.
    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        return s[first : last + 1]
    return s


def _make_fallback(query: str, language: str, reason: str) -> GiftFinderResponse:
    return GiftFinderResponse(
        query=query,
        language=language,                 # type: ignore[arg-type]
        recommendations=[],
        fallback=True,
        fallback_reason=reason,
    )


def _drop_hallucinated(
    resp: GiftFinderResponse, candidates: List[Candidate]
) -> tuple[GiftFinderResponse, int]:
    """
    Remove any recommendation whose product_name isn't in the candidate set.

    Returns (cleaned_response, n_dropped).
    """
    allowed = {c.product["name"] for c in candidates}
    kept: List[Recommendation] = []
    dropped = 0
    for r in resp.recommendations:
        if r.product_name in allowed:
            kept.append(r)
        else:
            dropped += 1

    if dropped == 0:
        return resp, 0

    # If too few survive, convert to fallback.
    if len(kept) < 3:
        fb = _make_fallback(
            resp.query,
            resp.language,
            "Model produced ungrounded items; insufficient verified matches.",
        )
        return fb, dropped

    cleaned = GiftFinderResponse(
        query=resp.query,
        language=resp.language,
        recommendations=kept,
        fallback=False,
    )
    return cleaned, dropped


def _generate_one_language(
    query: str,
    candidates: List[Candidate],
    language: str,
    currency_pref: str,
    debug: dict,
) -> GiftFinderResponse:
    """Run the LLM once (with one retry) for a single language."""
    builder = build_en_prompt if language == "en" else build_ar_prompt
    system, user = builder(query, candidates, currency_pref=currency_pref)

    last_err: Optional[str] = None
    for attempt in range(2):  # original + 1 retry
        try:
            resp = call_llm(system, user)
        except LLMError as e:
            last_err = f"llm_error: {e}"
            debug.setdefault(f"{language}_errors", []).append(last_err)
            continue

        debug.setdefault(f"{language}_latency_ms", []).append(resp.latency_ms)
        debug.setdefault(f"{language}_provider", resp.provider)
        debug.setdefault(f"{language}_model", resp.model)

        raw_json = _strip_to_json(resp.text)
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as e:
            last_err = f"json_decode: {e}"
            debug.setdefault(f"{language}_errors", []).append(last_err)
            # On retry, append a corrective hint.
            user = (
                user
                + "\n\nIMPORTANT: your previous output was not valid JSON. "
                  "Return ONLY a JSON object, no prose, no markdown fences."
            )
            continue

        # Force the language field to match what we asked for.
        data["language"] = language
        # Echo the query if missing.
        data.setdefault("query", query)

        try:
            parsed = GiftFinderResponse.model_validate(data)
        except ValidationError as e:
            last_err = f"schema: {e.errors()[:2]}"
            debug.setdefault(f"{language}_errors", []).append(last_err)
            user = (
                user
                + f"\n\nIMPORTANT: previous output failed schema validation: "
                  f"{e.errors()[:2]}. Fix and re-emit a valid JSON object."
            )
            continue

        # Grounding check: drop any product_name not in candidates.
        cleaned, n_dropped = _drop_hallucinated(parsed, candidates)
        if n_dropped:
            debug.setdefault(f"{language}_hallucinations_dropped", 0)
            debug[f"{language}_hallucinations_dropped"] += n_dropped
        return cleaned

    # Both attempts failed — return a structured fallback so the UI never breaks.
    return _make_fallback(
        query,
        language,
        f"Generation failed after retry ({last_err or 'unknown error'}).",
    )


#  Public API 

class GiftFinder:
    """Top-level façade. Construct once (loads embeddings), call .generate()."""

    def __init__(self) -> None:
        self.retriever = Retriever()

    def generate(self, query: str) -> GenerationResult:
        debug: dict = {}
        intent = parse_intent(query)
        debug["intent"] = {
            "budget": intent.budget,
            "currency": intent.currency,
            "baby_age_months": intent.baby_age_months,
            "recipient": intent.recipient,
            "in_scope": intent.in_scope,
            "injection_detected": intent.injection_detected,
            "notes": intent.notes,
        }

        input_lang = detect_language(query)
        currency_pref = intent.currency if intent.currency != "UNKNOWN" else "INR"

        # Refuse confidently for out-of-scope queries — no LLM call needed.
        if not intent.in_scope:
            reason_en = (
                "This query is outside the scope of a mom-and-baby gift finder."
            )
            reason_ar = (
                "هذا الطلب خارج نطاق مساعد اختيار هدايا الأمهات والأطفال."
            )
            return GenerationResult(
                en=_make_fallback(query, "en", reason_en),
                ar=_make_fallback(query, "ar", reason_ar),
                intent=intent,
                candidates=[],
                debug=debug,
            )

        candidates = self.retriever.retrieve(query, intent, top_k=8)
        debug["n_candidates"] = len(candidates)
        debug["candidate_ids"] = [c.product["id"] for c in candidates]

        if len(candidates) < 3:
            reason_en = (
                "Not enough catalog items match your constraints "
                "(budget, age, or category)."
            )
            reason_ar = (
                "لا توجد منتجات كافية في الكتالوج تطابق معاييرك "
                "(الميزانية أو العمر أو الفئة)."
            )
            return GenerationResult(
                en=_make_fallback(query, "en", reason_en),
                ar=_make_fallback(query, "ar", reason_ar),
                intent=intent,
                candidates=candidates,
                debug=debug,
            )

        en_resp = _generate_one_language(query, candidates, "en", currency_pref, debug)
        ar_resp = _generate_one_language(query, candidates, "ar", currency_pref, debug)
        debug["input_language"] = input_lang

        return GenerationResult(
            en=en_resp,
            ar=ar_resp,
            intent=intent,
            candidates=candidates,
            debug=debug,
        )
