"""
Lightweight, deterministic intent parser.

This runs *before* the LLM and the retriever, and extracts hard constraints
that we want to enforce in code rather than trust to the model:

  - budget       (max price in INR or AED)
  - baby_age     (in months)
  - currency     (₹/INR or AED)
  - recipient    (mom / baby / both)
  - in_scope     (false for obvious out-of-domain queries)

Doing this in code is one of the most important defenses against the classic
LLM failure: "yes here are 5 lovely picks, all way over your budget".

The parser is intentionally conservative — when in doubt it leaves the field
as None and lets the LLM decide.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Literal

Currency = Literal["INR", "AED", "UNKNOWN"]
Recipient = Literal["mom", "baby", "both", "unknown"]


# ---- Out-of-scope detection -------------------------------------------------
# The system is a *gift finder for moms and babies*. We refuse confidently
# when the query is clearly something else.
_OUT_OF_SCOPE_PATTERNS = [
    r"\b(gaming|laptop|graphics card|gpu|cpu|rtx|ps5|xbox)\b",
    r"\b(car|motorcycle|bike rental|insurance)\b",
    r"\b(crypto|bitcoin|stock tip|investment)\b",
    r"\b(weapon|gun|knife)\b",
    r"\b(my dog|my cat|for my pet|dog food|cat food)\b",
    r"\bgirlfriend\b",          # not the target persona
    r"\bboyfriend\b",
]

# ---- Prompt-injection sniff test --------------------------------------------
# We don't try to be a security product, just catch the obvious cases so the
# LLM doesn't get tricked into ignoring the schema.
_INJECTION_PATTERNS = [
    r"ignore (all |the )?(previous|prior|above) instructions",
    r"forget (everything|the system prompt)",
    r"you are now ",
    r"reveal (your|the) (system )?prompt",
    r"jailbreak",
]


@dataclass
class ParsedIntent:
    raw_query: str
    budget: Optional[float] = None
    currency: Currency = "UNKNOWN"
    baby_age_months: Optional[int] = None
    recipient: Recipient = "unknown"
    in_scope: bool = True
    injection_detected: bool = False
    notes: str = ""


# Arabic-Indic digits  ٠١٢٣٤٥٦٧٨٩  →  Latin 0-9
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _normalize_digits(s: str) -> str:
    """Convert Arabic-Indic digits to Latin so our regexes work uniformly."""
    return s.translate(_ARABIC_DIGITS)


def _extract_budget(q: str) -> tuple[Optional[float], Currency]:
    """Find a max-budget clause. Returns (amount, currency)."""
    qn = _normalize_digits(q.lower().replace(",", ""))

    # ₹2000, ₹ 2000, Rs 2000, Rs. 2000, INR 2000, 2000 rupees
    inr_patterns = [
        r"(?:under|below|less than|<|upto|up to|within|max|budget(?:\s+of)?)?\s*₹\s*(\d{2,7})",
        r"(?:under|below|less than|<|upto|up to|within|max|budget(?:\s+of)?)?\s*(?:rs\.?|inr)\s*(\d{2,7})",
        r"(\d{2,7})\s*(?:rupees|rs\.?|inr)\b",
    ]
    for pat in inr_patterns:
        m = re.search(pat, qn)
        if m:
            return float(m.group(1)), "INR"

    # AED 200, 200 AED, 200 dirhams, 200 درهم
    aed_patterns = [
        r"(?:under|below|less than|<|upto|up to|within|max|budget(?:\s+of)?)?\s*aed\s*(\d{2,6})",
        r"(\d{2,6})\s*aed\b",
        r"(\d{2,6})\s*dirhams?\b",
        r"(\d{2,6})\s*درهم",
        r"درهم\s*(\d{2,6})",
    ]
    for pat in aed_patterns:
        m = re.search(pat, qn)
        if m:
            return float(m.group(1)), "AED"

    return None, "UNKNOWN"


def _extract_age(q: str) -> Optional[int]:
    """Try to find baby's age in months. Returns months as int."""
    qn = _normalize_digits(q.lower())

    # "6-month-old", "6 month old", "6mo", "6 months"
    m = re.search(r"(\d{1,2})\s*[- ]?\s*(?:months?|mo|m)\b[- ]?old?", qn)
    if m:
        return int(m.group(1))

    m = re.search(r"\b(\d{1,2})\s*(?:months?|mo)\b", qn)
    if m:
        return int(m.group(1))

    # Arabic: "6 أشهر" / "٦ شهور" / "شهر"
    m = re.search(r"(\d{1,2})\s*(?:أشهر|شهور|شهرا?|شهر)", qn)
    if m:
        return int(m.group(1))

    # "1-year-old", "1 year old", "2 yo"
    m = re.search(r"(\d{1,2})\s*[- ]?\s*(?:year|yr|yo|y)s?[- ]?old?", qn)
    if m:
        return int(m.group(1)) * 12

    m = re.search(r"\b(\d{1,2})\s*(?:years?|yrs?)\b", qn)
    if m:
        return int(m.group(1)) * 12

    # Arabic: "سنة" / "سنوات" / "عام"
    m = re.search(r"(\d{1,2})\s*(?:سنوات|سنة|عام|أعوام)", qn)
    if m:
        return int(m.group(1)) * 12

    if re.search(r"\b(newborn|new[- ]born|infant)\b", qn) or "حديث الولادة" in qn or "رضيع" in qn:
        return 0

    if re.search(r"\btoddler\b", qn) or "طفل صغير" in qn or "دارج" in qn:
        return 18

    return None


def _extract_recipient(q: str) -> Recipient:
    qn = _normalize_digits(q.lower())
    has_mom = bool(re.search(
        r"\b(mom|mum|mother|new mom|breastfeeding|postpartum|maternity|pregnant|expecting)\b", qn
    )) or any(w in qn for w in ("أم", "للأم", "للوالدة", "حامل", "نفساء", "مرضعة"))
    has_baby = bool(re.search(
        r"\b(baby|infant|newborn|toddler|child|kid)\b", qn
    )) or any(w in qn for w in ("طفل", "رضيع", "بيبي", "مولود"))
    if has_mom and has_baby:
        return "both"
    if has_mom:
        return "mom"
    if has_baby:
        return "baby"
    if _extract_age(qn) is not None:
        return "baby"
    return "unknown"


def _is_out_of_scope(q: str) -> bool:
    qn = q.lower()
    return any(re.search(p, qn) for p in _OUT_OF_SCOPE_PATTERNS)


def _has_injection(q: str) -> bool:
    qn = q.lower()
    return any(re.search(p, qn) for p in _INJECTION_PATTERNS)


def parse_intent(query: str) -> ParsedIntent:
    """Parse a natural-language gift query into structured constraints."""
    q = (query or "").strip()
    if not q:
        return ParsedIntent(raw_query="", in_scope=False, notes="empty query")

    intent = ParsedIntent(raw_query=q)
    intent.budget, intent.currency = _extract_budget(q)
    intent.baby_age_months = _extract_age(q)
    intent.recipient = _extract_recipient(q)
    intent.injection_detected = _has_injection(q)

    if _is_out_of_scope(q):
        intent.in_scope = False
        intent.notes = "out of domain (not a mom/baby gift query)"
    elif intent.injection_detected:
        # We don't refuse outright — we strip and continue — but we flag it.
        intent.notes = "possible prompt injection; ignoring instruction-like content"

    return intent
