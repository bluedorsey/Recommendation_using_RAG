"""Language detection — minimal, dependency-free."""

from __future__ import annotations
from typing import Literal

Language = Literal["en", "ar"]


def detect_language(text: str) -> Language:
    """
    Detect Arabic vs English by counting Arabic-script characters.

    We don't import langdetect/fasttext: the only two languages we serve are
    English and Arabic, and Arabic uses a non-Latin script, so a character-set
    heuristic is both faster and more reliable than a probabilistic detector
    on short queries.
    """
    if not text:
        return "en"
    arabic = 0
    latin = 0
    for ch in text:
        cp = ord(ch)
        # Arabic, Arabic Supplement, Arabic Extended-A
        if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F or 0x08A0 <= cp <= 0x08FF:
            arabic += 1
        elif ch.isalpha():
            latin += 1
    if arabic > latin:
        return "ar"
    return "en"
