"""
LLM client.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Iterator, Optional

import requests

# Auto-load .env once, at import time, so users don't have to source it
# manually before `python cli.py` or `streamlit run`.
try:
    from dotenv import load_dotenv  # type: ignore[import-untyped]
    load_dotenv()
except ImportError:
    pass


class LLMError(RuntimeError):
    """Raised when the provider returns an unrecoverable error."""


@dataclass
class LLMResponse:
    text: str                      # raw text the model produced
    provider: str                  # openrouter or gemini
    model: str
    latency_ms: int


#  OpenRouter 
def _call_openrouter(
    system: str,
    user: str,
    *,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1500,
    timeout: int = 60,
) -> LLMResponse:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise LLMError("OPENROUTER_API_KEY not set")

    model = model or os.environ.get(
        "OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free"
    )
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        # JSON mode hint. Some free models honor it, some ignore it.
        "response_format": {"type": "json_object"},
    }

    t0 = time.time()
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    dt = int((time.time() - t0) * 1000)

    if r.status_code != 200:
        raise LLMError(f"OpenRouter HTTP {r.status_code}: {r.text[:300]}")

    data = r.json()
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise LLMError(f"OpenRouter unexpected response shape: {e} :: {str(data)[:300]}")
    return LLMResponse(text=text, provider="openrouter", model=model, latency_ms=dt)


#  Gemini (Google AI Studio) 
def _call_gemini(
    system: str,
    user: str,
    *,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    timeout: int = 60,
) -> LLMResponse:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise LLMError("GEMINI_API_KEY not set")

    model = model or os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    _RESPONSE_SCHEMA = {
        "type": "object",
        "properties": {
            "query":           {"type": "string"},
            "language":        {"type": "string"},
            "recommendations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "product_name": {"type": "string"},
                        "price":        {"type": "string"},
                        "reason":       {"type": "string"},
                        "confidence":   {"type": "number"},
                    },
                    "required": ["product_name", "price", "reason", "confidence"],
                },
            },
            "fallback":        {"type": "boolean"},
            "fallback_reason": {"type": "string"},
        },
        "required": ["query", "language", "recommendations", "fallback"],
    }
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
            "responseSchema": _RESPONSE_SCHEMA,
            # Disable thinking for structured output: thinking tokens count
            # against maxOutputTokens and leave too little room for JSON.
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }

    t0 = time.time()
    r = requests.post(url, json=payload, timeout=timeout)
    dt = int((time.time() - t0) * 1000)

    if r.status_code != 200:
        raise LLMError(f"Gemini HTTP {r.status_code}: {r.text[:300]}")

    data = r.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise LLMError(f"Gemini unexpected response shape: {e} :: {str(data)[:300]}")
    return LLMResponse(text=text, provider="gemini", model=model, latency_ms=dt)


def _stream_openrouter(system: str, user: str, temperature: float = 0.2, max_tokens: int = 1500, timeout: int = 60) -> Iterator[str]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise LLMError("OPENROUTER_API_KEY not set")
    model = os.environ.get("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free")
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "response_format": {"type": "json_object"},
    }
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload, stream=True, timeout=timeout,
    )
    if r.status_code != 200:
        raise LLMError(f"OpenRouter HTTP {r.status_code}: {r.text[:300]}")
    for line in r.iter_lines():
        if not line:
            continue
        s = line.decode("utf-8") if isinstance(line, bytes) else line
        if not s.startswith("data: "):
            continue
        data = s[6:]
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
            token = chunk["choices"][0]["delta"].get("content", "")
            if token:
                yield token
        except (json.JSONDecodeError, KeyError, IndexError):
            pass


def _stream_gemini(system: str, user: str, temperature: float = 0.2, max_tokens: int = 4096, timeout: int = 120) -> Iterator[str]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise LLMError("GEMINI_API_KEY not set")
    model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:streamGenerateContent?alt=sse&key={api_key}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    r = requests.post(url, json=payload, stream=True, timeout=timeout)
    if r.status_code != 200:
        raise LLMError(f"Gemini HTTP {r.status_code}: {r.text[:300]}")
    for line in r.iter_lines():
        if not line:
            continue
        s = line.decode("utf-8") if isinstance(line, bytes) else line
        if not s.startswith("data: "):
            continue
        try:
            chunk = json.loads(s[6:])
            token = chunk["candidates"][0]["content"]["parts"][0].get("text", "")
            if token:
                yield token
        except (json.JSONDecodeError, KeyError, IndexError):
            pass


def stream_llm(system: str, user: str) -> Iterator[str]:
    """Yield raw text tokens as they arrive. Prefers OpenRouter, falls back to Gemini."""
    if os.environ.get("OPENROUTER_API_KEY"):
        try:
            yield from _stream_openrouter(system, user)
            return
        except LLMError:
            pass
    if os.environ.get("GEMINI_API_KEY"):
        yield from _stream_gemini(system, user)
        return
    raise LLMError("No LLM provider configured. Set OPENROUTER_API_KEY or GEMINI_API_KEY.")


#  Public entry point
def call_llm(system: str, user: str, **kwargs) -> LLMResponse:
    """
    Call whichever provider is configured. Prefer OpenRouter, then Gemini.
    Raise LLMError if neither is available or both fail.
    """
    last_err: Optional[Exception] = None
    if os.environ.get("OPENROUTER_API_KEY"):
        try:
            return _call_openrouter(system, user, **kwargs)
        except LLMError as e:
            last_err = e
    if os.environ.get("GEMINI_API_KEY"):
        try:
            return _call_gemini(system, user, **kwargs)
        except LLMError as e:
            last_err = e
    if last_err:
        raise last_err
    raise LLMError(
        "No LLM provider configured. Set OPENROUTER_API_KEY or GEMINI_API_KEY."
    )
