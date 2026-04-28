"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv
load_dotenv()

from llm.generator import GiftFinder, GenerationResult
from llm.client import stream_llm
from llm.prompts import build_en_prompt, build_ar_prompt
from rag.retriever import Retriever
from utils.intent import parse_intent


def _print_human(result: GenerationResult) -> None:
    print()
    print("─" * 70)
    print(f"Query:         {result.intent.raw_query}")
    print(f"Parsed intent: budget={result.intent.budget} {result.intent.currency}, "
          f"age={result.intent.baby_age_months}mo, "
          f"recipient={result.intent.recipient}, "
          f"in_scope={result.intent.in_scope}")
    print(f"Retrieved:     {result.debug.get('n_candidates', 0)} candidates "
          f"({', '.join(result.debug.get('candidate_ids', []))})")
    print("─" * 70)

    for label, resp in (("English", result.en), ("Arabic", result.ar)):
        print(f"\n[{label}]")
        if resp.fallback:
            print(f"  FALLBACK: {resp.fallback_reason}")
            continue
        for r in resp.recommendations:
            print(f"  • {r.product_name}  —  {r.price}  (conf {r.confidence:.2f})")
            print(f"      {r.reason}")
    print()


def _print_json(result: GenerationResult) -> None:
    """Print EN and AR responses in the schema format."""
    print("[English]")
    print(json.dumps(result.en.model_dump(exclude_none=True), ensure_ascii=False, indent=2))
    print("\n[Arabic]")
    print(json.dumps(result.ar.model_dump(exclude_none=True), ensure_ascii=False, indent=2))


def _run_streaming(query: str) -> None:
    """Stream tokens to stdout as they arrive, then show the parsed summary."""
    intent = parse_intent(query)
    retriever = Retriever()
    candidates = retriever.retrieve(query, intent, top_k=8)

    currency_pref = intent.currency if intent.currency != "UNKNOWN" else "INR"

    print(f"\nIntent: budget={intent.budget} {intent.currency}, "
          f"age={intent.baby_age_months}mo, recipient={intent.recipient}\n")

    # ── English ──────────────────────────────────────────────────────────────
    print("[English — streaming]")
    system_en, user_en = build_en_prompt(query, candidates, currency_pref=currency_pref)
    en_chunks: list[str] = []
    for token in stream_llm(system_en, user_en):
        print(token, end="", flush=True)
        en_chunks.append(token)
    print()

    # ── Arabic ───────────────────────────────────────────────────────────────
    print("\n[Arabic — streaming]")
    system_ar, user_ar = build_ar_prompt(query, candidates, currency_pref=currency_pref)
    ar_chunks: list[str] = []
    for token in stream_llm(system_ar, user_ar):
        print(token, end="", flush=True)
        ar_chunks.append(token)
    print("\n")


def main() -> int:
    p = argparse.ArgumentParser(description="Gift Finder CLI")
    p.add_argument("query", nargs="?", help="Natural-language gift query")
    p.add_argument("--human", action="store_true", help="Human-readable output instead of JSON")
    p.add_argument("--stream", action="store_true", help="Stream tokens as they arrive")
    p.add_argument("--interactive", action="store_true", help="REPL mode")
    args = p.parse_args()

    if args.stream:
        if args.interactive:
            print("Gift Finder streaming REPL — type 'quit' to exit.")
            while True:
                try:
                    q = input("\n> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    return 0
                if q.lower() in {"q", "quit", "exit"}:
                    return 0
                if q:
                    _run_streaming(q)
        if not args.query:
            p.print_help()
            return 2
        _run_streaming(args.query)
        return 0

    finder = GiftFinder()

    if args.interactive:
        print("Gift Finder REPL — type 'quit' to exit.")
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if q.lower() in {"q", "quit", "exit"}:
                return 0
            if not q:
                continue
            result = finder.generate(q)
            _print_human(result) if args.human else _print_json(result)

    if not args.query:
        p.print_help()
        return 2

    result = finder.generate(args.query)
    _print_human(result) if args.human else _print_json(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
