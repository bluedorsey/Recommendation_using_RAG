"""Command-line interface."""

from __future__ import annotations
import argparse
import json
import sys
from llm.generator import GiftFinder, GenerationResult

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
    payload = {
        "en": result.en.model_dump(exclude_none=True),
        "ar": result.ar.model_dump(exclude_none=True),
        "debug": result.debug,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> int:
    p = argparse.ArgumentParser(description="Gift Finder CLI")
    p.add_argument("query", nargs="?", help="Natural-language gift query")
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    p.add_argument("--interactive", action="store_true", help="REPL mode")
    args = p.parse_args()

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
            _print_json(result) if args.json else _print_human(result)
        # unreachable

    if not args.query:
        p.print_help()
        return 2

    result = finder.generate(args.query)
    _print_json(result) if args.json else _print_human(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
