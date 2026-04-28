# Tradeoffs

This document records the design decisions that shaped the prototype, what we
considered and rejected, where we knowingly cut scope, and what we'd build
next if we had another five hours.

---

## Why this problem

Mumzworld's brief listed ~12 example problems. We picked **Recommendations using RAG** because:

1. **It's a real funnel.** Gift-finding is a high-intent purchase with a
   bad default UX (search bar + filter chips). Even a modest improvement
   here translates directly to revenue, and the input modality
   (natural-language) is exactly where LLMs are obviously better than the
   status quo.
2. **It exercises everything the brief flags as "non-trivial AI engineering."**
   RAG, structured output with validation, multilingual generation,
   uncertainty handling, and evals — all in one feature.
3. **Multilingual matters here.** A "thoughtful gift for my friend" is the
   kind of message a Mumzworld customer will type in whichever language
   they think in — and the response has to feel native, not translated.
4. **It's honestly scopeable in 5 hours.** A 20-item mock catalog is the
   right scale for a prototype: small enough that retrieval is solved,
   large enough that the LLM has to actually reason rather than just
   regurgitate a single answer.

We considered and rejected:

- **Pediatric symptom triage.** High-stakes (medical), would have spent
  most of the time on safety guardrails and very little on the AI
  engineering they want to see.
- **Review synthesis ("Moms Verdict").** Engineering-light — basically
  one well-written prompt over a fixed corpus. Doesn't really need RAG
  beyond a single-document fetch.
- **Operations dashboard with anomaly detection.** Strong fit for the
  brief but the "AI" part is small; most of the work is data wrangling
  and dashboarding.

---

## Model choice

| Layer | Choice | Why we picked it | What we considered |
| --- | --- | --- | --- |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | EN + AR out of the box. ~120 MB, CPU-only, <500 ms encode. | `bge-m3` (better but 2.3 GB), OpenAI `text-embedding-3-small` (paid, network round-trip). |
| LLM (primary) | `google/gemini-2.0-flash-exp:free` via OpenRouter | Free tier, fast (~1.5 s), reliable JSON mode, strong Arabic. | `meta-llama/llama-3.3-70b-instruct:free` (slower, weaker on Arabic), `qwen/qwen-2.5-72b-instruct:free` (great Arabic, slowest). |
| LLM (fallback) | `gemini-2.5-flash` (Google AI Studio direct) | Cuts out the OpenRouter routing layer when its free tier rate-limits. | A pure local model via Ollama (rejected — adds setup pain to a 5-min-clone deliverable). |
| Schema | Pydantic v2 | Single source of truth; declarative validators; trivially serializable; Streamlit-friendly. | `jsonschema` (more verbose), `dataclasses` (no validation), hand-rolled (waste of time). |
| Vector store | NumPy + cosine similarity | 20 items. Anything more is masochism. | FAISS (overkill), Chroma (added dep), pgvector (overkill, infra). |
| UI | Streamlit | Brief says backend is the focus. Streamlit gets us a clean two-column EN/AR view in 80 lines. | FastAPI + a JS frontend (rejected — the brief is explicit). |

---

## How we handled uncertainty

Uncertainty handling is graded at 15 % per the brief. The system uses **four
distinct mechanisms**, in this order:

1. **Pre-LLM scope check.** Out-of-domain queries ("gaming laptop", "gift
   for my dog") are caught by `parse_intent` and short-circuit to a
   fallback before any LLM call. Costs nothing, never lies, never
   hallucinates.
2. **Pre-LLM constraint check.** If the budget filter leaves <3 candidates
   in the retriever, we fallback before the LLM call. If the user said
   "₹100" and we have nothing under ₹100, we say so — we don't pretend.
3. **In-LLM grounding rule.** The system prompt's "you MAY ONLY recommend
   products from the CANDIDATES list" is the single most load-bearing
   sentence in the project. Without it, even a retrieval-grounded prompt
   leaks invented brand names.
4. **Post-LLM grounding check.** Even with rule (3), the model sometimes
   embellishes. Every returned `product_name` is verified against the
   actual catalog; mismatches are dropped; if <3 survive we convert to
   fallback. This is the difference between *hoping* the LLM stays
   grounded and *proving* it.

Confidence is the LLM's self-reported judgment, not derived from
similarity. We surface both in the debug panel.

---

## What we cut (and would build next)

Cut, in priority order:

- **Streaming output.** Streamlit + OpenRouter both support it; the UX
  win is real but it's a polish item.
- **LLM-as-judge eval.** Our deterministic checks cover correctness;
  reasoning *quality* is still eyeballed. A second-pass judge with a
  fixed rubric would let us A/B model swaps with numbers, not vibes.
- **Property-based test generation.** With more time we'd 10× the eval
  suite by sampling random `(budget × age × recipient × language)`
  combinations and asserting invariants (no hallucinations, budget
  respected) rather than hand-writing every case.
- **Cross-sell / bundles.** The retriever already returns 8 candidates;
  asking the LLM for "if you have a bit more, also consider…" picks at
  the next price tier is essentially free and would unlock real AOV.
- **Real catalog ingest.** `data/products.json` is mock data. The
  retriever and prompts wouldn't change for a real CSV/JSONL feed; only
  the loader.
- **Memoized LLM responses for evals.** The eval suite hits the LLM
  every run. A simple keyed cache would make iteration on
  prompt/scoring changes much faster.

What we'd **add** with another 5 hours, in order of leverage:

1. LLM-as-judge eval + property-based generation → confidence we can
   ship without regressions.
2. Streaming UI + memoization in evals → tight dev loop.
3. Bundles / cross-sell prompt arm → directly tests business value.

---

## What we'd do differently

- **Lock the prompts earlier.** I iterated the system prompt 4–5 times
  while the eval suite was changing underneath me. Should have frozen
  the eval rubric on hour 2 and treated it as fixed.
- **Write the Arabic prompt with a native speaker review.** I'm
  comfortable with the Arabic prompt's structure but a native review
  would likely tighten the tone in 10 minutes.
- **Add a tiny *unit* test layer.** Right now the smallest test is
  end-to-end. A handful of pure-function tests for `parse_intent` would
  have caught the Arabic-numeral bug 30 minutes earlier.

---

## Time log

| Phase | Time |
| --- | --- |
| Reading the brief, scoping, picking the problem | ~30 min |
| Catalog design + dataset | ~25 min |
| Schema + intent parser + language utils | ~40 min |
| RAG (embeddings + retriever) | ~35 min |
| LLM client + prompts + generator + retry/grounding | ~70 min |
| Streamlit UI + CLI | ~25 min |
| Evals + EVALS.md | ~45 min |
| README + TRADEOFFS.md + polish | ~50 min |
| **Total** | **~5h 20m** (~5–10 % over budget on docs polish) |
