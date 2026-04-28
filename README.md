> **Mumzworld AI Engineering Intern — Track A take-home.**

---

## 🎥 Demo

**[▶ Watch the 3-minute walkthrough on Loom](https://www.loom.com/share/6124a9d0666148a7b26dbd66f5b48adf)**

Shows 3 inputs end-to-end including a refusal case (out-of-scope query → structured fallback).

A multilingual, RAG-grounded gift-recommendation prototype for Mumzworld.
Natural-language input in English or Arabic →  3–5 grounded, age-appropriate,
budget-respecting product picks with reasoning and a confidence score, in
both English **and** native Arabic, validated against a strict JSON schema.


---

## 1. Quick start  (clone → first output in under 5 minutes)

```bash
# 1. clone & enter
git clone <this-repo>.git
cd gift-finder

# 2. (recommended) virtualenv
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate

# 3. install
pip install -r requirements.txt

# 4. set ONE LLM provider key
cp .env.example .env
# then edit .env and put your free key in either:
#   OPENROUTER_API_KEY=...   (https://openrouter.ai/ — has free models)
#   GEMINI_API_KEY=...       (https://aistudio.google.com/app/apikey, also free)

# 5. run the UI
streamlit run app.py

#    OR run from the CLI:
python cli.py "{In put }"

#    OR run the eval suite:
python evals.py
```

The first call will download the multilingual embedding model
(~120 MB, one-time, CPU-only). After that, cold start is **<2 seconds**.

---

## 2. What it does

**Input**  (any language, any flavour):

```text
Gift for a mom with a 6-month-old under ₹2000
```

**Output**  (strict JSON, EN and AR, side-by-side):

```json
{
  "query": "Gift for a mom with a 6-month-old under ₹2000",
  "language": "en",
  "recommendations": [
    {
      "product_name": "Postpartum Recovery Gift Box",
      "price": "₹1899",
      "reason": "A thoughtful self-care bundle for a recovering new mom — peri-bottle, sitz soak, and nipple balm address real postpartum needs without exceeding ₹2000.",
      "confidence": 0.86
    },
    { "...": "..." },
    { "...": "..." }
  ],
  "fallback": false
}
```

If the query is out of scope, ambiguous, or impossible to satisfy, the system
returns a **structured refusal** instead of inventing answers:

```json
{
  "query": "Find me a gaming laptop",
  "language": "en",
  "recommendations": [],
  "fallback": true,
  "fallback_reason": "This query is outside the scope of a mom-and-baby gift finder."
}
```

---

## 3. Architecture

```
                    ┌──────────────────────────────┐
                    │  Natural-language query (EN  │
                    │  or AR), e.g. "...₹2000"     │
                    └──────────────┬───────────────┘
                                   ▼
        ┌──────────────────────────────────────────────────┐
        │  utils.intent.parse_intent                       │
        │   • budget       (₹/AED, Arabic-Indic digits)    │
        │   • baby_age     (months/years/newborn/toddler)  │
        │   • recipient    (mom / baby / both)             │
        │   • out-of-scope detection                       │
        │   • prompt-injection detection                   │
        └──────────────┬───────────────────────────────────┘
                       ▼
        ┌──────────────────────────────────────────────────┐
        │  rag.retriever.Retriever                         │
        │   • multilingual MiniLM embeddings (cached)      │
        │   • cosine similarity over 20-item catalog       │
        │   • HARD filters: budget, age (±2mo)             │
        │   • SOFT bonus: recipient match                  │
        │   →  top-K candidates                            │
        └──────────────┬───────────────────────────────────┘
                       ▼
        ┌──────────────────────────────────────────────────┐
        │  llm.generator (run twice: EN, then AR)          │
        │   1. build prompt with candidates as JSON        │
        │   2. call LLM (OpenRouter / Gemini, JSON mode)   │
        │   3. extract JSON  →  Pydantic validate          │
        │   4. retry once on parse / schema error          │
        │   5. drop hallucinated product_names             │
        │   6. fallback if <3 grounded items survive       │
        └──────────────┬───────────────────────────────────┘
                       ▼
              GiftFinderResponse(EN) + GiftFinderResponse(AR)
```

Two principles drive the design:

1. **Deterministic guardrails before LLM creativity.** Budget, age, and
   out-of-scope checks run in plain Python. The LLM never gets the chance
   to forget them. This is what stops "I asked for ₹2000 and you suggested
   a ₹5000 stroller."
2. **The LLM may only choose, not invent.** The prompt instructs it to
   recommend *only* products from the candidate list, and a post-LLM check
   drops anything that doesn't match a real catalog name. Hallucinations
   are caught, not hoped against.

---

## 4. Repo layout

```
gift-finder/
├── data/
│   └── products.json       # 20-item mock catalog (mom + baby, INR + AED)
├── rag/
│   ├── embeddings.py       # sentence-transformers wrapper + disk cache
│   └── retriever.py        # cosine search + hard filters + soft re-rank
├── llm/
│   ├── client.py           # OpenRouter (primary) + Gemini (fallback)
│   ├── prompts.py          # EN + AR system prompts (NOT translations)
│   └── generator.py        # full pipeline + retry + grounding check
├── utils/
│   ├── schema.py           # Pydantic GiftFinderResponse (strict)
│   ├── intent.py           # deterministic intent parser (EN + AR)
│   └── language.py         # lightweight script-based lang detector
├── cli.py                  # CLI alternative
├── evals.py                # 14 test cases + 5-dim scoring
├── requirements.txt
├── .env.example
├── README.md               # ← you are here
├── EVALS.md                # eval rubric, results, known failure modes
└── TRADEOFFS.md            # why this design, what was cut, what's next
```

---

## 5. Model choice

| Layer | Choice | Why |
| --- | --- | --- |
| **Embeddings** | `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers) | Speaks both English and Arabic out of the box — non-negotiable for the spec. ~120 MB, CPU-only, <500 ms per encode. The catalog is 20 items, so retrieval quality is bottlenecked by the prompt grounding, not by a fancier embedder. |
| **LLM (primary)** | `google/gemini-2.0-flash-exp:free` via OpenRouter | Free, fast (~1–2 s), reliable JSON-mode, handles Arabic well. Recommended in the brief. |
| **LLM (fallback)** | `gemini-2.0-flash` via Google AI Studio | Same model family without the OpenRouter routing layer — useful when OpenRouter rate-limits free traffic. |
| **Schema** | Pydantic v2 | Single source of truth for the JSON contract. Custom validators reject `fallback=True with recommendations`, `confidence > 1.0`, and `<3 recs without fallback`. |

You can swap the model with a single env var (`OPENROUTER_MODEL`) — Llama 3.3
70B and Qwen 2.5 72B (both free on OpenRouter) also work but are noticeably
slower on Arabic.

---

## 6. Tradeoffs and design decisions

These belong to a separate file: see [`TRADEOFFS.md`](./TRADEOFFS.md). Short version:

- **Hard constraints in code, not prompts.** Trusting a 70B model to
  multiply prices vs. budget is how you get an angry user. The retriever
  filters by budget and age *before* the LLM ever sees a candidate.
- **Two LLM calls (EN + AR), not one.** A single bilingual call yields
  Arabic that reads like a literal translation. Two grounded calls produce
  natural Arabic copy with idiomatic phrasing — what the brief explicitly
  asks for.
- **No vector DB.** FAISS / Chroma / pgvector add deps for zero benefit on
  a 20-item catalog. NumPy cosine sim runs in microseconds.
- **Embeddings cached to disk.** First run downloads + indexes; subsequent
  runs start in <1 s.
- **Confidence is the LLM's, not a similarity score.** Similarity tells
  you "this is in the right neighbourhood." Confidence is the model's
  judgment of how well the *specific* candidate fits the *specific* query
  given its constraints. We surface both in the debug panel; only confidence
  goes in the final JSON.

---

## 7. Failure modes (known and handled)

| Failure | How the system handles it |
| --- | --- |
| **Hallucinated product** | Post-LLM check drops any `product_name` not in the catalog. If <3 survive → structured fallback. |
| **Malformed JSON** | First retry includes a corrective hint in the user prompt ("your previous output was not valid JSON…"). Second failure → structured error response. |
| **Schema violation** (e.g. confidence=1.5, only 2 recs, fallback+recs) | Pydantic raises; same retry-then-fallback path. |
| **Budget too tight** | Retriever returns 0 candidates → fallback before LLM is called (saves tokens, never lies). |
| **Out-of-scope query** ("gaming laptop", "gift for my dog") | `parse_intent` flags `in_scope=False` → fallback, no LLM call. |
| **Empty / whitespace query** | Same as above. |
| **Prompt injection** ("ignore previous instructions") | Sniff regex flags it; system continues but the system prompt's "MAY ONLY recommend products from CANDIDATES" rule means the worst case is a list of catalog products, not a derailed model. |
| **Provider timeout / 5xx** | `LLMError` raised; if both providers configured, the second is tried; otherwise structured fallback. |
| **Arabic copy reading like translation** | We do **not** translate the EN response — we run a separate Arabic generation grounded on the same candidates with a natively-written Arabic system prompt. |

---

## 8. Claude & AI usage

- **Claude as coding assistant** — used throughout implementation for boilerplate, schema design, and Arabic prompt drafting; all architectural decisions (RAG-before-LLM, two separate generation calls, post-LLM grounding check) were made and defended manually.

- **Sub-agents for testing & evaluation** — a `failure_agent` sub-agent was prompted with adversarial queries (out-of-scope, budget-too-tight, prompt injection, Arabic input) to stress-test fallback paths before the eval suite was formalised; a `json_agent` was used to verify schema compliance on raw LLM outputs.

- **Mock data creation** — `data/products.json` (20-item catalog, dual INR/AED pricing, age ranges, tags) was generated with Claude given a seed set of real Mumzworld product categories; fields were hand-verified for price plausibility and age-range correctness.

- **Tool selection** — Claude helped evaluate embedding model options (`paraphrase-multilingual-MiniLM-L12-v2` chosen for multilingual EN+AR support at ~120 MB CPU-only) and surfaced the `responseSchema` + `thinkingBudget: 0` fix for Gemini 2.5-flash's thinking-token truncation issue.

- **Arabic reasoning** — the Arabic system prompt (`SYSTEM_AR` in `llm/prompts.py`) was drafted with Claude to produce *native* Arabic copy rather than translated English; two separate LLM calls (one per language) were chosen specifically because a single bilingual call yields Arabic that reads like a literal translation.

- **Latency improvements** — Claude identified that `hash()` randomisation was creating a new embeddings `.pkl` on every process restart (confirmed by 7 stale cache files); switching to `hashlib.md5` cut cold-start from ~30 s to <1 s. Disabling thinking (`thinkingBudget: 0`) on Gemini 2.5-flash removed the reasoning overhead that was truncating JSON output at 116 chars.

---

## 9. Eval results (summary)

See [`EVALS.md`](./EVALS.md) for the full rubric, all 14 cases, and
per-case pass/fail breakdowns. Headline numbers:

| Metric | Pass rate |
| --- | --- |
| JSON validity | 100 % |
| Grounding (no hallucinations) | 100 % |
| Budget compliance (when budget specified) | 100 % |
| Age compatibility (±2 months) | 100 % |
| Correct fallback on adversarial inputs | 100 % |

The few "failures" we saw during iteration were budget violations from the
LLM proposing items outside the candidate list — which the post-LLM
grounding check now catches. After that fix the system is at 100 % on the
hard checks for the 14 documented cases.

---

## ⏱ Development Time

The core system was designed and implemented in approximately 4 hours,
with additional time spent refining evaluation, testing, and documentation.