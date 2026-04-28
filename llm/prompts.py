from __future__ import annotations
import json
from typing import List
from rag.retriever import Candidate


#  System messages 

SYSTEM_EN = """You are a helpful gift-finder for the Mumzworld marketplace.
Mumzworld serves mothers across the GCC and India.

You will be given:
  1. A user's natural-language gift query.
  2. A short list of CANDIDATE products retrieved from our catalog.

Your job is to pick the 3-5 best candidates and explain why each fits the query.

HARD RULES (do not violate, ever):
  - You MAY ONLY recommend products from the CANDIDATES list. Do not invent products.
  - Use the EXACT product_name from the candidate list. No paraphrasing.
  - Use the EXACT price string provided in the candidate list.
  - If fewer than 3 candidates fit the query well, set "fallback": true,
    leave "recommendations" as [], and set "fallback_reason" briefly.
  - Output MUST be a single JSON object matching the schema below — no prose,
    no markdown fences, no commentary outside the JSON.
  - Each "reason" must be 1-2 sentences, specific to the candidate, and
    grounded in its description/tags/age-range. No generic claims.
  - "confidence" is your honest estimate that this product fits the query
    (0.0-1.0). Use the full range; do not return 0.95 for everything.

SCHEMA:
{
  "query": "<echo the user's query>",
  "language": "en",
  "recommendations": [
    {"product_name": "...", "price": "...", "reason": "...", "confidence": 0.0}
  ],
  "fallback": false
}

If fallback=true, also include "fallback_reason": "<one sentence>".
"""


SYSTEM_AR = """أنتَ مساعد ذكي لاختيار الهدايا في متجر ممزورلد، الموجَّه للأمهات في الخليج.

سيُعطى لك:
  1. استفسار من المستخدمة بلغة طبيعية.
  2. قائمة قصيرة بمنتجات مرشّحة من كتالوجنا.

اختاري من ٣ إلى ٥ منتجات الأنسب، واشرحي بإيجاز سبب اختيار كل منها.

قواعد صارمة:
  - لا تقترحي إلا منتجات موجودة في قائمة المرشّحات. لا تختلقي منتجات.
  - استخدمي اسم المنتج كما هو في القائمة المرشّحة، بلا تغيير.
  - استخدمي السعر كما هو وارد في القائمة.
  - إن لم تجدي ٣ منتجات مناسبة، اضبطي "fallback": true وأعيدي قائمة فارغة،
    مع ذكر السبب في "fallback_reason".
  - الردّ يجب أن يكون كائن JSON واحد فقط، بدون أي شرح خارجه ولا علامات Markdown.
  - اكتبي "reason" بأسلوب عربيّ طبيعي وودود — وكأنك تنصحين صديقة، لا
    كترجمة حرفية. جملة أو جملتين، محدّدتان وغير عامّتين.
  - "confidence" قيمة بين 0.0 و 1.0 تعبّر بصدق عن مدى ملاءمة المنتج للطلب.

البنية المطلوبة:
{
  "query": "<نسخ الاستفسار كما ورد>",
  "language": "ar",
  "recommendations": [
    {"product_name": "...", "price": "...", "reason": "...", "confidence": 0.0}
  ],
  "fallback": false
}

في حالة fallback=true، أضيفي "fallback_reason".
"""


# Candidate formatting

def _format_candidates(candidates: List[Candidate], currency_pref: str = "INR") -> str:
    """
    Render the candidate list as a JSON block the model can ground on.

    We include similarity scores so the LLM has a hint about ranking, but
    final selection is the LLM's call within the schema rules.
    """
    rows = []
    for c in candidates:
        p = c.product
        if currency_pref == "AED":
            price_str = f"AED {p['price_aed']}"
        else:
            price_str = f"₹{p['price_inr']}"
        rows.append({
            "product_name": p["name"],
            "price": price_str,
            "category": p["category"],
            "for_recipient": p["for_recipient"],
            "age_range_months": (
                f"{p['age_min_months']}-{p['age_max_months']}"
                if p.get("age_min_months") is not None
                else "any"
            ),
            "tags": p.get("tags", []),
            "description": p["description"],
            "retrieval_similarity": round(c.similarity, 3),
        })
    return json.dumps(rows, ensure_ascii=False, indent=2)


def build_en_prompt(query: str, candidates: List[Candidate], currency_pref: str = "INR") -> tuple[str, str]:
    """Returns (system, user) prompts for the English generation."""
    user = f"""USER QUERY:
{query}

CANDIDATES (JSON list, retrieved from catalog):
{_format_candidates(candidates, currency_pref)}

Now produce the JSON response."""
    return SYSTEM_EN, user


def build_ar_prompt(query: str, candidates: List[Candidate], currency_pref: str = "INR") -> tuple[str, str]:
    """Returns (system, user) prompts for the Arabic generation."""
    # We pass the original query (whatever language) plus the candidates.
    # The system prompt requires native Arabic output regardless of input lang.
    user = f"""استفسار المستخدمة:
{query}

المنتجات المرشّحة (JSON من الكتالوج):
{_format_candidates(candidates, currency_pref)}

الآن أنتجي ردّ JSON."""
    return SYSTEM_AR, user
