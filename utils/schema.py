"""
Strict output schema, enforced by Pydantic.

Every recommendation the system returns is validated against `GiftFinderResponse`.
If the LLM produces output that does not satisfy this schema, the generator
retries once and then falls back to a structured error response.
"""

from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class Recommendation(BaseModel):
    """A single product recommendation with reasoning and confidence."""

    product_name: str = Field(..., min_length=1, description="Name of the product (must match catalog).")
    price: str = Field(..., min_length=1, description="Display price including currency, e.g. '₹1599' or 'AED 65'.")
    reason: str = Field(..., min_length=10, description="Why this product fits the query — concise, specific.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence in [0,1].")

    @field_validator("reason")
    @classmethod
    def _reason_not_generic(cls, v: str) -> str:
        # Cheap heuristic: reject obviously empty / placeholder reasons.
        stripped = v.strip()
        if len(stripped) < 10:
            raise ValueError("reason is too short")
        return stripped


class GiftFinderResponse(BaseModel):
    """Top-level response shape — matches the spec exactly."""

    query: str = Field(..., min_length=1)
    language: Literal["en", "ar"]
    recommendations: List[Recommendation] = Field(default_factory=list)
    fallback: bool = False
    fallback_reason: Optional[str] = Field(
        default=None,
        description="Human-readable explanation when fallback=True. Omitted on success.",
    )

    @field_validator("recommendations")
    @classmethod
    def _max_five(cls, v: List[Recommendation]) -> List[Recommendation]:
        if len(v) > 5:
            raise ValueError("at most 5 recommendations allowed")
        return v

    def model_post_init(self, __context) -> None:
        # Invariant: if fallback is False there must be 3-5 recommendations.
        # If fallback is True the recommendations list must be empty.
        if self.fallback:
            if self.recommendations:
                raise ValueError("fallback=True must have empty recommendations")
            if not self.fallback_reason:
                raise ValueError("fallback=True requires fallback_reason")
        else:
            if not (3 <= len(self.recommendations) <= 5):
                raise ValueError(
                    f"non-fallback responses must have 3-5 recommendations, got {len(self.recommendations)}"
                )
