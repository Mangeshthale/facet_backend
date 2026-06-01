"""
backend/app/models/schemas.py

All Pydantic v2 models used across the API.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


# ── Inbound ───────────────────────────────────────────────────────────────────

class ConversationTurn(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., min_length=1)


class EvaluateRequest(BaseModel):
    conversation: list[ConversationTurn] = Field(
        ..., min_length=1, description="Full conversation history"
    )
    turn_index: int = Field(
        default=-1,
        description="Which turn to score (-1 = last turn)"
    )
    facet_categories: Optional[list[str]] = Field(
        default=None,
        description="Filter to specific categories. None = score all."
    )
    facet_ids: Optional[list[str]] = Field(
        default=None,
        description="Filter to specific facet IDs. Overrides facet_categories."
    )

    model_config = {"json_schema_extra": {
        "example": {
            "conversation": [
                {"role": "user", "content": "I'm really frustrated with my manager."},
                {"role": "assistant", "content": "That sounds stressful. What happened?"},
            ],
            "turn_index": 0,
            "facet_categories": ["emotional", "social"],
        }
    }}


# ── Outbound ──────────────────────────────────────────────────────────────────

class FacetScore(BaseModel):
    score: int = Field(..., ge=-2, le=2, description="Score from -2 to +2")
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., description="One-sentence explanation")
    facet_id: str
    category: str
    polarity: str   # positive | negative


class EvaluationMetadata(BaseModel):
    model: str
    total_facets_scored: int
    facets_requested: int
    processing_time_ms: int
    batches_used: int
    turn_evaluated: str   # the actual text of the turn that was scored


class EvaluateResponse(BaseModel):
    scores: dict[str, FacetScore]   # key = facet_name
    metadata: EvaluationMetadata


# ── Facets endpoint ───────────────────────────────────────────────────────────

class FacetMeta(BaseModel):
    facet_id: str
    facet_name: str
    category: str
    polarity: str
    eval_difficulty: str
    required_context_turns: int
    is_observable: bool
    prompt_template_id: str
    score_min: int
    score_max: int


class FacetsResponse(BaseModel):
    total: int
    facets: list[FacetMeta]
    categories: list[str]
