"""
backend/app/api/facets.py

GET /facets — list all enriched facets with metadata
"""

from fastapi import APIRouter, Query
from typing import Optional

from app.models.schemas import FacetMeta, FacetsResponse
from app.core.batch_router import get_facets, get_all_categories

router = APIRouter()


@router.get("/facets", response_model=FacetsResponse)
async def list_facets(
    category: Optional[str] = Query(default=None, description="Filter by category"),
    difficulty: Optional[str] = Query(default=None, description="easy | medium | hard"),
):
    """Return all facets with their metadata. Use query params to filter."""
    categories = [category] if category else None
    facets = get_facets(categories=categories)

    if difficulty:
        facets = [f for f in facets if f["eval_difficulty"] == difficulty]

    facet_models = [FacetMeta(**f) for f in facets]

    return FacetsResponse(
        total=len(facet_models),
        facets=facet_models,
        categories=get_all_categories(),
    )
