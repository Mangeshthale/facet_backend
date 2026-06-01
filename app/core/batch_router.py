"""
backend/app/core/batch_router.py

Splits the full facet list into micro-batches of ~N facets each.
This is the KEY architectural component that lets the system scale
to 5000+ facets without any redesign — you just get more batches,
each handled by the same worker code.

Design decisions:
- Groups facets by category within batches (similar facets together
  → better LLM scoring coherence per call)
- Batch size is configurable via settings.FACETS_PER_BATCH
- Returns list of batches; caller can process them in parallel
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

from app.core.config import settings


# ── Facet store (loaded once at startup) ──────────────────────────────────────

_FACETS: list[dict] = []


def load_facets() -> list[dict]:
    """Load enriched facets JSON. Called once at app startup."""
    global _FACETS
    if _FACETS:
        return _FACETS
    path: Path = settings.FACETS_JSON
    if not path.exists():
        raise FileNotFoundError(
            f"Facets JSON not found at {path}. "
            "Run: python scripts/preprocess_facets.py"
        )
    with open(path, encoding="utf-8") as f:
        _FACETS = json.load(f)
    return _FACETS


def get_facets(
    categories: Optional[list[str]] = None,
    facet_ids: Optional[list[str]] = None,
) -> list[dict]:
    """Return filtered facets based on request parameters."""
    facets = load_facets()
    if facet_ids:
        facets = [f for f in facets if f["facet_id"] in facet_ids]
    elif categories:
        facets = [f for f in facets if f["category"] in categories]
    return facets


def make_batches(
    facets: list[dict],
    batch_size: int | None = None,
) -> list[list[dict]]:
    """
    Split facets into micro-batches.

    Strategy: sort by category first so each batch is thematically
    coherent. This improves scoring quality because the LLM can
    reason about related traits together.

    Args:
        facets: list of facet dicts
        batch_size: override settings.FACETS_PER_BATCH

    Returns:
        List of batches, each a list of facet dicts.
    """
    size = batch_size or settings.FACETS_PER_BATCH

    # Sort by category for coherent batches
    sorted_facets = sorted(facets, key=lambda f: (f["category"], f["facet_name"]))

    batches = []
    for i in range(0, len(sorted_facets), size):
        batches.append(sorted_facets[i : i + size])

    return batches


def get_all_categories() -> list[str]:
    """Return sorted list of unique categories."""
    facets = load_facets()
    return sorted(set(f["category"] for f in facets))
