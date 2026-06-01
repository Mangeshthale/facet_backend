"""
backend/app/services/aggregator.py

Runs all batch workers in parallel (up to MAX_CONCURRENT_WORKERS at once)
and merges the results into a single scored dict.

This is where scalability lives:
  300 facets / 20 per batch = 15 batches
  With 3 concurrent workers → ~5 rounds of parallel calls
  With 10 workers → ~2 rounds
  At 5000 facets → 250 batches, same code, just more rounds
"""

from __future__ import annotations
import asyncio
import time
import logging
from typing import Optional

from app.services.scorer import score_batch
from app.core.batch_router import make_batches, get_facets
from app.core.config import settings

logger = logging.getLogger(__name__)


async def run_evaluation(
    conversation: list[dict],
    turn_index: int = -1,
    categories: Optional[list[str]] = None,
    facet_ids: Optional[list[str]] = None,
) -> dict:
    """
    Main entry point. Returns:
    {
        "scores": {facet_name: {score, confidence, reasoning, ...}},
        "meta": {batches, processing_time_ms, ...}
    }
    """
    start_ms = time.time()

    # Resolve which turn to evaluate
    if turn_index < 0:
        turn_index = len(conversation) + turn_index  # -1 → last turn

    # Get and batch the facets
    facets = get_facets(categories=categories, facet_ids=facet_ids)
    batches = make_batches(facets)

    logger.info(
        f"Evaluating turn {turn_index} | "
        f"{len(facets)} facets | {len(batches)} batches | "
        f"model={settings.MODEL_NAME}"
    )

    # Semaphore limits how many LLM calls run at once
    semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_WORKERS)

    async def bounded_score(batch):
        async with semaphore:
            return await score_batch(batch, conversation, turn_index)

    # Launch all batches concurrently (bounded by semaphore)
    tasks = [bounded_score(batch) for batch in batches]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Merge all batch results into one flat dict
    merged_scores: dict = {}
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Batch {i} failed: {result}")
            # Fallback zeros for the failed batch already handled in scorer.py
            continue
        merged_scores.update(result)

    elapsed_ms = int((time.time() - start_ms) * 1000)

    turn_text = conversation[turn_index]["content"]

    return {
        "scores": merged_scores,
        "meta": {
            "model": settings.MODEL_NAME,
            "total_facets_scored": len(merged_scores),
            "facets_requested": len(facets),
            "processing_time_ms": elapsed_ms,
            "batches_used": len(batches),
            "turn_evaluated": turn_text[:200],
        },
    }
