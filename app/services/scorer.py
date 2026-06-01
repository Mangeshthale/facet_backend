"""
backend/app/services/scorer.py

Builds few-shot prompts for each micro-batch of facets and
parses the structured JSON response from the LLM.

Key design choices:
1. Few-shot prompting (NOT one-shot) — satisfies Hard Constraint #1
2. Signed -2..+2 scale — more intuitive than 0-4
3. JSON-mode output with explicit fallback parsing
4. Per-facet confidence extracted from response or estimated from
   the model's own self-report (proxy for logit confidence)
"""

from __future__ import annotations
# from curses import raw
import json
import re
import asyncio
import logging
from typing import Optional
# from urllib import response

from app.services.llm_client import ollama
from app.core.config import settings

logger = logging.getLogger(__name__)


# ── Prompt templates keyed by category ───────────────────────────────────────

SYSTEM_PROMPT = """You are an expert conversation analyst. Your job is to score
a conversation turn on specific psychological, linguistic, and behavioral facets.

SCORING SCALE (use ONLY these integers):
  -2 = Strongly absent / strongly negative expression of this trait
  -1 = Mildly absent / weak expression
   0 = Neutral, unclear, or insufficient signal in this turn
   1 = Mildly present / moderate expression
   2 = Strongly present / dominant expression

CRITICAL RULE — score and reasoning must agree:
  If your reasoning says the trait IS present → score must be +1 or +2
  If your reasoning says the trait is ABSENT  → score must be -1 or -2
  If your reasoning says unclear/no evidence  → score must be 0
  NEVER give a negative score with reasoning that describes presence of the trait.

RULES:
- Score ONLY what is observable in the provided text
- If the conversation does not provide direct evidence for a facet,
  you MUST assign score = 0. Do NOT infer absence. Do NOT infer presence.
  Lack of evidence means score = 0.
- Confidence is a float 0.0–1.0 (1.0 = you are certain)
- Reasoning must be one sentence max
- Return ONLY valid JSON, no markdown, no preamble"""


FEW_SHOT_EXAMPLES = """
EXAMPLE 1:
Turn: "I absolutely refuse to listen to this nonsense. You have no idea what you're talking about."

Scores for [Hostility, Openness, Assertiveness]:
{
  "Hostility": {"score": 2, "confidence": 0.95, "reasoning": "Strong dismissal and insult indicate high hostility."},
  "Openness": {"score": -2, "confidence": 0.90, "reasoning": "Speaker explicitly refuses engagement and new perspectives."},
  "Assertiveness": {"score": 2, "confidence": 0.88, "reasoning": "Forceful refusal and declarative statements show high assertiveness."}
}

EXAMPLE 2:
Turn: "That's a good point. I hadn't thought of it that way — maybe you're right."

Scores for [Hostility, Openness, Assertiveness]:
{
  "Hostility": {"score": -2, "confidence": 0.92, "reasoning": "Acknowledgment and agreement show no hostility whatsoever."},
  "Openness": {"score": 2, "confidence": 0.91, "reasoning": "Speaker explicitly updates their view based on new information."},
  "Assertiveness": {"score": -1, "confidence": 0.72, "reasoning": "Hedging language ('maybe') indicates mild lack of assertiveness."}
}

EXAMPLE 3 — CRITICAL: score must match reasoning direction:
Turn: "I feel like nothing I do is ever good enough."

Scores for [Perfectionism, Contentment]:
{
  "Perfectionism": {"score": 2, "confidence": 0.90, "reasoning": "Feeling nothing is good enough directly signals high perfectionism."},
  "Contentment": {"score": -2, "confidence": 0.92, "reasoning": "Explicit dissatisfaction with everything indicates very low contentment."}
}
NOTE: Perfectionism scores +2 because the reasoning identifies it as PRESENT.
"""


SCORING_PROMPT_TEMPLATE = """{system}

{few_shot}

---
NOW SCORE THE FOLLOWING:

Conversation context:
{context}

Turn to score (turn {turn_idx}):
"{turn_text}"

Facets to score:
{facet_list}

Return ONLY a JSON object with this exact structure:
{{
  "FacetName": {{"score": <int -2 to 2>, "confidence": <float 0-1>, "reasoning": "<one sentence>"}},
  ...
}}
"""


def _build_context_str(conversation: list[dict], turn_idx: int) -> str:
    """Format the conversation as readable context."""
    lines = []
    for i, turn in enumerate(conversation):
        marker = ">>> [THIS TURN] <<<" if i == turn_idx else ""
        lines.append(f"{turn['role'].upper()}: {turn['content']} {marker}".strip())
    return "\n".join(lines)


def _build_facet_list(batch: list[dict]) -> str:
    """Format batch of facets as a numbered list for the prompt."""
    items = []
    for f in batch:
        polarity_hint = "(note: high score = negative trait)" if f["polarity"] == "negative" else ""
        items.append(f"- {f['facet_name']} {polarity_hint}".strip())
    return "\n".join(items)


def _parse_llm_response(raw: str, batch: list[dict]) -> dict:
    """
    Parse the LLM's JSON response.
    Returns dict of facet_name → {score, confidence, reasoning}.
    Falls back to score=0, confidence=0.0 for any missing facets.
    """
    # Strip markdown fences if the model added them despite instructions
    cleaned = re.sub(r"```json|```", "", raw).strip()

    # Find the first {...} block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        logger.warning("No JSON found in LLM response. Using fallback zeros.")
        return _fallback_scores(batch)

    try:
        raw_json = match.group()
        # Fix trailing commas before closing brace — common Qwen artifact
        raw_json = re.sub(r",\s*\}", "}", raw_json)
        raw_json = re.sub(r",\s*\]", "]", raw_json)
        logger.info("RAW MODEL OUTPUT:")
        logger.info(raw_json)
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error: {e}. Using fallback zeros.")
        return _fallback_scores(batch)

    # Validate and normalise each entry
    result = {}
    for facet in batch:
        name = facet["facet_name"]
        if name in parsed:
            entry = parsed[name]
            score = int(entry.get("score", 0))
            score = max(-2, min(2, score))   # clamp to valid range
            confidence = float(entry.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            result[name] = {
                "score": score,
                "confidence": confidence,
                "reasoning": str(entry.get("reasoning", ""))[:200],
                "facet_id": facet["facet_id"],
                "category": facet["category"],
                "polarity": facet["polarity"],
            }
        else:
            # Model skipped this facet — use neutral fallback
            result[name] = _make_fallback(facet)

    return result


def _fallback_scores(batch: list[dict]) -> dict:
    return {f["facet_name"]: _make_fallback(f) for f in batch}


def _make_fallback(facet: dict) -> dict:
    return {
        "score": 0,
        "confidence": 0.0,
        "reasoning": "Model did not return a score for this facet.",
        "facet_id": facet["facet_id"],
        "category": facet["category"],
        "polarity": facet["polarity"],
    }


async def score_batch(
    batch: list[dict],
    conversation: list[dict],
    turn_idx: int,
) -> dict:
    """
    Score one micro-batch of facets for the given conversation turn.
    Returns dict of facet_name → score dict.
    """
    turn_text = conversation[turn_idx]["content"]
    context_str = _build_context_str(conversation, turn_idx)
    facet_list_str = _build_facet_list(batch)

    prompt = SCORING_PROMPT_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        few_shot=FEW_SHOT_EXAMPLES,
        context=context_str,
        turn_idx=turn_idx,
        turn_text=turn_text,
        facet_list=facet_list_str,
    )

    try:
        raw_response = await ollama.generate(prompt)
        return _parse_llm_response(raw_response, batch)
    except Exception as e:
        logger.error(f"LLM call failed for batch starting {batch[0]['facet_name']}: {e}")
        return _fallback_scores(batch)
