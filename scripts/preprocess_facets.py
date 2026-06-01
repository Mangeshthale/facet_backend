"""
scripts/preprocess_facets.py

Run this FIRST before starting the backend.
It cleans the raw Facets_Assignment.csv and produces
data/processed/facets_enriched.csv with extra metadata columns.

Usage:
    python scripts/preprocess_facets.py
"""

import re
import csv
import json
import sys
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent
RAW_CSV  = ROOT / "data" / "raw" / "Facets_Assignment.csv"
OUT_CSV  = ROOT / "data" / "processed" / "facets_enriched.csv"
OUT_JSON = ROOT / "data" / "processed" / "facets_enriched.json"


# ── Category mapping (keyword → category) ───────────────────────────────────
CATEGORY_RULES = {
    "emotional": [
        "emotion", "affect", "mood", "feeling", "happiness", "sadness",
        "anxiety", "depression", "anger", "fear", "joy", "grief",
        "compassion", "empathy", "warmth", "hostility", "irritab",
        "merriness", "blissful", "discontentment", "moroseness",
        "enthusiasm", "contentment", "distress", "burnout",
    ],
    "cognitive": [
        "reasoning", "memory", "intelligence", "attention", "processing",
        "logic", "analytic", "critical", "spatial", "numerical",
        "working memory", "iq", "arithmetic", "comprehension",
        "synthesis", "judgment", "decision", "planning",
    ],
    "linguistic": [
        "sentence", "language", "spelling", "grammar", "brevity",
        "storytelling", "vocabulary", "verbal", "communication",
        "listening", "articula", "fluency", "discourse",
    ],
    "personality": [
        "openness", "conscientiousness", "extraversion", "agreeableness",
        "neuroticism", "assertiveness", "introvert", "hexaco",
        "big five", "enneagram", "mbti", "perceiving", "judging",
        "psychoticism", "impulsiv",
    ],
    "social": [
        "social", "collaboration", "leadership", "teamwork", "empathy",
        "relationship", "affiliation", "trust", "community",
        "peer", "volunteer", "cooperation", "civility",
    ],
    "safety": [
        "harm", "violence", "aggression", "danger", "risk",
        "self-harm", "hostility", "dishonesty", "deception",
        "manipulation", "drug", "substance",
    ],
    "spiritual": [
        "spiritual", "religious", "sufi", "buddhist", "islamic",
        "hindu", "jewish", "sikh", "kabbalah", "meditation",
        "prayer", "pilgrimage", "quran", "mantra", "holiness",
    ],
    "health": [
        "health", "sleep", "pain", "medical", "clinical",
        "metabolic", "dietary", "nutrition", "caffeine",
        "basophil", "hormonal", "genetic", "immune",
    ],
    "behavioral": [
        "behavior", "habit", "frequency", "activity", "exercise",
        "lifestyle", "routine", "procrastin", "compulsive",
        "avoidance", "seeking",
    ],
    "professional": [
        "work", "career", "skill", "leadership", "delegation",
        "meeting", "deadline", "productivity", "feedback",
        "training", "professional",
    ],
}

# ── Polarity mapping (high score = positive or negative?) ────────────────────
NEGATIVE_POLARITY_KEYWORDS = [
    "hostility", "aggression", "harm", "dishonesty", "manipulation",
    "depression", "anxiety", "burnout", "impulsivity", "addiction",
    "laziness", "sloth", "coarseness", "disrespect", "hatefulness",
    "brazenness", "passive-aggressive", "cantankerous", "martyrdom",
    "inefficiency", "inattentive", "unassertive", "immaturity",
    "impractical", "discontentment", "moroseness", "naivety",
    "acidity", "cunning", "prejudice", "ethnocentrism",
]

# ── Context length heuristic ─────────────────────────────────────────────────
LONG_CONTEXT_KEYWORDS = [
    "leadership", "relationship", "childhood", "perseverance",
    "life", "history", "orientation", "style", "pattern",
]


def clean_facet_name(raw: str) -> str:
    """Strip numeric prefixes, trailing colons, extra whitespace."""
    name = raw.strip()
    # Remove leading numbers like "800. " or "644. "
    name = re.sub(r"^\d+\.\s*", "", name)
    # Remove trailing colon
    name = name.rstrip(":")
    # Collapse multiple spaces
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def is_valid_facet(name: str) -> bool:
    """Filter out category headers and junk rows."""
    if len(name) < 4:
        return False
    # These are section headers, not scorable facets
    header_patterns = [
        r"^(facets?|components?|subcomponents?|parameters?|styles?|behaviors?|types?|end points?)$",
        r"^(additional|moral and ethical|behavioral tendencies)",
        r"facet$",   # ends with just "facet"
    ]
    lower = name.lower()
    for pattern in header_patterns:
        if re.search(pattern, lower):
            return False
    return True


def get_category(name: str) -> str:
    lower = name.lower()
    for category, keywords in CATEGORY_RULES.items():
        for kw in keywords:
            if kw in lower:
                return category
    return "general"


def get_polarity(name: str) -> str:
    lower = name.lower()
    for kw in NEGATIVE_POLARITY_KEYWORDS:
        if kw in lower:
            return "negative"
    return "positive"


def get_difficulty(name: str) -> str:
    lower = name.lower()
    # Clearly measurable = easy
    if any(kw in lower for kw in ["count", "frequency", "rate", "level", "score", "hours", "km"]):
        return "easy"
    # Requires deep inference = hard
    if any(kw in lower for kw in LONG_CONTEXT_KEYWORDS + ["spiritual", "ego", "soul", "aura"]):
        return "hard"
    return "medium"


def get_context_turns(name: str) -> int:
    difficulty = get_difficulty(name)
    if difficulty == "easy":
        return 1
    elif difficulty == "medium":
        return 3
    else:
        return 5


def get_prompt_template(category: str, difficulty: str) -> str:
    """Return a template ID that the scorer will look up."""
    return f"{category}_{difficulty}"


def get_observable(name: str) -> bool:
    """True if the facet can be directly observed in text."""
    lower = name.lower()
    observable_keywords = [
        "sentence", "spelling", "grammar", "brevity", "storytelling",
        "listening", "language", "communication", "verbal",
    ]
    return any(kw in lower for kw in observable_keywords)


def main():
    if not RAW_CSV.exists():
        print(f"ERROR: Raw CSV not found at {RAW_CSV}")
        sys.exit(1)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    facets = []
    seen = set()

    with open(RAW_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_name = row.get("Facets", "").strip()
            cleaned = clean_facet_name(raw_name)

            if not is_valid_facet(cleaned):
                continue
            if cleaned in seen:
                continue
            seen.add(cleaned)

            category = get_category(cleaned)
            polarity = get_polarity(cleaned)
            difficulty = get_difficulty(cleaned)

            facets.append({
                "facet_id": f"F{len(facets)+1:04d}",
                "facet_name": cleaned,
                "facet_name_raw": raw_name,
                "category": category,
                "polarity": polarity,          # positive | negative
                "eval_difficulty": difficulty,  # easy | medium | hard
                "required_context_turns": get_context_turns(cleaned),
                "prompt_template_id": get_prompt_template(category, difficulty),
                "is_observable": get_observable(cleaned),
                "score_min": -2,
                "score_max": 2,
            })

    # Write CSV
    fieldnames = list(facets[0].keys())
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(facets)

    # Write JSON (easier for backend to consume)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(facets, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n✅ Processed {len(facets)} valid facets")
    print(f"   Saved CSV  → {OUT_CSV}")
    print(f"   Saved JSON → {OUT_JSON}")

    # Category breakdown
    from collections import Counter
    cats = Counter(f["category"] for f in facets)
    print("\n📊 Category breakdown:")
    for cat, count in cats.most_common():
        print(f"   {cat:<20} {count}")

    diff = Counter(f["eval_difficulty"] for f in facets)
    print("\n🎯 Difficulty breakdown:")
    for d, count in diff.most_common():
        print(f"   {d:<20} {count}")


if __name__ == "__main__":
    main()
