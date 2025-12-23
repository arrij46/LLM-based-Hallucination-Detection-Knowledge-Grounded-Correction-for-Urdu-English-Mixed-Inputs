# pipeline/utils/confidence_scoring.py

from typing import Dict
import re
import math


# -----------------------------
# Source Reliability Weights
# -----------------------------
SOURCE_WEIGHTS = {
    "Wikipedia": 0.90,
    "BBC Urdu": 0.95,
    "Dawn": 0.95,
    "World Bank": 0.98,
    "IMF": 0.98,
    "WHO": 0.98,
    "CDC": 0.95
}


# -----------------------------
# Public API
# -----------------------------
def score_retrieval(query: str, fact_doc: Dict) -> float:
    """
    Computes a unified confidence score for a retrieved fact.

    fact_doc must contain:
    {
        "text": "...",
        "source": "...",
        "date": "YYYY" (optional)
    }
    """

    source_score = source_reliability(fact_doc.get("source"))
    relevance_score = relevance_similarity(query, fact_doc.get("text", ""))
    specificity_score = specificity(fact_doc.get("text", ""))
    freshness_score = freshness(fact_doc.get("date"))

    confidence = (
        source_score +
        relevance_score +
        specificity_score +
        freshness_score
    ) / 4

    return round(confidence, 3)


# -----------------------------
# Scoring Components
# -----------------------------
def source_reliability(source: str) -> float:
    return SOURCE_WEIGHTS.get(source, 0.5)


def relevance_similarity(query: str, text: str) -> float:
    """
    Lightweight semantic relevance approximation
    (Embedding-based similarity can be swapped later)
    """
    query_tokens = set(normalize(query).split())
    text_tokens = set(normalize(text).split())

    if not query_tokens or not text_tokens:
        return 0.0

    overlap = query_tokens.intersection(text_tokens)
    return len(overlap) / len(query_tokens)


def specificity(text: str) -> float:
    """
    Checks presence of dates, numbers, named entities
    """
    number_pattern = r"\b\d+(\.\d+)?\b"
    date_pattern = r"\b(19|20)\d{2}\b"

    has_number = bool(re.search(number_pattern, text))
    has_date = bool(re.search(date_pattern, text))

    if has_number and has_date:
        return 0.95
    if has_number or has_date:
        return 0.8
    return 0.6


def freshness(date: str | None) -> float:
    if not date:
        return 0.7
    return 0.9


def normalize(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9 ]", "", text.lower())
