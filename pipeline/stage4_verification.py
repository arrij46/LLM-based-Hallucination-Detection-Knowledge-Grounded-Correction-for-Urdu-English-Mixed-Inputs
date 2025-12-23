# pipeline/stage4_verification.py

from typing import List, Dict
from pipeline.utils.confidence_scoring import score_retrieval
from pipeline.utils.fallback_search import fallback_search


CONFIDENCE_THRESHOLD = 0.70
MIN_SOURCE_AGREEMENT = 2


def verify_fact(
    query: str,
    retrieved_docs: List[Dict]
) -> Dict:
    """
    Core Stage 4 entry point.

    retrieved_docs format:
    [
        {
            "text": "...",
            "source": "...",
            "date": "YYYY"
        }
    ]
    """

    if not retrieved_docs:
        retrieved_docs = fallback_search(query)

    # Score best candidate
    best_doc = retrieved_docs[0]
    confidence = score_retrieval(query, best_doc)

    # CRAG: conditional re-retrieval
    if confidence < CONFIDENCE_THRESHOLD:
        retrieved_docs = fallback_search(query)

    verified_fact, sources = cross_verify(retrieved_docs)
    correction_candidates = generate_corrections(verified_fact)

    return {
        "verified_fact": verified_fact,
        "confidence": confidence,
        "sources": sources,
        "correction_candidates": correction_candidates
    }


# -----------------------------
# Helpers
# -----------------------------
def cross_verify(docs: List[Dict]):
    """
    Accepts facts that appear across multiple sources
    """
    fact_votes = {}
    source_map = {}

    for doc in docs:
        normalized = normalize_fact(doc["text"])
        fact_votes[normalized] = fact_votes.get(normalized, 0) + 1
        source_map.setdefault(normalized, set()).add(doc["source"])

    best_fact = max(fact_votes, key=fact_votes.get)

    if fact_votes[best_fact] < MIN_SOURCE_AGREEMENT:
        return "Verification failed due to insufficient agreement", []

    return best_fact, list(source_map[best_fact])


def generate_corrections(verified_fact: str) -> List[str]:
    return [
        verified_fact,
        f"According to reliable sources, {verified_fact}",
        f"Verified information: {verified_fact}"
    ]


def normalize_fact(text: str) -> str:
    return text.strip().lower()
