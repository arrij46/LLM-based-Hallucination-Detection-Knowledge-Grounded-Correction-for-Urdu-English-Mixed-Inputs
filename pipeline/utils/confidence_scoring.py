
# ==============================================================================
# FILE 2: pipeline/utils/confidence_scoring.py
# ==============================================================================

"""
Confidence scoring module for Stage 4 verification.
Computes unified confidence scores from multiple signals.
"""

from typing import Dict, Optional
import re
from datetime import datetime


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
    "CDC": 0.95,
    "Reuters": 0.95,
    "Associated Press": 0.95,
    "Express Tribune": 0.85,
    "Geo News": 0.80,
    "ARY News": 0.80
}


def score_retrieval(query: str, fact_doc: Dict) -> float:
    """
    Computes a unified confidence score for a retrieved fact.

    Args:
        query: The search query
        fact_doc: Dict with 'text', 'source', 'date' (optional)

    Returns:
        Confidence score between 0.0 and 1.0
    """
    
    # Validate inputs
    if not query or not fact_doc:
        return 0.0
    
    text = fact_doc.get("text", "")
    if not text:
        return 0.0

    # Calculate component scores
    source_score = source_reliability(fact_doc.get("source"))
    relevance_score = relevance_similarity(query, text)
    specificity_score = specificity(text)
    freshness_score = freshness(fact_doc.get("date"))

    # Weighted average (adjust weights based on your needs)
    weights = {
        'source': 0.30,      # Source trustworthiness
        'relevance': 0.35,   # Query-document match
        'specificity': 0.20, # Has concrete facts
        'freshness': 0.15    # Recency of information
    }
    
    confidence = (
        source_score * weights['source'] +
        relevance_score * weights['relevance'] +
        specificity_score * weights['specificity'] +
        freshness_score * weights['freshness']
    )

    return round(min(max(confidence, 0.0), 1.0), 3)


def source_reliability(source: Optional[str]) -> float:
    """
    Returns reliability score for a given source.
    Case-insensitive matching with partial matches.
    """
    if not source:
        return 0.5
    
    source_lower = source.lower()
    
    # Exact match first
    for known_source, weight in SOURCE_WEIGHTS.items():
        if known_source.lower() == source_lower:
            return weight
    
    # Partial match for flexibility (e.g., "BBC Urdu News" matches "BBC Urdu")
    for known_source, weight in SOURCE_WEIGHTS.items():
        if known_source.lower() in source_lower:
            return weight * 0.95  # Slight penalty for partial match
    
    return 0.5  # Default for unknown sources


def relevance_similarity(query: str, text: str) -> float:
    """
    Improved semantic relevance with better handling of:
    - Multi-word phrases
    - Stop words
    - Urdu transliteration variations
    """
    if not query or not text:
        return 0.0
    
    # Normalize and tokenize
    query_tokens = set(normalize(query).split())
    text_tokens = set(normalize(text).split())
    
    # Remove very short tokens (likely noise)
    query_tokens = {t for t in query_tokens if len(t) > 2}
    text_tokens = {t for t in text_tokens if len(t) > 2}
    
    if not query_tokens:
        return 0.5
    if not text_tokens:
        return 0.0
    
    # Calculate overlap
    overlap = query_tokens.intersection(text_tokens)
    
    # Jaccard similarity with bias toward query coverage
    recall = len(overlap) / len(query_tokens)
    precision = len(overlap) / len(text_tokens) if text_tokens else 0
    
    # F1-like score with emphasis on recall
    if recall + precision == 0:
        return 0.0
    
    f_score = (2 * recall * precision) / (recall + precision)
    
    # Boost score if query coverage is high
    if recall > 0.7:
        f_score = min(f_score * 1.2, 1.0)
    
    return round(f_score, 3)


def specificity(text: str) -> float:
    """
    Enhanced specificity scoring with better pattern recognition.
    Checks presence of dates, numbers, named entities.
    """
    if not text:
        return 0.5
    
    score = 0.5  # Base score
    
    # Date patterns (multiple formats)
    date_patterns = [
        r'\b(19|20)\d{2}\b',  # Years: 1990, 2024
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # DD/MM/YYYY
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
        r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}\b'
    ]
    
    has_date = any(re.search(pattern, text, re.IGNORECASE) for pattern in date_patterns)
    
    # Number patterns
    number_patterns = [
        r'\b\d+(\.\d+)?%?\b',  # Numbers with optional decimals and percentages
        r'\b\d{1,3}(,\d{3})*(\.\d+)?\b',  # Numbers with thousand separators
        r'\$\d+',  # Currency
        r'\d+\s*(million|billion|trillion|thousand)',  # Large numbers
    ]
    
    has_number = any(re.search(pattern, text, re.IGNORECASE) for pattern in number_patterns)
    
    # Named entities (simple heuristic - proper nouns)
    has_proper_nouns = bool(re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text))
    
    # Score calculation
    if has_date and has_number:
        score = 0.95
    elif has_date or has_number:
        score = 0.80
    elif has_proper_nouns:
        score = 0.70
    
    # Penalty for very short text (likely incomplete)
    if len(text) < 20:
        score *= 0.8
    
    return round(score, 3)


def freshness(date: Optional[str]) -> float:
    """
    Calculate freshness score based on document date.
    More recent = higher score.
    """
    if not date:
        return 0.7  # Default for missing dates
    
    try:
        # Try to parse year
        if isinstance(date, str):
            year_match = re.search(r'\b(19|20)\d{2}\b', date)
            if year_match:
                doc_year = int(year_match.group())
            else:
                return 0.7
        else:
            doc_year = int(date)
        
        current_year = datetime.now().year
        age = current_year - doc_year
        
        # Scoring based on age
        if age == 0:  # Current year
            return 1.0
        elif age == 1:
            return 0.95
        elif age <= 3:
            return 0.90
        elif age <= 5:
            return 0.80
        elif age <= 10:
            return 0.70
        else:
            return 0.60
            
    except (ValueError, AttributeError):
        return 0.7


def normalize(text: str) -> str:
    """
    Enhanced normalization for multilingual text.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Keep alphanumeric, spaces, and some Urdu-relevant characters
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()
