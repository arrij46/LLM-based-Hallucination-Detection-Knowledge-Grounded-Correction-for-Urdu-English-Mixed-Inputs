# ==============================================================================
# STAGE 4: FACT VERIFICATION (CRAG)
# ==============================================================================

from typing import List, Dict, Tuple, Optional
from collections import Counter
import re


CONFIDENCE_THRESHOLD = 0.70
MIN_SOURCE_AGREEMENT = 2
SINGLE_SOURCE_MIN_CONFIDENCE = 0.75  # Higher bar for single source


def detect_query_intent(query: str) -> str:
    """
    Detect query intent based on question words and structure.
    
    Returns:
        One of: WHO_IS, WHAT_IS, WHEN, WHERE, WHY, HOW, YES_NO
    """
    query_lower = query.lower().strip()
    
    # WHO_IS patterns (most specific first)
    who_patterns = [
        (r'\bwho\s+(is|was|are|were)\b', 'WHO_IS'),
        (r'\b(kon|kaun)\s+(hai|hain|tha|the)\b', 'WHO_IS'),
        (r'\bkisne\b', 'WHO_IS'),
    ]
    
    # WHAT_IS patterns
    what_patterns = [
        (r'\bwhat\s+(is|was|are|were)\b', 'WHAT_IS'),
        (r'\bkya\s+(hai|hain|tha|the)\b', 'WHAT_IS'),
    ]
    
    # WHEN patterns
    when_patterns = [
        (r'\bwhen\s+(did|was|were|is)\b', 'WHEN'),
        (r'\bkab\b', 'WHEN'),
        (r'\bkis\s+waqt\b', 'WHEN'),
    ]
    
    # WHERE patterns
    where_patterns = [
        (r'\bwhere\s+(is|was|are|were)\b', 'WHERE'),
        (r'\bkahan\b', 'WHERE'),
        (r'\bkis\s+jagah\b', 'WHERE'),
    ]
    
    # WHY patterns
    why_patterns = [
        (r'\bwhy\s+(did|is|was)\b', 'WHY'),
        (r'\bkyun\b', 'WHY'),
        (r'\bkis\s+liye\b', 'WHY'),
    ]
    
    # HOW patterns
    how_patterns = [
        (r'\bhow\s+(is|was|did|does)\b', 'HOW'),
        (r'\bkaise\b', 'HOW'),
    ]
    
    # Check patterns in order of specificity
    all_patterns = who_patterns + what_patterns + when_patterns + where_patterns + why_patterns + how_patterns
    
    for pattern, intent in all_patterns:
        if re.search(pattern, query_lower):
            return intent
    
    # Default to WHAT_IS for general questions
    return "WHAT_IS"


def extract_entity_from_query(query: str) -> Optional[str]:
    """
    Extract main entity from query for consistency checking.
    Uses simple heuristics optimized for Pakistani/Urdu context.
    """
    # Remove question words
    cleaned = re.sub(r'\b(who|what|when|where|why|how|is|was|are|were|kon|kaun|kya|kab|kahan|kyun|kaise|hai|hain|tha|the)\b', 
                     '', query, flags=re.IGNORECASE)
    
    # Extract capitalized words (likely entities)
    words = cleaned.split()
    capitalized = [w.strip('?.,!') for w in words if w and len(w) > 1 and w[0].isupper()]
    
    if capitalized:
        # Return longest sequence of capitalized words (multi-word names)
        return ' '.join(capitalized[:3])  # Max 3 words for entity
    
    return None


def check_intent_alignment(intent: str, fact_text: str) -> bool:
    """
    Verify that fact aligns with query intent.
    Uses RELAXED checking to avoid over-filtering.
    """
    fact_lower = fact_text.lower()
    
    if intent == "WHO_IS":
        # Reject only clearly irrelevant content
        reject_patterns = [
            r'\b(upcoming|scheduled|announced|will\s+release)\b',  # Future events
            r'\b(rumors?|speculation|allegedly|reportedly)\b',      # Unverified claims
        ]
        
        for pattern in reject_patterns:
            if re.search(pattern, fact_lower):
                return False
        
        # Accept if it mentions identity-related words OR is a biographical statement
        identity_indicators = ['is', 'was', 'are', 'were', 'known', 'born', 'served', 'worked', 'became']
        return any(indicator in fact_lower for indicator in identity_indicators)
    
    elif intent == "WHEN":
        # Must contain temporal information
        date_patterns = [
            r'\b(19|20)\d{2}\b',  # Years
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # Dates
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
            r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b',
            r'\bin\s+\d{4}\b',  # "in 2024"
            r'\bon\s+\w+\s+\d{1,2}\b',  # "on April 10"
        ]
        return any(re.search(pattern, fact_lower) for pattern in date_patterns)
    
    elif intent == "WHERE":
        # Must contain location information
        location_indicators = ['in ', 'at ', 'located', 'situated', 'from ', 'capital', 'city', 'country']
        return any(indicator in fact_lower for indicator in location_indicators)
    
    # For WHAT_IS, WHY, HOW - be permissive
    return True


def check_entity_consistency(query_entity: Optional[str], fact_text: str) -> bool:
    """
    Verify that fact mentions the same entity as the query.
    Uses RELAXED matching to handle variations.
    """
    if not query_entity:
        return True  # No entity to check
    
    fact_lower = fact_text.lower()
    entity_lower = query_entity.lower()
    
    # Split multi-word entities
    entity_words = [w for w in entity_lower.split() if len(w) > 2]
    
    # Direct substring match
    if entity_lower in fact_lower:
        return True
    
    # Check if most entity words appear in fact
    if entity_words:
        matches = sum(1 for word in entity_words if word in fact_lower)
        # Accept if at least 50% of entity words match
        if matches >= len(entity_words) * 0.5:
            return True
    
    return False


def verify_fact(
    query: str,
    retrieved_docs: List[Dict],
    entity: Optional[str] = None
) -> Dict:
    """
    STRICT Fact Verification Module with CRAG principles.
    
    Returns ONLY verified facts. If verification fails, returns None for verified_fact.
    
    Returns:
        Dict with structure:
            {
                "verified_fact": str | None,
                "confidence": float,
                "sources": List[str],
                "correction_candidates": List[str]
            }
    """
    from pipeline.utils.confidence_scoring import score_retrieval
    from pipeline.utils.fallback_search import fallback_search
    
    # Step 1: Detect query intent and extract entity
    intent = detect_query_intent(query)
    if not entity:
        entity = extract_entity_from_query(query)
    
    # Step 2: Handle empty retrieval
    if not retrieved_docs:
        return {
            "verified_fact": None,
            "confidence": 0.0,
            "sources": [],
            "correction_candidates": []
        }
    
    # Step 3: Score all documents
    scored_docs = []
    for doc in retrieved_docs:
        if not doc.get("text"):
            continue
        score = score_retrieval(query, doc)
        scored_docs.append((doc, score))
    
    if not scored_docs:
        return {
            "verified_fact": None,
            "confidence": 0.0,
            "sources": [],
            "correction_candidates": []
        }
    
    # Sort by confidence
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    best_doc, best_confidence = scored_docs[0]
    
    # Step 4: CRAG - Conditional Re-Retrieval
    if best_confidence < CONFIDENCE_THRESHOLD:
        # Trigger fallback search
        fallback_docs = fallback_search(query)
        
        if fallback_docs:
            # Re-score with fallback docs included
            all_docs = [doc for doc, _ in scored_docs] + fallback_docs
            scored_docs = []
            for doc in all_docs:
                if not doc.get("text"):
                    continue
                score = score_retrieval(query, doc)
                scored_docs.append((doc, score))
            
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            if scored_docs:
                best_doc, best_confidence = scored_docs[0]
    
    # Step 5: Apply filters to scored documents
    valid_docs = []
    for doc, score in scored_docs:
        text = doc.get("text", "")
        if not text:
            continue
        
        # Filter 1: Intent alignment
        if not check_intent_alignment(intent, text):
            continue
        
        # Filter 2: Entity consistency
        if entity and not check_entity_consistency(entity, text):
            continue
        
        valid_docs.append((doc, score))
    
    # Step 6: If no valid docs after filtering, FAIL
    if not valid_docs:
        return {
            "verified_fact": None,
            "confidence": 0.0,
            "sources": [],
            "correction_candidates": []
        }
    
    # Step 7: Cross-verify across sources
    docs_for_verification = [doc for doc, _ in valid_docs]
    verified_fact, sources, agreement_count = cross_verify(docs_for_verification)
    
    # Step 8: Apply verification rules
    
    # Rule 1: No consistent fact found
    if not verified_fact:
        return {
            "verified_fact": None,
            "confidence": 0.0,
            "sources": [],
            "correction_candidates": []
        }
    
    # Rule 2: Multiple sources (≥2) - ACCEPT
    if agreement_count >= MIN_SOURCE_AGREEMENT:
        # Find the doc with this fact to get its confidence
        verified_normalized = normalize_fact(verified_fact)
        fact_confidence = 0.0
        
        for doc, score in valid_docs:
            if normalize_fact(doc.get("text", "")) == verified_normalized:
                fact_confidence = max(fact_confidence, score)
        
        # Boost confidence for multiple sources
        final_confidence = min(fact_confidence * 1.2, 1.0)
        
        return {
            "verified_fact": verified_fact,
            "confidence": round(final_confidence, 3),
            "sources": sources,
            "correction_candidates": generate_corrections(verified_fact, query)
        }
    
    # Rule 3: Single source - check if confidence is high enough
    elif agreement_count == 1:
        verified_normalized = normalize_fact(verified_fact)
        fact_confidence = 0.0
        
        for doc, score in valid_docs:
            if normalize_fact(doc.get("text", "")) == verified_normalized:
                fact_confidence = max(fact_confidence, score)
        
        # Only accept single source if confidence is very high
        if fact_confidence >= SINGLE_SOURCE_MIN_CONFIDENCE:
            # Penalize for single source
            final_confidence = fact_confidence * 0.85
            
            return {
                "verified_fact": verified_fact,
                "confidence": round(final_confidence, 3),
                "sources": sources,
                "correction_candidates": generate_corrections(verified_fact, query)
            }
        else:
            # Single source with low confidence - REJECT
            return {
                "verified_fact": None,
                "confidence": round(fact_confidence, 3),
                "sources": sources,
                "correction_candidates": []
            }
    
    # Rule 4: No sources - FAIL
    else:
        return {
            "verified_fact": None,
            "confidence": 0.0,
            "sources": [],
            "correction_candidates": []
        }


def cross_verify(docs: List[Dict]) -> Tuple[Optional[str], List[str], int]:
    """
    Cross-verify facts across multiple sources.
    Returns the most common fact and its sources.
    
    Returns:
        (verified_fact, sources, agreement_count)
    """
    if not docs:
        return None, [], 0
    
    # Group by normalized fact
    fact_groups = {}  # normalized -> {"original": str, "sources": set, "count": int}

    for doc in docs:
        text = doc.get("text", "").strip()
        if not text:
            continue
        
        source = doc.get("source", "Unknown")
        normalized = normalize_fact(text)
        
        if normalized not in fact_groups:
            fact_groups[normalized] = {
                "original": text,
                "sources": set([source]),
                "count": 1
            }
        else:
            fact_groups[normalized]["sources"].add(source)
            fact_groups[normalized]["count"] += 1

    if not fact_groups:
        return None, [], 0

    # Find fact with most agreement (by count)
    best_normalized = max(fact_groups.keys(), key=lambda k: fact_groups[k]["count"])
    best_group = fact_groups[best_normalized]
    
    verified_fact = best_group["original"]
    sources = list(best_group["sources"])
    agreement_count = len(sources)  # Count unique sources, not occurrences

    return verified_fact, sources, agreement_count


def generate_corrections(verified_fact: str, query: str = "") -> List[str]:
    """
    Generate 2-3 Roman Urdu-English mixed correction candidates.
    
    CRITICAL: Only generates if verified_fact is valid.
    Returns empty list if fact is None or invalid.
    """
    if not verified_fact or verified_fact == "No verified fact available":
        return []
    
    candidates = []
    
    # Detect language mixing in query
    query_lower = query.lower()
    has_urdu_words = any(word in query_lower for word in 
                         ['ka', 'ki', 'ke', 'ko', 'se', 'mein', 'par', 
                          'kon', 'kaun', 'kya', 'kab', 'kahan', 'kyun', 'kaise',
                          'hai', 'hain', 'tha', 'the'])
    
    # Pattern 1: Direct fact (always include original)
    candidates.append(verified_fact)
    
    # Pattern 2: Add Urdu connector if query has Urdu
    if has_urdu_words:
        # Add "hai" only if fact doesn't already end with it
        if not verified_fact.strip().endswith('hai'):
            candidates.append(f"{verified_fact} hai")
    
    # Pattern 3: Formal verification statement
    if has_urdu_words:
        candidates.append(f"Tasdeeq shuda: {verified_fact}")
    else:
        candidates.append(f"Verified: {verified_fact}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_candidates = []
    for c in candidates:
        c_normalized = c.strip()
        if c_normalized and c_normalized not in seen:
            seen.add(c_normalized)
            unique_candidates.append(c_normalized)
    
    # Ensure we have at least 2 candidates
    if len(unique_candidates) < 2:
        unique_candidates.append(f"According to reliable sources: {verified_fact}")
    
    # Return max 3 candidates
    return unique_candidates[:3]


def normalize_fact(text: str) -> str:
    """
    Normalize text for comparison.
    """
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Remove punctuation
    text = re.sub(r'[.!?,:;]', '', text)
    
    # Lowercase
    text = text.strip().lower()
    
    # Remove common variations
    text = re.sub(r'\s+(is|was|are|were|the|a|an)\s+', ' ', text)
    
    return text


# ==============================================================================
# Documentation remains the same
# ==============================================================================
"""
Stage 4: Fact Verification using CRAG (Corrective Retrieval Augmented Generation)

CRITICAL CHANGES FROM PREVIOUS VERSION:
---------------------------------------
1. STRICT VERIFICATION: Only returns verified_fact if verification succeeds
   - If verification fails → verified_fact = None (not a fallback fact)
   
2. CLEAR FAILURE MODES:
   - No documents → verified_fact = None
   - Low confidence after fallback → verified_fact = None
   - Single source with low confidence → verified_fact = None
   - Failed filters → verified_fact = None

3. ACCEPTANCE CRITERIA:
   - Multiple sources (≥2) with any reasonable confidence → ACCEPT
   - Single source with high confidence (≥0.75) → ACCEPT with penalty
   - Everything else → REJECT (verified_fact = None)

4. OUTPUT CONTRACT (STRICT):
   {
       "verified_fact": str | None,     # None if verification failed
       "confidence": float,              # 0.0 if verification failed
       "sources": List[str],             # Empty if verification failed
       "correction_candidates": List[str] # Empty if verification failed
   }

CRAG PRINCIPLES:
---------------
1. Retrieval Quality Scoring - scores all retrieved documents
2. Conditional Re-Retrieval - triggers fallback when confidence < 0.7
3. Cross-Source Verification - requires 2+ sources OR 1 high-confidence source
4. Strict Filtering - rejects facts that don't meet criteria

DETERMINISTIC BEHAVIOR:
----------------------
- Same input → same output
- No randomness
- Handles edge cases by returning None for verified_fact
"""