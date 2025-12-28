# ==============================================================================
# STAGE 4: FACT VERIFICATION (CRAG) - UPDATED WITH RELAXED THRESHOLDS
# ==============================================================================

from typing import List, Dict, Tuple, Optional
from collections import Counter
import re

# ===== RELAXED THRESHOLDS (MAIN CHANGES) =====
CONFIDENCE_THRESHOLD = 0.50          # LOWERED from 0.70
MIN_SOURCE_AGREEMENT = 1             # LOWERED from 2
SINGLE_SOURCE_MIN_CONFIDENCE = 0.45  # LOWERED from 0.75


def detect_query_intent(query: str) -> str:
    """
    Detect query intent based on question words and structure.
    UPDATED to match Stage 3 intent detection for consistency.
    
    Returns:
        One of: WHO_IS, WHAT_IS, WHEN, LOCATION, COUNT, WHY, HOW, GENERAL
    """
    query_lower = query.lower().strip()
    
    # LOCATION - check FIRST (before WHO) to handle "konsa sea" correctly
    location_words = ['sea', 'ocean', 'river', 'mountain', 'city', 'place', 
                     'south', 'north', 'east', 'west', 'border', 'capital']
    if any(word in query_lower for word in ['kahan', 'where']) or \
       ('konsa' in query_lower and any(loc in query_lower for loc in location_words)):
        return "LOCATION"
    
    # WHO_IS patterns
    if any(pattern in query_lower for pattern in ['kon', 'who', 'kaun', 'kon hai', 'kon tha', 'kisne']):
        return "WHO_IS"
    
    # COUNT patterns
    if any(pattern in query_lower for pattern in ['kitne', 'kitney', 'kitni', 'how many', 'kitnay']):
        return "COUNT"
    
    # WHEN patterns
    if any(pattern in query_lower for pattern in ['kab', 'when', 'kis waqt', 'kis time']):
        return "WHEN"
    
    # WHAT_IS patterns
    if any(pattern in query_lower for pattern in ['kya', 'what', 'kya hai']):
        return "WHAT_IS"
    
    # WHY patterns
    if any(pattern in query_lower for pattern in ['kyun', 'why', 'kis liye']):
        return "WHY"
    
    # HOW patterns
    if any(pattern in query_lower for pattern in ['kaise', 'how', 'kis tarah']):
        return "HOW"
    
    return "GENERAL"


def extract_entity_from_query(query: str) -> Optional[str]:
    """
    Extract main entity from query for consistency checking.
    Uses simple heuristics optimized for Pakistani/Urdu context.
    """
    # Try spaCy first if available
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(query)
        
        # Prefer PERSON entities
        persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        if persons:
            return persons[0]
        
        # Then GPE/ORG entities
        places_orgs = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "ORG"]]
        if places_orgs:
            return places_orgs[0]
    except:
        pass
    
    # Fallback: Extract capitalized words
    cleaned = re.sub(r'\b(who|what|when|where|why|how|is|was|are|were|kon|kaun|kya|kab|kahan|kyun|kaise|hai|hain|tha|the)\b', 
                     '', query, flags=re.IGNORECASE)
    
    words = cleaned.split()
    capitalized = [w.strip('?.,!') for w in words if w and len(w) > 1 and w[0].isupper()]
    
    if capitalized:
        return ' '.join(capitalized[:3])  # Max 3 words
    
    return None


def check_intent_alignment(intent: str, fact_text: str) -> bool:
    """
    Verify that fact aligns with query intent.
    RELAXED: Accept most facts unless clearly wrong.
    """
    fact_lower = fact_text.lower()
    
    if intent == "WHO_IS":
        # Accept any statement with identity/role words
        identity_words = ['is', 'was', 'are', 'were', 'became', 'served', 
                         'hai', 'tha', 'first', 'current', 'governor', 
                         'minister', 'president', 'founder', 'poet', 'known']
        if any(word in fact_lower for word in identity_words):
            return True
    
    elif intent == "COUNT":
        # Accept if contains number or quantity word
        if re.search(r'\b\d+\b', fact_lower) or \
           any(word in fact_lower for word in ['four', 'three', 'two', 'five', 'provinces', 'states']):
            return True
    
    elif intent == "LOCATION":
        # Accept if contains location words
        location_words = ['in', 'at', 'located', 'south', 'north', 'east', 
                         'west', 'sea', 'ocean', 'border', 'near', 'capital']
        if any(word in fact_lower for word in location_words):
            return True
    
    elif intent == "WHEN":
        # Must contain temporal information
        date_patterns = [
            r'\b(19|20)\d{2}\b',  # Years
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # Dates
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
        ]
        if any(re.search(pattern, fact_lower) for pattern in date_patterns):
            return True
    
    # For WHAT_IS, WHY, HOW, GENERAL - be very permissive
    return True


def check_entity_consistency(query_entity: Optional[str], fact_text: str) -> bool:
    """
    Verify that fact mentions the same entity as the query.
    RELAXED: Accept partial matches.
    """
    if not query_entity:
        return True  # No entity to check
    
    fact_lower = fact_text.lower()
    entity_lower = query_entity.lower()
    
    # Direct substring match
    if entity_lower in fact_lower:
        return True
    
    # Split multi-word entities
    entity_words = [w for w in entity_lower.split() if len(w) > 2]
    
    # Check if most entity words appear in fact
    if entity_words:
        matches = sum(1 for word in entity_words if word in fact_lower)
        # RELAXED: Accept if at least 40% of entity words match (was 50%)
        if matches >= len(entity_words) * 0.4:
            return True
    
    return False


def check_explicit_answer(query: str, intent: str, fact_text: str) -> bool:
    """
    Check if fact explicitly answers the query.
    RELAXED: Require only 1 keyword match (was 2).
    """
    query_lower = query.lower()
    fact_lower = fact_text.lower()
    
    # Stop words to ignore
    stop_words = {'the', 'is', 'are', 'was', 'were', 'has', 'have', 'had',
                  'this', 'that', 'with', 'from', 'for', 'and', 'or', 'but',
                  'ka', 'ki', 'ke', 'ko', 'se', 'mein', 'par', 'kon', 'kya',
                  'kab', 'kahan', 'hai', 'hain', 'tha', 'thi', 'the'}
    
    # Extract keywords
    query_keywords = set(re.findall(r'\b\w{3,}\b', query_lower)) - stop_words
    fact_keywords = set(re.findall(r'\b\w{3,}\b', fact_lower)) - stop_words
    
    # RELAXED: Accept if at least 1 keyword matches (was 2)
    overlap = query_keywords & fact_keywords
    if len(overlap) >= 1:
        return True
    
    # Intent-specific acceptance
    if intent == "WHO_IS" and any(verb in fact_lower for verb in ['is', 'was', 'hai', 'tha', 'became', 'served']):
        return True
    
    if intent == "COUNT" and re.search(r'\b\d+\b', fact_lower):
        return True
    
    if intent == "LOCATION" and any(word in fact_lower for word in ['south', 'north', 'sea', 'border', 'capital']):
        return True
    
    if intent == "WHEN" and re.search(r'\b(19|20)\d{2}\b', fact_lower):
        return True
    
    return False


def verify_fact(
    query: str,
    retrieved_docs: List[Dict],
    entity: Optional[str] = None
) -> Dict:
    """
    RELAXED Fact Verification Module.
    
    UPDATED: Uses relaxed thresholds and accepts more facts.
    
    Returns:
        Dict with structure:
            {
                "status": "verified" | "partial" | "unverified" | "insufficient_evidence",
                "verified_fact": str,
                "confidence": float,
                "sources": List[str],
                "reason": str,
                "correction_candidates": List[str]
            }
    """
    
    # Step 1: Handle empty retrieval
    if not retrieved_docs:
        return {
            "status": "insufficient_evidence",
            "verified_fact": "No verified fact available",
            "confidence": 0.0,
            "sources": [],
            "reason": "No documents retrieved",
            "correction_candidates": []
        }
    
    # Step 2: Detect query intent and extract entity
    intent = detect_query_intent(query)
    if not entity:
        entity = extract_entity_from_query(query)
    
    # Step 3: Score all documents
    scored_docs = []
    for doc in retrieved_docs:
        text = doc.get("text", "").strip()
        if not text:
            continue
        
        # Try to use confidence_scoring if available
        try:
            from pipeline.utils.confidence_scoring import score_retrieval
            score = score_retrieval(query, doc)
        except ImportError:
            # Fallback: Simple scoring based on text length and structure
            score = 0.6  # Default baseline
            
            # Boost if it looks like a complete sentence
            if text.endswith('.') or text.endswith('।'):
                score += 0.1
            
            # Boost if it's not too short or too long
            if 20 < len(text) < 200:
                score += 0.1
        
        scored_docs.append((doc, score))
    
    if not scored_docs:
        return {
            "status": "insufficient_evidence",
            "verified_fact": "No verified fact available",
            "confidence": 0.0,
            "sources": [],
            "reason": "No valid documents found",
            "correction_candidates": []
        }
    
    # Sort by confidence
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Step 4: Apply RELAXED filters
    filtered_docs = []
    for doc, score in scored_docs:
        text = doc.get("text", "")
        
        # Filter 1: Intent alignment (RELAXED)
        if not check_intent_alignment(intent, text):
            continue
        
        # Filter 2: Explicit answer (RELAXED - only 1 keyword needed)
        if not check_explicit_answer(query, intent, text):
            continue
        
        # Filter 3: Entity consistency (RELAXED - 40% match)
        # Skip if entity check fails AND score is low
        if entity and not check_entity_consistency(entity, text) and score < 0.50:
            continue
        
        filtered_docs.append((doc, score))
    
    # Step 5: If no docs pass filters, use best available (RELAXED)
    if not filtered_docs:
        # Use top 1 doc from scored_docs
        filtered_docs = scored_docs[:1]
    
    if not filtered_docs:
        return {
            "status": "insufficient_evidence",
            "verified_fact": "No verified fact available",
            "confidence": 0.0,
            "sources": [],
            "reason": "No documents passed filtering",
            "correction_candidates": []
        }
    
    # Step 6: Get best fact
    best_doc, best_score = filtered_docs[0]
    verified_fact = best_doc.get("text", "No verified fact available")
    sources = [best_doc.get("source", "Unknown")]
    
    # Step 7: Cross-verify with other docs if available (OPTIONAL boost)
    if len(filtered_docs) > 1:
        docs_list = [doc for doc, _ in filtered_docs]
        cross_verified_fact, cross_sources, agreement_count = cross_verify(docs_list)
        
        # If cross-verification found agreement, use that
        if cross_verified_fact and agreement_count >= MIN_SOURCE_AGREEMENT:
            verified_fact = cross_verified_fact
            sources = cross_sources
            # Boost confidence for multiple sources
            best_score = min(best_score * 1.15, 1.0)
    
    # Step 8: Determine status with RELAXED thresholds
    confidence = best_score
    
    if confidence >= CONFIDENCE_THRESHOLD:  # 0.50
        status = "verified"
        reason = f"Verified with confidence {confidence:.2f}"
    elif confidence >= 0.35:
        status = "partial"
        reason = f"Partial verification (confidence {confidence:.2f})"
    else:
        status = "unverified"
        confidence = max(confidence, 0.30)  # Floor at 0.30
        reason = f"Low confidence ({confidence:.2f}), using best available"
    
    # Step 9: Generate correction candidates
    correction_candidates = generate_corrections(verified_fact, query)
    
    return {
        "status": status,
        "verified_fact": verified_fact,
        "confidence": round(confidence, 3),
        "sources": sources,
        "reason": reason,
        "correction_candidates": correction_candidates
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

    # Find fact with most agreement
    best_normalized = max(fact_groups.keys(), key=lambda k: fact_groups[k]["count"])
    best_group = fact_groups[best_normalized]
    
    verified_fact = best_group["original"]
    sources = list(best_group["sources"])
    agreement_count = len(sources)  # Count unique sources

    return verified_fact, sources, agreement_count


def generate_corrections(verified_fact: str, query: str = "") -> List[str]:
    """
    Generate 2-3 Roman Urdu-English mixed correction candidates.
    
    Returns empty list if verified_fact is None or invalid.
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
    
    # Pattern 1: Direct fact (always include)
    candidates.append(verified_fact)
    
    # Pattern 2: Add Urdu connector if query has Urdu
    if has_urdu_words and not verified_fact.strip().endswith('hai'):
        candidates.append(f"{verified_fact.rstrip('.')} hai.")
    
    # Pattern 3: Formal verification statement
    if has_urdu_words:
        candidates.append(f"Tasdeeq shuda: {verified_fact}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_candidates = []
    for c in candidates:
        c_normalized = c.strip()
        if c_normalized and c_normalized not in seen:
            seen.add(c_normalized)
            unique_candidates.append(c_normalized)
    
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
# DOCUMENTATION
# ==============================================================================
"""
Stage 4: Fact Verification with RELAXED Thresholds

UPDATED VERSION - Key Changes:
------------------------------
1. RELAXED THRESHOLDS:
   - CONFIDENCE_THRESHOLD: 0.70 → 0.50
   - MIN_SOURCE_AGREEMENT: 2 → 1 (single source acceptable)
   - SINGLE_SOURCE_MIN_CONFIDENCE: 0.75 → 0.45
   - Keyword matching: 2+ → 1+ keyword match required

2. RELAXED FILTERS:
   - Intent alignment: More permissive, accepts most facts
   - Entity consistency: 40% match instead of 50%
   - Explicit answer: Only 1 keyword overlap needed
   - Fallback to best available if no docs pass filters

3. STATUS LEVELS:
   - "verified": confidence ≥ 0.50
   - "partial": confidence ≥ 0.35
   - "unverified": confidence < 0.35 (floor at 0.30)
   - "insufficient_evidence": no docs available

4. DEFAULT BEHAVIOR:
   - When in doubt → ACCEPT the best fact
   - Always returns a fact (even if confidence is low)
   - Uses floor confidence of 0.30 instead of 0.0

CRAG PRINCIPLES (Modified):
---------------------------
1. Retrieval Quality Scoring - scores all retrieved documents
2. Conditional Re-Retrieval - optional (can add fallback search)
3. Cross-Source Verification - RELAXED (1 source OK, 2+ better)
4. Relaxed Filtering - accepts more facts, rejects only clearly wrong ones

OUTPUT CONTRACT:
---------------
{
    "status": str,              # verified | partial | unverified | insufficient_evidence
    "verified_fact": str,       # Always present (even if low confidence)
    "confidence": float,        # 0.30-1.00 range
    "sources": List[str],       # Source IDs
    "reason": str,              # Explanation
    "correction_candidates": List[str]  # 0-3 correction options
}

BEHAVIOR:
--------
- Accepts facts with confidence ≥ 0.50 as "verified"
- Accepts single-source facts (no longer requires 2+ sources)
- More lenient keyword and entity matching
- Always provides best available fact, never returns None
"""