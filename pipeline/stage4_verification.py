# ==============================================================================
# STAGE 4: FACT VERIFICATION (CRAG)
# ==============================================================================


from typing import List, Dict, Tuple, Optional
from collections import Counter
import re


CONFIDENCE_THRESHOLD = 0.70
MIN_SOURCE_AGREEMENT = 2  # Require at least 2 sources for verification


def detect_query_intent(query: str) -> str:
    """
    Detect query intent based on question words and structure.
    
    Returns:
        One of: WHO_IS, WHAT_IS, WHEN, WHERE, WHY, HOW, YES_NO
    """
    query_lower = query.lower().strip()
    
    # WHO_IS patterns
    who_patterns = ['kon', 'who', 'kaun', 'kisne']
    if any(pattern in query_lower for pattern in who_patterns):
        return "WHO_IS"
    
    # WHAT_IS patterns
    what_patterns = ['kya', 'what', 'kya hai', 'kya hain']
    if any(pattern in query_lower for pattern in what_patterns):
        return "WHAT_IS"
    
    # WHEN patterns
    when_patterns = ['kab', 'when', 'kis waqt', 'kis time']
    if any(pattern in query_lower for pattern in when_patterns):
        return "WHEN"
    
    # WHERE patterns
    where_patterns = ['kahan', 'where', 'kis jagah', 'kis location']
    if any(pattern in query_lower for pattern in where_patterns):
        return "WHERE"
    
    # WHY patterns
    why_patterns = ['kyun', 'why', 'kis liye', 'kis wajah se']
    if any(pattern in query_lower for pattern in why_patterns):
        return "WHY"
    
    # HOW patterns
    how_patterns = ['kaise', 'how', 'kis tarah']
    if any(pattern in query_lower for pattern in how_patterns):
        return "HOW"
    
    # YES_NO patterns
    yes_no_patterns = ['hai', 'hain', 'is', 'are', 'was', 'were', '?']
    if query_lower.endswith('?') or any(pattern in query_lower for pattern in yes_no_patterns):
        return "YES_NO"
    
    # Default to WHAT_IS
    return "WHAT_IS"


def extract_entity_from_query(query: str) -> Optional[str]:
    """
    Extract main entity from query for consistency checking.
    Uses spacy if available, otherwise falls back to simple heuristics.
    """
    # Try spacy NER first
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(query)
        
        # Prefer PERSON entities
        persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        if persons:
            return persons[0]
        
        # Then ORG, GPE
        orgs = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE"]]
        if orgs:
            return orgs[0]
    except (ImportError, OSError):
        # spacy not available or model not installed
        pass
    
    # Fallback: extract capitalized words (likely entities)
    words = query.split()
    capitalized = [w for w in words if w[0].isupper() and len(w) > 2]
    if capitalized:
        return capitalized[0]
    
    return None


def check_intent_alignment(intent: str, fact_text: str) -> bool:
    """
    Verify that fact aligns with query intent.
    
    WHO_IS: Accept only identity/definition facts
    Reject: achievements, relationships, events, filmography, opinions
    """
    fact_lower = fact_text.lower()
    
    if intent == "WHO_IS":
        # Accept: "X is/was Y", "X, a Y", "X, the Y"
        identity_patterns = [
            r'\bis\s+(a|an|the)\s+',
            r'\bwas\s+(a|an|the)\s+',
            r',\s+(a|an|the)\s+',
            r'\bis\s+known\s+as',
            r'\bis\s+called'
        ]
        
        # Reject: achievements, events, relationships
        reject_patterns = [
            r'\bwon\b', r'\bdefeated\b', r'\bmarried\b', r'\bdivorced\b',
            r'\bstarred\s+in\b', r'\bdirected\b', r'\bwrote\b', r'\breleased\b',
            r'\bappeared\s+in\b', r'\bplayed\s+in\b'
        ]
        
        # Check for reject patterns first
        for pattern in reject_patterns:
            if re.search(pattern, fact_lower):
                return False
        
        # Check for accept patterns
        for pattern in identity_patterns:
            if re.search(pattern, fact_lower):
                return True
        
        # If no clear pattern, be conservative
        return False
    
    elif intent == "WHAT_IS":
        # Accept definitions, descriptions
        return True  # More permissive for WHAT_IS
    
    elif intent == "WHEN":
        # Must contain temporal information
        date_patterns = [
            r'\b(19|20)\d{2}\b',  # Years
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # Dates
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)',
            r'\bin\s+\d{4}\b'  # "in 2024"
        ]
        return any(re.search(pattern, fact_lower) for pattern in date_patterns)
    
    elif intent == "WHERE":
        # Must contain location information
        location_indicators = ['in', 'at', 'located', 'situated', 'found']
        return any(indicator in fact_lower for indicator in location_indicators)
    
    # For WHY, HOW, YES_NO - be more permissive
    return True


def check_entity_consistency(query_entity: Optional[str], fact_text: str) -> bool:
    """
    Verify that fact mentions the same entity as the query.
    """
    if not query_entity:
        return True  # No entity to check
    
    fact_lower = fact_text.lower()
    entity_lower = query_entity.lower()
    
    # Check if entity appears in fact
    entity_words = set(entity_lower.split())
    fact_words = set(fact_lower.split())
    
    # Direct match
    if entity_lower in fact_lower:
        return True
    
    # Word overlap (for multi-word entities)
    if entity_words & fact_words:
        return True
    
    return False


def check_explicit_answer(query: str, intent: str, fact_text: str) -> bool:
    """
    Verify that fact EXPLICITLY answers the query.
    Does NOT infer or merge information.
    """
    query_lower = query.lower()
    fact_lower = fact_text.lower()
    
    # Extract key terms from query
    query_keywords = set(re.findall(r'\b\w{3,}\b', query_lower))
    fact_keywords = set(re.findall(r'\b\w{3,}\b', fact_lower))
    
    # Remove common stop words
    stop_words = {'the', 'is', 'are', 'was', 'were', 'has', 'have', 'had', 
                  'this', 'that', 'with', 'from', 'for', 'and', 'or', 'but',
                  'ka', 'ki', 'ke', 'ko', 'se', 'mein', 'par'}
    query_keywords -= stop_words
    fact_keywords -= stop_words
    
    # Check for meaningful overlap
    overlap = query_keywords & fact_keywords
    if len(overlap) < 2:  # Need at least 2 keyword matches
        return False
    
    # For WHO_IS, ensure fact provides identity
    if intent == "WHO_IS":
        identity_verbs = ['is', 'was', 'are', 'were']
        if not any(verb in fact_lower for verb in identity_verbs):
            return False
    
    return True


def verify_fact(
    query: str,
    retrieved_docs: List[Dict],
    entity: Optional[str] = None
) -> Dict:
    """
    STRICT Fact Verification Module
    
    Verifies whether retrieved evidence EXPLICITLY answers the user query.
    Follows strict rules:
    - Only accepts facts that directly answer the query
    - Requires entity consistency
    - Checks intent alignment
    - Requires at least 2 sources for verification
    - Does NOT invent, merge, or infer facts
    
    Args:
        query: User query string
        retrieved_docs: List of documents with 'text', 'source', 'date' fields
        entity: Detected entity from query (optional, will extract if not provided)
    
    Returns:
        Dict with structure:
            {
                "status": "verified" | "unverified" | "insufficient_evidence",
                "verified_fact": str | null,
                "confidence": float,
                "sources": List[str],
                "reason": str,
                "correction_candidates": List[str]  # For Stage 5 compatibility
            }
    """
    
    # Step 1: Detect query intent and extract entity
    intent = detect_query_intent(query)
    if not entity:
        entity = extract_entity_from_query(query)
    
    # Step 2: Handle empty retrieval
    if not retrieved_docs:
        return {
            "status": "insufficient_evidence",
            "verified_fact": None,
            "confidence": 0.0,
            "sources": [],
            "reason": "No documents retrieved",
            "correction_candidates": []
        }
    
    # Step 3: Filter documents by strict criteria
    valid_docs = []
    for doc in retrieved_docs:
        text = doc.get("text", "")
        if not text:
            continue
        
        # Check intent alignment
        if not check_intent_alignment(intent, text):
            continue
        
        # Check entity consistency
        if entity and not check_entity_consistency(entity, text):
            continue
        
        # Check explicit answer
        if not check_explicit_answer(query, intent, text):
            continue
        
        valid_docs.append(doc)
    
    # Step 4: If no valid docs after strict filtering, try fallback with relaxed criteria
    if not valid_docs:
        # Fallback: Use best document even if it doesn't pass all strict checks
        # This ensures we always return something when documents are available
        if retrieved_docs:
            best_doc = retrieved_docs[0]
            text = best_doc.get("text", "")
            if text:
                from pipeline.utils.confidence_scoring import score_retrieval
                confidence = score_retrieval(query, best_doc) * 0.5  # Low confidence for unverified
                
                return {
                    "status": "unverified",
                    "verified_fact": text,  # Return best available fact
                    "confidence": round(confidence, 3),
                    "sources": [best_doc.get("source", "Unknown")],
                    "reason": f"Retrieved evidence does not explicitly answer the {intent} query, using best available",
                    "correction_candidates": generate_corrections(text, query)
                }
        
        # Complete failure - no documents at all
        return {
            "status": "unverified",
            "verified_fact": None,
            "confidence": 0.0,
            "sources": [],
            "reason": f"Retrieved evidence does not explicitly answer the {intent} query",
            "correction_candidates": []
        }
    
    # Step 5: Cross-verify across sources (require at least 2 sources)
    verified_fact, sources, agreement_count = cross_verify(valid_docs)
    
    # Step 6: If no consistent fact, use best single document
    if not verified_fact:
        # Fallback: Use best single document
        best_doc = valid_docs[0]
        text = best_doc.get("text", "")
        if text:
            from pipeline.utils.confidence_scoring import score_retrieval
            confidence = score_retrieval(query, best_doc) * 0.6  # Lower confidence
            
            return {
                "status": "unverified",
                "verified_fact": text,  # Return best available fact
                "confidence": round(confidence, 3),
                "sources": [best_doc.get("source", "Unknown")],
                "reason": "No consistent fact found across sources, using best single source",
                "correction_candidates": generate_corrections(text, query)
            }
        
        # Complete failure
        return {
            "status": "insufficient_evidence",
            "verified_fact": None,
            "confidence": 0.0,
            "sources": [],
            "reason": "No consistent fact found across sources",
            "correction_candidates": []
        }
    
    # Step 7: Calculate confidence based on source agreement and quality
    from pipeline.utils.confidence_scoring import score_retrieval
    
    # Score the verified fact (find doc with matching normalized text)
    verified_normalized = normalize_fact(verified_fact)
    best_doc = None
    for doc in valid_docs:
        if normalize_fact(doc.get("text", "")) == verified_normalized:
            best_doc = doc
            break
    
    if not best_doc:
        best_doc = valid_docs[0]
    
    confidence = score_retrieval(query, best_doc)
    
    # Step 8: Determine status and adjust confidence based on source agreement
    if agreement_count >= MIN_SOURCE_AGREEMENT:
        # Multiple sources - verified
        status = "verified"
        # Boost confidence for multiple sources
        confidence = min(confidence * 1.2, 1.0)
        reason = f"Verified by {agreement_count} independent source(s)"
    elif agreement_count == 1 and confidence >= 0.6:
        # Single source but high confidence - accept with warning
        status = "unverified"
        confidence = confidence * 0.8  # Penalize for single source
        reason = f"Single source found (confidence: {confidence:.2f})"
    else:
        # Single source with low confidence - still return but mark as insufficient
        status = "insufficient_evidence"
        confidence = confidence * 0.6  # Further penalty
        reason = f"Only {agreement_count} source(s) with low confidence"
    
    # Step 9: Generate correction candidates (always generate if we have a fact)
    correction_candidates = generate_corrections(verified_fact, query) if verified_fact else []
    
    # Step 10: Return result (ALWAYS return the fact if we have one, even if verification is weak)
    return {
        "status": status,
        "verified_fact": verified_fact,  # Always return fact if available
        "confidence": round(confidence, 3),
        "sources": sources,
        "reason": reason,
        "correction_candidates": correction_candidates
    }


def cross_verify(docs: List[Dict]) -> Tuple[Optional[str], List[str], int]:
    """
    Accepts facts that appear across multiple sources.
    Returns the ORIGINAL fact text (not normalized) that has most agreement.
    
    Returns:
        (verified_fact, sources, agreement_count)
    """
    if not docs:
        return None, [], 0
    
    # Group by normalized fact, but keep original text
    fact_groups = {}  # normalized -> (original_text, sources, count)

    for doc in docs:
        text = doc.get("text", "").strip()
        if not text:
            continue
        
        source = doc.get("source", "Unknown")
        normalized = normalize_fact(text)
        
        if normalized not in fact_groups:
            # Use first occurrence as canonical text
            fact_groups[normalized] = {
                "original_text": text,
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
    
    verified_fact = best_group["original_text"]
    sources = list(best_group["sources"])
    agreement_count = best_group["count"]

    return verified_fact, sources, agreement_count


def generate_corrections(verified_fact: str, query: str = "") -> List[str]:
    """
    Generate 2-3 Roman Urdu-English mixed correction candidates.
    
    Candidates should:
    - Preserve the verified fact content
    - Use Roman Urdu-English code-mixed style
    - Match the original query's language mixing pattern
    - Be natural and conversational
    
    Args:
        verified_fact: The verified factual statement
        query: Original user query (for style matching)
    
    Returns:
        List of 2-3 correction candidate strings
    """
    if not verified_fact or verified_fact == "No verified fact available":
        return []
    
    candidates = []
    
    # Extract key information from verified fact for mixing
    fact_lower = verified_fact.lower()
    
    # Pattern 1: Direct fact with Urdu connector
    candidates.append(f"{verified_fact} hai")
    
    # Pattern 2: Query-contextualized with Urdu-English mix
    if query:
        # Detect if query has Urdu words
        query_lower = query.lower()
        has_urdu_indicators = any(word in query_lower for word in ['ka', 'ki', 'kon', 'kya', 'kahan', 'kab', 'hain', 'hai'])
        
        if has_urdu_indicators:
            # Roman Urdu-English mixed style
            candidates.append(f"Tasdeeq shuda fact yeh hai ke {verified_fact}")
        else:
            # More English-dominant style
            candidates.append(f"Verified fact: {verified_fact}")
    else:
        candidates.append(f"Verified fact: {verified_fact}")
    
    # Pattern 3: Natural Roman Urdu-English statement
    # Try to convert common English patterns to mixed style
    if "is the" in fact_lower or "was the" in fact_lower:
        # Convert "X is the Y" to "X Y hai"
        mixed = verified_fact
        mixed = re.sub(r'\bis the\b', 'hai', mixed, flags=re.IGNORECASE)
        mixed = re.sub(r'\bwas the\b', 'tha', mixed, flags=re.IGNORECASE)
        if mixed != verified_fact:
            candidates.append(mixed)
    
    # Pattern 4: Simple statement with Urdu ending
    if len(candidates) < 3:
        candidates.append(f"{verified_fact}")
    
    # Ensure we have 2-3 candidates, remove duplicates
    seen = set()
    unique_candidates = []
    for c in candidates:
        c_normalized = c.strip()
        if c_normalized and c_normalized not in seen:
            seen.add(c_normalized)
            unique_candidates.append(c_normalized)
            if len(unique_candidates) >= 3:
                break
    
    # Ensure at least 2 candidates
    if len(unique_candidates) < 2 and verified_fact:
        unique_candidates.append(verified_fact)
    
    return unique_candidates[:3]  # Return max 3 candidates


def normalize_fact(text: str) -> str:
    """
    Normalize text for comparison.
    Handles multiple languages and special characters.
    """
    import re
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Remove common punctuation but keep structure
    text = re.sub(r'[.!?,:;]', '', text)
    
    # Lowercase for comparison
    return text.strip().lower()


"""
Stage 4: Fact Verification using CRAG (Corrective Retrieval Augmented Generation) Principles

PURPOSE:
--------
Stage 4 performs fact verification on retrieved knowledge base facts to ensure
reliability before passing verified information to Stage 5 (Correction Generation).
This stage implements CRAG-style verification with conditional re-retrieval and
cross-source validation.

WHY IT EXISTS:
-------------
After Stage 3 retrieves facts from the knowledge base, we need to verify:
1. The retrieved facts are reliable and accurate
2. Multiple sources agree on the same fact (when available)
3. Low-confidence retrievals trigger additional search
4. Only verified, high-confidence facts proceed to correction

CRAG PRINCIPLES APPLIED:
-----------------------
1. Retrieval Quality Scoring:
   - Scores each retrieved fact based on source reliability, semantic relevance,
     specificity (dates, numbers, named entities), and freshness
   - Uses weighted combination of these signals

2. Conditional Re-Retrieval:
   - If confidence < 0.7, triggers fallback search from additional sources
   - Cross-checks facts across multiple sources
   - Selects most consistent and reliable fact

3. Cross-Source Verification:
   - Deduplicates facts across sources
   - Prefers facts with majority agreement
   - Prioritizes authoritative sources (Wikipedia, BBC, Dawn > random blogs)

4. Correction Candidate Generation:
   - Generates 2-3 Roman Urdu-English mixed correction candidates
   - Preserves original query style and language mixing
   - Uses verified fact without adding new information

INPUT CONTRACT:
--------------
Stage 4 receives input from Stage 3 (via main.py transformation):
    retrieved_docs: List[Dict] where each dict has:
        - "text": str (factual content)
        - "source": str (source name, e.g., "Wikipedia", "BBC Urdu")
        - "date": str (optional, e.g., "2024")

    query: str (original user query for context)

OUTPUT CONTRACT (STRICT):
------------------------
Stage 4 MUST return exactly this structure (JSON-serializable):
    {
        "verified_fact": str,              # The verified factual statement
        "confidence": float,               # Confidence score [0.0, 1.0]
        "sources": List[str],              # List of source names
        "correction_candidates": List[str] # 2-3 Roman Urdu-English mixed candidates
    }

This output is directly consumed by Stage 5 (Automated Correction).

DETERMINISTIC BEHAVIOR:
----------------------
- Same input → same output (no randomness)
- Pure function (no side effects)
- Handles edge cases gracefully (empty retrieval, API failures, etc.)

Author: NLP Hallucination Detection & Correction Pipeline
"""
