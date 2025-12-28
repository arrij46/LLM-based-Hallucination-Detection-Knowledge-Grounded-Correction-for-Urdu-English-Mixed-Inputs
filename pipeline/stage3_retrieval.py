import json
import os
import torch
import spacy
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# ----------------------------
# Paths
# ----------------------------
KB_DIR = "knowledge_base/entities"
EMBEDDING_DIR="knowledge_base/entities/embeddings"
TRIPLES_FILE = os.path.join(KB_DIR, "triples.json")
ENTITIES_FILE = os.path.join(KB_DIR, "entities.json")
EMBEDDINGS_FILE = os.path.join(EMBEDDING_DIR, "triple_embeddings.pt")
ENTITY_EMBEDDINGS_FILE = os.path.join(EMBEDDING_DIR, "entity_embeddings.pt")

# ----------------------------
# Load KB data
# ----------------------------
with open(TRIPLES_FILE, "r", encoding="utf-8") as f:
    triples = json.load(f)

with open(ENTITIES_FILE, "r", encoding="utf-8") as f:
    entities = json.load(f)

# Load embeddings (precomputed)
triple_embeddings = torch.load(EMBEDDINGS_FILE)
entity_embeddings = torch.load(ENTITY_EMBEDDINGS_FILE)

# Load TransE embeddings and mapper for entity linking
TRANSE_EMBEDDINGS_FILE = os.path.join(KB_DIR, "entity_embeddings_transe.pt")
MAPPER_FILE = os.path.join(KB_DIR, "mapper.pth")
if os.path.exists(TRANSE_EMBEDDINGS_FILE) and os.path.exists(MAPPER_FILE):
    transe_embeddings = torch.load(TRANSE_EMBEDDINGS_FILE)
    mapper = torch.nn.Linear(512, 50)
    mapper.load_state_dict(torch.load(MAPPER_FILE))
    mapper.eval()
    print("[INFO] TransE embeddings and mapper loaded for entity linking")
else:
    transe_embeddings = None
    mapper = None
    print("[INFO] Using sentence embeddings for entity linking")

# Build entity mappings
entity_text_to_id = {e["id"].lower(): e["id"] for e in entities}
entity_id_to_text = {e["id"]: e["id"].replace("_", " ") for e in entities}

# Normalization function
def normalize_text(text: str) -> str:
    if not text:
        return ""
    # Lowercase, replace underscores with spaces, strip, and collapse multiple spaces
    return " ".join(text.lower().replace("_", " ").split())

# Robust entity extraction from query
def extract_query_entity(query: str) -> str:
    query_lower = query.lower()
    # Special handling for "kon" questions: extract entity before "kon"
    if 'kon' in query_lower:
        words = query_lower.split()
        try:
            kon_idx = words.index('kon')
            if kon_idx > 0:
                return ' '.join(words[:kon_idx]).strip()
        except ValueError:
            pass
    # Otherwise, use NER
    doc = nlp(query)
    # Prefer PERSON entities
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    if persons:
        return persons[0]
    # Fallback to other allowed entities
    ents = [ent.text for ent in doc.ents if ent.label_ in ALLOWED_LABELS and len(ent.text) > 2]
    if ents:
        return ents[0]
    # Fallback: check for known entities in query
    query_lower = query.lower()
    query_words = set(query_lower.split())
    for e in entities:
        if e.get("type") in ["PERSON", "GPE", "ORG"]:
            entity_norm = normalize_text(e["id"])
            entity_words = set(entity_norm.split())
            if entity_norm in query_lower or (query_words & entity_words):
                return e["id"]
    # Last resort: return the query as is
    return query

# Update entity mappings with normalization
for e in entities:
    normalized = normalize_text(e["id"])
    entity_text_to_id[normalized] = e["id"]

entity_id_to_index = {e["id"]: i for i, e in enumerate(entities)}

# ----------------------------
# Load multilingual embedding model
# ----------------------------
model = SentenceTransformer("distiluse-base-multilingual-cased-v2")

# Load QA pipeline for extractive QA
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
ALLOWED_LABELS = {"PERSON", "ORG", "GPE", "LOC", "EVENT", "DATE", "NORP", "FAC"}

# ----------------------------
# Roman Urdu normalization
# ----------------------------
def normalize_roman_urdu(text: str) -> str:
    # lowercase, remove extra spaces
    return " ".join(text.lower().split())

# ----------------------------
# Entity linking using embedding similarity with fallback
# ----------------------------
def link_entity(query: str, entities: list, entity_embeddings: torch.Tensor, model, entity_text_to_id: dict, transe_embeddings=None, mapper=None, threshold=0.5) -> str:
    if not query.strip():
        return None
    
    normalized_query = normalize_text(query)
    
    # First, check for exact match
    if normalized_query in entity_text_to_id:
        return entity_text_to_id[normalized_query]
    
    # Second, check for substring match
    query_words = set(normalized_query.split())
    for entity_id, normalized_entity in [(e["id"], normalize_text(e["id"])) for e in entities]:
        entity_words = set(normalized_entity.split('_'))
        if query_words & entity_words:
            return entity_id
    
    # Third, embedding similarity
    query_emb = model.encode(query, convert_to_tensor=True).detach()
    
    if transe_embeddings is not None and mapper is not None:
        # Use TransE space for similarity (L2 distance)
        with torch.no_grad():
            query_transe = mapper(query_emb)  # (1, 50)
        distances = torch.norm(transe_embeddings - query_transe, dim=1)  # L2 distances
        best_score, best_idx = torch.min(distances, dim=0)
        best_score = -float(best_score)
    else:
        # Fallback to sentence embeddings (cosine)
        cos_scores = util.cos_sim(query_emb, entity_embeddings)[0]
        best_score, best_idx = torch.max(cos_scores, dim=0)
        best_score = float(best_score)
    
    if best_score > threshold:
        return entities[best_idx.item()]["id"]
    
    return None

# ============================================================================
# NEW FUNCTIONS FOR INTENT-AWARE RETRIEVAL
# ============================================================================

def detect_query_intent(query: str) -> dict:
    """
    Detect what type of question is being asked.
    Returns: dict with 'type', 'keywords', and 'relations'
    """
    query_lower = query.lower()
    
    # LOCATION questions - CHECK FIRST (before WHO)
    # "konsa sea" = which sea (location)
    location_words = ['sea', 'ocean', 'river', 'mountain', 'city', 'place', 
                     'south', 'north', 'east', 'west', 'border', 'capital']
    if any(word in query_lower for word in ['kahan', 'where']) or \
       ('konsa' in query_lower and any(loc in query_lower for loc in location_words)):
        return {
            'type': 'LOCATION',
            'keywords': ['sea', 'ocean', 'south', 'north', 'border', 'located', 'capital', 'arabian', 'indian'],
            'relations': ['borders_sea', 'south_of', 'located_in', 'capital_of', 'borders']
        }
    
    # WHO questions - asking about identity/person
    if any(word in query_lower for word in ['kon', 'who', 'kaun', 'kon hai', 'kon tha', 'kisne']):
        return {
            'type': 'WHO_IDENTITY',
            'keywords': ['first', 'governor', 'general', 'founder', 'minister', 'president', 'poet', 'jinnah', 'quaid'],
            'relations': ['was_first', 'is', 'was', 'became', 'founder', 'national_poet', 'first_governor']
        }
    
    # COUNT questions - asking about numbers
    if any(word in query_lower for word in ['kitne', 'kitney', 'kitni', 'how many', 'kitnay']):
        return {
            'type': 'COUNT',
            'keywords': ['provinces', 'states', 'cities', 'four', '4', 'number'],
            'relations': ['has_provinces', 'has', 'consists_of', 'contains']
        }
    
    # TIME questions - asking about dates
    if any(word in query_lower for word in ['kab', 'when', 'date', 'day']):
        return {
            'type': 'TIME',
            'keywords': ['independence', 'date', 'day', '1947', 'august', 'founded'],
            'relations': ['independence_date', 'founded', 'established']
        }
    
    # WHAT questions - asking about definitions
    if any(word in query_lower for word in ['kya', 'what', 'kya hai']):
        return {
            'type': 'WHAT',
            'keywords': ['national', 'language', 'animal', 'bird', 'flower'],
            'relations': ['national_language', 'national_animal', 'national_bird', 'is']
        }
    
    # Default: GENERAL
    return {
        'type': 'GENERAL',
        'keywords': [],
        'relations': []
    }


def extract_query_keywords(query: str, intent: dict) -> set:
    """
    Extract important keywords from query based on intent.
    """
    query_lower = query.lower()
    
    # Remove stop words
    stop_words = {'ka', 'ke', 'ki', 'hai', 'hain', 'the', 'is', 'are', 'of', 'a', 'an'}
    
    # Split and filter
    words = query_lower.split()
    keywords = set([w for w in words if len(w) > 2 and w not in stop_words])
    
    # Add intent-specific keywords
    if intent['keywords']:
        keywords.update([k for k in intent['keywords'] if k in query_lower])
    
    return keywords


def score_triple_with_intent(triple: dict, query_keywords: set, intent: dict, semantic_score: float) -> float:
    """
    Score a triple based on semantic similarity + intent matching + keyword overlap.
    """
    score = semantic_score * 0.5  # Base semantic score (reduced weight)
    
    # 1. Keyword matching (most important!)
    triple_text = (triple['sentence'] + ' ' + triple['relation'] + ' ' + 
                  triple['head'] + ' ' + triple['tail']).lower()
    triple_words = set(triple_text.split())
    
    # Count keyword matches
    keyword_matches = query_keywords & triple_words
    if keyword_matches:
        keyword_score = len(keyword_matches) / max(len(query_keywords), 1)
        score += keyword_score * 0.3  # 30% weight for keywords
    
    # 2. Special boost for "first" in WHO_IDENTITY questions
    if intent['type'] == 'WHO_IDENTITY' and 'first' in query_keywords:
        if 'first' in triple_text:
            score += 0.25  # BIG BONUS for "first" matching
    
    # 3. Relation type matching
    if intent['relations']:
        for rel in intent['relations']:
            if rel.lower() in triple['relation'].lower():
                score += 0.2  # 20% bonus for relation match
                break
    
    # 4. Intent-specific keyword boost
    if intent['keywords']:
        for kw in intent['keywords']:
            if kw.lower() in triple_text:
                score += 0.15  # 15% bonus per intent keyword
                break
    
    return score


# ============================================================================
# MAIN RETRIEVAL FUNCTION (UPDATED WITH INTENT AWARENESS)
# ============================================================================

def retrieve_facts(query: str, top_k=5, alpha=0.7, beta=0.3):
    """
    Enhanced retrieval with intent awareness.
    """
    norm_query = normalize_roman_urdu(query)
    
    # NEW: Detect query intent
    intent = detect_query_intent(query)
    query_keywords = extract_query_keywords(query, intent)
    
    #print(f"[DEBUG] Query intent: {intent['type']}")
    #print(f"[DEBUG] Query keywords: {query_keywords}")
    
    # Extract entity from query using robust method
    query_entity = extract_query_entity(norm_query)
    query_entities = [query_entity] if query_entity else []
    
    # Link entities
    linked_kb_ids = [link_entity(ent, entities, entity_embeddings, model, entity_text_to_id, transe_embeddings, mapper) for ent in query_entities]
    linked_kb_ids = [id for id in linked_kb_ids if id]
    linked_kb_id = linked_kb_ids[0] if linked_kb_ids else None
    
    #print(f"[DEBUG] Linked entity: {linked_kb_id}")

    # Filter triples by linked entity (but MORE PERMISSIVE now)
    if linked_kb_id:
        # Get triples mentioning the entity
        indices = [i for i, t in enumerate(triples) if linked_kb_id in [t["head"], t["tail"]] or linked_kb_id.lower() in t["sentence"].lower()]
        indices = [i for i in indices if i < len(triple_embeddings)]
        candidate_triples = [triples[i] for i in indices]
        candidate_embeddings = triple_embeddings[indices] if indices else triple_embeddings[:0]
        
        # If too few candidates, expand search
        if len(candidate_triples) < 10:
            #print(f"[DEBUG] Only {len(candidate_triples)} candidates, expanding search...")
            candidate_triples = triples[:len(triple_embeddings)]
            candidate_embeddings = triple_embeddings
    else:
        # No entity found, search all triples
        candidate_triples = triples[:len(triple_embeddings)]
        candidate_embeddings = triple_embeddings

    # Handle empty case
    if len(candidate_embeddings) == 0:
        return {
            "entity": query_entities[0] if query_entities else norm_query,
            "linked_kb_id": linked_kb_id,
            "retrieved_facts": [],
            "retrieval_confidence": 0.0
        }

    # Embed query
    query_emb = model.encode(norm_query, convert_to_tensor=True)

    # Compute cosine similarity
    cos_scores = util.cos_sim(query_emb, candidate_embeddings)[0]

    # NEW: Score with intent awareness
    final_scores = []
    for i, t in enumerate(candidate_triples):
        semantic_score = float(cos_scores[i])
        
        # Use intent-aware scoring
        final_score = score_triple_with_intent(t, query_keywords, intent, semantic_score)
        
        final_scores.append((final_score, i))

    # Get top-k
    final_scores.sort(reverse=True, key=lambda x: x[0])
    
    # Debug: print top 3 scores
    print(f"[DEBUG] Top 3 results:")
    for score, idx in final_scores[:3]:
        t = candidate_triples[idx]
        print(f"  Score {score:.3f}: {t['sentence'][:80]}")
    
    top_results = final_scores[:top_k]

    retrieved = []
    for score, idx in top_results:
        t = candidate_triples[idx]
        retrieved.append({
            "head": t["head"],
            "relation": t["relation"],
            "tail": t["tail"],
            "sentence": t["sentence"],
            "category": t["category"],
            "score": score
        })

    # Return best answer
    extracted_answer = retrieved[0]["sentence"] if retrieved else "No answer found"
    entity = query_entities[0] if query_entities else norm_query
    retrieval_confidence = retrieved[0]["score"] if retrieved else 0.0

    return {
        "entity": entity,
        "linked_kb_id": linked_kb_id,
        "retrieved_facts": retrieved,
        "extracted_answer": extracted_answer,
        "retrieval_confidence": retrieval_confidence,
        "query_intent": intent['type']  # NEW: include intent in output
    }


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Test with simple queries first
    test_queries = [
        "who is the first governor general of pakistan",
        "pakistan ke kitney provinces hain",
        "pakistan ke south mai konsa sea hai",
        "Pakistan ka national poet kon hai?",
        "Pakistan ka capital kya hai?",
    ]

    print("=" * 80)
    print("TESTING FIXED STAGE 3 RETRIEVAL")
    print("=" * 80)
    
    outputs = []
    for q in test_queries:
        print(f"\n{'='*80}")
        print(f"QUERY: {q}")
        print(f"{'='*80}")
        
        output = retrieve_facts(q, top_k=3)
        
        print(f"\n[RESULT]")
        print(f"Intent: {output.get('query_intent', 'N/A')}")
        print(f"Entity: {output.get('entity', 'N/A')}")
        print(f"Linked KB ID: {output.get('linked_kb_id', 'N/A')}")
        print(f"Top Answer: {output.get('extracted_answer', 'N/A')}")
        print(f"Confidence: {output.get('retrieval_confidence', 0.0):.3f}")
        
        outputs.append({"query": q, "result": output})

    # Save results
    OUTPUT_FILE = "./data"
    out_file = os.path.join(OUTPUT_FILE, "stage3_output.json")
    
    try:
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(outputs, f, ensure_ascii=False, indent=2)
        print(f"\n[SUCCESS] Saved retrieval output to {out_file}")
    except Exception as e:
        print(f"\n[WARNING] Could not save output: {e}")
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)