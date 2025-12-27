import json
import os
import torch
import spacy
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# ----------------------------
# Paths
# ----------------------------
KB_DIR = "./knowledge_base/entities"
EMBEDDING_DIR="./knowledge_base/entities/embeddings"
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
    entities = [ent.text for ent in doc.ents if ent.label_ in ALLOWED_LABELS and len(ent.text) > 2]
    if entities:
        return entities[0]
    # Fallback: check for known entities in query
    query_lower = query.lower()
    query_words = set(query_lower.split())
    for e in entities:
        if e["type"] in ["PERSON", "GPE", "ORG"]:
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
# Load multilingual embedding model (offline)
# ----------------------------
MODEL_PATH = r"C:\Users\MyPC\.cache\huggingface\hub\models--sentence-transformers--distiluse-base-multilingual-cased-v2\snapshots\bfe45d0732ca50787611c0fe107ba278c7f3f889"
model = SentenceTransformer(MODEL_PATH)
# model=SentenceTransformer("distiluse-base-multilingual-cased-v2")

# Load QA pipeline for extractive QA
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
ALLOWED_LABELS = {"PERSON", "ORG", "GPE", "LOC", "EVENT", "DATE", "NORP", "FAC"}

# ----------------------------
# Roman Urdu normalization using simple transliteration + lowercasing
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
        best_score = -float(best_score)  # Negative for similarity-like scoring (lower distance = higher score)
    else:
        # Fallback to sentence embeddings (cosine)
        cos_scores = util.cos_sim(query_emb, entity_embeddings)[0]
        best_score, best_idx = torch.max(cos_scores, dim=0)
        best_score = float(best_score)
    
    if best_score > threshold:
        return entities[best_idx.item()]["id"]
    
    return None

# ----------------------------

import re

# ----------------------------
# Retrieve relevant KB triples
# ----------------------------
def retrieve_facts(query: str, top_k=5, alpha=0.7, beta=0.3):
    """
    alpha: weight for semantic similarity
    beta: weight for entity match
    """
    norm_query = normalize_roman_urdu(query)
    
    # Extract entity from query using robust method
    query_entity = extract_query_entity(norm_query)
    query_entities = [query_entity] if query_entity else []
    
    # Link entities with updated function signature
    linked_kb_ids = [link_entity(ent, entities, entity_embeddings, model, entity_text_to_id, transe_embeddings, mapper) for ent in query_entities]
    linked_kb_ids = [id for id in linked_kb_ids if id]
    linked_kb_id = linked_kb_ids[0] if linked_kb_ids else None

    # Filter triples by linked entity (head, tail, or in sentence)
    if linked_kb_id:
        indices = [i for i, t in enumerate(triples) if linked_kb_id in [t["head"], t["tail"]] or linked_kb_id.lower() in t["sentence"].lower() or linked_kb_id.lower() in t["sentence"].lower()]
        # Ensure indices are within embeddings bounds
        indices = [i for i in indices if i < len(triple_embeddings)]
        candidate_triples = [triples[i] for i in indices]
        candidate_embeddings = triple_embeddings[indices] if indices else triple_embeddings[:0]
    else:
        candidate_triples = triples[:len(triple_embeddings)]  # Limit to available embeddings
        candidate_embeddings = triple_embeddings

    # Handle case when no embeddings
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

    # Combine semantic score with entity match boost and TransE score
    final_scores = []
    for i, t in enumerate(candidate_triples):
        semantic_score = float(cos_scores[i])
        transe_score = 0.0
        if transe_embeddings is not None:
            head_idx = entity_id_to_index.get(t["head"])
            tail_idx = entity_id_to_index.get(t["tail"])
            if head_idx is not None and tail_idx is not None and head_idx < len(transe_embeddings) and tail_idx < len(transe_embeddings):
                distance = torch.norm(transe_embeddings[head_idx] - transe_embeddings[tail_idx])
                transe_score = -float(distance)  # higher for closer entities in TransE space
        entity_boost = beta if linked_kb_id and linked_kb_id in [t["head"], t["tail"]] else 0.0
        sentence_boost = 0.2 if linked_kb_id and linked_kb_id.lower() in t["sentence"].lower() else 0.0
        keyword_boost = 0.2 if set(norm_query.split()) & set(t["sentence"].lower().split()) else 0.0
        final_score = alpha * semantic_score + entity_boost + sentence_boost + keyword_boost
        final_scores.append((final_score, i))

    # Get top-k, filtering low scores
    final_scores.sort(reverse=True)
    filtered_scores = [(score, idx) for score, idx in final_scores if score >= 0.5]
    if filtered_scores:
        top_results = filtered_scores[:top_k]
    else:
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

    # Return the top retrieved sentence as answer
    extracted_answer = retrieved[0]["sentence"] if retrieved else "No answer found"

    entity = query_entities[0] if query_entities else norm_query
    retrieval_confidence = retrieved[0]["score"] if retrieved else 0.0

    return {
        "entity": entity,
        "linked_kb_id": linked_kb_id,
        "retrieved_facts": retrieved,
        "extracted_answer": extracted_answer,
        "retrieval_confidence": retrieval_confidence
    }

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Expanded list of question-style queries (answerable from KB)
    test_queries = [
        "Pakistan ka national poet kon hai?",
        "Pakistan ki majority population ka kiya religion hai?",
        "Jinnah kon hain?",
        "Pakistan ka capital kya hai?",
        "Sindh ka capital kahan hai?",
        "Pakistan independent kab bana?",
        "Quaid-e-Azak kon thy?",
        "Pakistan ka founder kon hai?",
        "Lahore ki history kya hai?",
        "Who is the founder of Pakistan?",
        "What is the capital of Pakistan?",
        "When did Pakistan gain independence?",
        "Tell me something about Allama Iqbal.",
        "Who was Quaid-e-Azam?",
        "Pakistan kab bana?",
        "Pakistan ki sabse unchi mountain kaun si hai?",
        "K2 kahan hai?",
        "Pakistan ki national flower kaun si hai?",
        "Pakistan ki national bird kaun si hai?",
        "Pakistan ki national animal kaun si hai?",
        "Pakistan ki national sport kaun sa hai?",
        "Pakistan ne cricket worldcup kab jeeta?",
        "Sqash player Jahangir Khan kis cheez ke liye famous hai?",
        # Politics
        "Pakistan ki judiciary system kaisi hai?",
        "Pakistan ki election system kaisi hai?",
        "Imran Khan ki political party kaun si hai?",
        "Pakistan ki military history kya hai?",
        "Pakistan ki political history kya hai?",
        "Pakistan ki current political situation kaisi hai?",
        # Culture
        "Eid Pakistan mein kaise manaty hain?",
        "Pakistan ki traditional kapde kya hain?",
        "Shalimar Gardens Lahore kahan hai?",
        "Pakistan ki famous sweets kaun si hain?",
        "Truck art Pakistan mein kya hai?",
        "Pakistan ki folk music kaisi hoti hai?",
        "Pakistan ki traditional dance kaun si hai?",
        "Pakistan ki famous festivals kaun si hain?",
        "Pakistan ki national dress kya hai?",
        "Pakistan ki famous handicrafts kya hain?", 
        "Pakistan ki traditional food kya hai?",
        "Pakistan ki famous art kaun kaun se hain?",
        "Pakistan ki literature ki history kya hai?",
        "Pakistan ki cinema industry kaisi hai?",
        "Pakistan ki famous poets kaun kaun se hain?",
        "Pakistan ki traditional music instruments kya hain?",

        # Geography
        "Thar Desert kahan hai?",
        "Pakistan ki national parks kaun si hain?",
        "Pakistan ki borders kis se lagti hain?",
        "Lahore ki history kya hai?",
        "Quetta ki climate kaisi hai?",
        "Pakistan ki longest river kaun si hai?",
        "Pakistan ki highest mountain kaun si hai?",
        "Pakistan ki largest desert kaun si hai?",
        "Pakistan ki national flower kaun si hai?",
        "Pakistan ki national bird kaun si hai?",
        "Pakistan ki national animal kaun si hai?",
        "Pakistan ki national sport kaun sa hai?",
        "Pakistan ki national fruit kaun sa hai?",
        "Pakistan ki economy kaisi hai?",
        "Sialkot ki famous cheezen kya hain?",
        "Multan ki history kya hai?",
        "Peshawar ki culture kaisi hai?",
        "KPK ki geography kaisi hai?"
    ]

    outputs = []
    for q in test_queries:
        output = retrieve_facts(q, top_k=5)
        print(json.dumps(output, indent=2, ensure_ascii=False))
        outputs.append({"query": q, "result": output})
        OUTPUT_FILE = "./data"

    # Save to stage_3_output.json in project root
    out_file = os.path.join(OUTPUT_FILE, "stage3_output.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    print(f"[SUCCESS] Saved retrieval output to {out_file}")

