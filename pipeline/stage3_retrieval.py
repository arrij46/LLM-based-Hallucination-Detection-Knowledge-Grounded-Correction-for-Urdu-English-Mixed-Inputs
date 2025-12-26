import json
import os
import torch
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import process, fuzz

# ----------------------------
# Paths
# ----------------------------
KB_DIR = "../knowledge_base/entities/english"
TRIPLES_FILE = os.path.join(KB_DIR, "triples.json")
ENTITIES_FILE = os.path.join(KB_DIR, "entities.json")
EMBEDDINGS_FILE = os.path.join(KB_DIR, "triple_embeddings.pt")

# ----------------------------
# Load KB data
# ----------------------------
with open(TRIPLES_FILE, "r", encoding="utf-8") as f:
    triples = json.load(f)

with open(ENTITIES_FILE, "r", encoding="utf-8") as f:
    entities = json.load(f)

# Load embeddings (precomputed)
triple_embeddings = torch.load(EMBEDDINGS_FILE)

# Build entity mapping
entity_text_to_id = {e["id"].lower(): e["id"] for e in entities}

# ----------------------------
# Load multilingual embedding model (offline)
# ----------------------------
MODEL_PATH = r"C:\Users\MyPC\.cache\huggingface\hub\models--sentence-transformers--distiluse-base-multilingual-cased-v2\snapshots\bfe45d0732ca50787611c0fe107ba278c7f3f889"
model = SentenceTransformer(MODEL_PATH)

# ----------------------------
# Roman Urdu normalization using simple transliteration + lowercasing
# ----------------------------
def normalize_roman_urdu(text: str) -> str:
    # lowercase, remove extra spaces
    return " ".join(text.lower().split())

# ----------------------------
# Entity linking using fuzzy matching
# ----------------------------
def link_entity(query: str, entity_map: dict, threshold=70) -> str:
    candidates = list(entity_map.keys())
    result = process.extractOne(query.lower(), candidates, scorer=fuzz.ratio)
    if result is None:
        return None
    match, score = result[:2]  # take only first 2 elements
    if score >= threshold:
        return entity_map[match]
    return None

# ----------------------------
# Retrieve relevant KB triples
# ----------------------------
def retrieve_facts(query: str, top_k=5, alpha=0.7, beta=0.3):
    """
    alpha: weight for semantic similarity
    beta: weight for entity match
    """
    norm_query = normalize_roman_urdu(query)
    linked_kb_id = link_entity(norm_query, entity_text_to_id)

    # Filter triples by linked entity (head or tail)
    if linked_kb_id:
        candidate_triples = [t for t in triples if linked_kb_id in [t["head"], t["tail"]]]
        candidate_embeddings = triple_embeddings[[i for i, t in enumerate(triples) if linked_kb_id in [t["head"], t["tail"]]]]
    else:
        candidate_triples = triples
        candidate_embeddings = triple_embeddings

    # Embed query
    query_emb = model.encode(norm_query, convert_to_tensor=True)

    # Compute cosine similarity
    cos_scores = util.cos_sim(query_emb, candidate_embeddings)[0]

    # Combine semantic score with entity match boost
    final_scores = []
    for i, t in enumerate(candidate_triples):
        score = float(cos_scores[i])
        if linked_kb_id and linked_kb_id in [t["head"], t["tail"]]:
            score = alpha * score + beta * 1.0  # entity boost
        final_scores.append((score, i))

    # Get top-k
    final_scores.sort(reverse=True)
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

    return {
        "query": query,
        "linked_kb_id": linked_kb_id,
        "retrieved_facts": retrieved,
        "retrieval_confidence": retrieved[0]["score"] if retrieved else 0.0
    }

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    test_queries = [
        "Pakistan 1947 mein azaad mulk bana tha.",
        "Pakistan ki majority abadi Muslims par mushtamil hai.",
        "2010 ke survey mein zyada tar Pakistanis ne khud ko Muslim pehle kaha.",
        "PEW Research Centre ke mutabiq Pakistan mein Sharia law ki support zyada hai.",
        "Sindh mein Shahbaz Qalander ka mazar Sehwan shehar mein waqay hai.",

    ]

    for q in test_queries:
        output = retrieve_facts(q, top_k=3)
        print(json.dumps(output, indent=2, ensure_ascii=False))

