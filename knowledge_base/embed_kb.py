import json
import os
import torch
from sentence_transformers import SentenceTransformer

os.environ["TRANSFORMERS_OFFLINE"] = "1"

KB_DIR = "./entities"
EMBEDDING_DIR = "./entities/embeddings"
TRIPLES_FILE = os.path.join(KB_DIR, "triples.json")
EMBEDDINGS_FILE = os.path.join(EMBEDDING_DIR, "triple_embeddings.pt")
ENTITIES_FILE = os.path.join(KB_DIR, "entities.json")
ENTITY_EMBEDDINGS_FILE = os.path.join(EMBEDDING_DIR, "entity_embeddings.pt")
MODEL_PATH = r"C:\Users\MyPC\.cache\huggingface\hub\models--sentence-transformers--distiluse-base-multilingual-cased-v2\snapshots\bfe45d0732ca50787611c0fe107ba278c7f3f889"

# Load triples
with open(TRIPLES_FILE, "r", encoding="utf-8") as f:
    triples = json.load(f)

triple_texts = [
    f"{t['head']} {t['relation']} {t['tail']}. {t['sentence']}"
    for t in triples
]

# Load entities
with open(ENTITIES_FILE, "r", encoding="utf-8") as f:
    entities = json.load(f)

entity_texts = [e["id"].replace("_", " ") for e in entities]

# Load model
model = SentenceTransformer(MODEL_PATH)

print("[INFO] Embedding KB triples (one-time operation)...")
embeddings = model.encode(
    triple_texts,
    convert_to_tensor=True,
    show_progress_bar=True
)

# Save embeddings
torch.save(embeddings, EMBEDDINGS_FILE)
print(f"[SUCCESS] Saved embeddings to {EMBEDDINGS_FILE}")

print("[INFO] Embedding entities...")
entity_embeddings = model.encode(
    entity_texts,
    convert_to_tensor=True,
    show_progress_bar=True
)

# Save entity embeddings
torch.save(entity_embeddings, ENTITY_EMBEDDINGS_FILE)
print(f"[SUCCESS] Saved entity embeddings to {ENTITY_EMBEDDINGS_FILE}")

print("[INFO] Building TransE-inspired entity-relation mapping...")
# Build relation graph for entity linking enhancement
all_entities = set()
for t in triples:
    all_entities.add(t["head"])
    all_entities.add(t["tail"])

entity_to_relations = {}
for entity in all_entities:
    entity_to_relations[entity] = {
        "heads_to": [],
        "tails_to": []
    }

for t in triples:
    entity_to_relations[t["head"]]["heads_to"].append({
        "relation": t["relation"],
        "tail": t["tail"]
    })
    entity_to_relations[t["tail"]]["tails_to"].append({
        "relation": t["relation"],
        "head": t["head"]
    })

# Save relation graph
RELATION_GRAPH_FILE = os.path.join(KB_DIR, "entity_relations.json")
with open(RELATION_GRAPH_FILE, "w", encoding="utf-8") as f:
    json.dump(entity_to_relations, f, indent=2, ensure_ascii=False)
print(f"[SUCCESS] Saved entity relation graph to {RELATION_GRAPH_FILE}")
