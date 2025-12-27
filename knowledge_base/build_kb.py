import json
import os
import re
from itertools import combinations

# Paths
INPUT_ENTITIES_FILE = "./entities/entities.json"
INPUT_SENTENCES_FILE = "./entities/all_entities.json"
OUTPUT_DIR = "./entities"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_entity_id(id_str):
    # Strip and replace multiple spaces with single
    id_str = re.sub(r'\s+', ' ', id_str.strip())
    # Replace spaces with underscores
    id_str = id_str.replace(' ', '_')
    # Remove non-alphanumeric except underscores
    id_str = re.sub(r'[^a-zA-Z0-9_]', '', id_str)
    return id_str

# Load entities
with open(INPUT_ENTITIES_FILE, "r", encoding="utf-8") as f:
    entities = json.load(f)

# Normalize entity IDs
for ent in entities:
    ent["id"] = clean_entity_id(ent["id"])

# Save normalized entities to KB
with open(os.path.join(OUTPUT_DIR, "entities.json"), "w", encoding="utf-8") as f:
    json.dump(entities, f, indent=2, ensure_ascii=False)
print(f"[SUCCESS] Saved {len(entities)} entities to {OUTPUT_DIR}/entities.json")

# Build mapping text -> ID for sentences
entity_text_to_id = {ent["id"].replace("_", " ").lower(): ent["id"] for ent in entities}

# Load sentences with entities
with open(INPUT_SENTENCES_FILE, "r", encoding="utf-8") as f:
    sentences = json.load(f)

# Step 1: Build relations.json (KG-style)
relations = []
for record in sentences:
    entity_ids = []
    for ent in record["entities"]:
        text_lower = ent["text"].lower()
        if text_lower in entity_text_to_id:
            entity_ids.append(entity_text_to_id[text_lower])
        else:
            entity_ids.append(ent["text"].replace(" ", "_"))

    # Create all pairs for 2+ entities
    for head, tail in combinations(entity_ids, 2):
        relations.append({
            "head": head,
            "relation": "related_to",
            "tail": tail
        })

# Deduplicate relations
unique_relations = { (r["head"], r["relation"], r["tail"]) : r for r in relations }
relations_list = list(unique_relations.values())

# Save relations.json
with open(os.path.join(OUTPUT_DIR, "relations.json"), "w", encoding="utf-8") as f:
    json.dump(relations_list, f, indent=2, ensure_ascii=False)
print(f"[SUCCESS] Saved {len(relations_list)} relations to {OUTPUT_DIR}/relations.json")

# Step 2: Build triples.json (contextual, ready for embedding)
triples = []
for record in sentences:
    entity_ids = []
    for ent in record["entities"]:
        text_lower = ent["text"].lower()
        if text_lower in entity_text_to_id:
            entity_ids.append(entity_text_to_id[text_lower])
        else:
            entity_ids.append(ent["text"].replace(" ", "_"))

    for head, tail in combinations(entity_ids, 2):
        triples.append({
            "head": head,
            "relation": "related_to",
            "tail": tail,
            "sentence": record["sentence"],
            "category": record["category"]
        })

# Save triples.json
with open(os.path.join(OUTPUT_DIR, "triples.json"), "w", encoding="utf-8") as f:
    json.dump(triples, f, indent=2, ensure_ascii=False)
print(f"[SUCCESS] Saved {len(triples)} contextual triples to {OUTPUT_DIR}/triples.json")
