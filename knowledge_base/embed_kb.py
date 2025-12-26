import json
import os
import torch
from sentence_transformers import SentenceTransformer

os.environ["TRANSFORMERS_OFFLINE"] = "1"

KB_DIR = "./entities/english"
TRIPLES_FILE = os.path.join(KB_DIR, "triples.json")
EMBEDDINGS_FILE = os.path.join(KB_DIR, "triple_embeddings.pt")

MODEL_PATH = r"C:\Users\MyPC\.cache\huggingface\hub\models--sentence-transformers--distiluse-base-multilingual-cased-v2\snapshots\bfe45d0732ca50787611c0fe107ba278c7f3f889"

# Load triples
with open(TRIPLES_FILE, "r", encoding="utf-8") as f:
    triples = json.load(f)

triple_texts = [
    f"{t['head']} {t['relation']} {t['tail']}. {t['sentence']}"
    for t in triples
]

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
