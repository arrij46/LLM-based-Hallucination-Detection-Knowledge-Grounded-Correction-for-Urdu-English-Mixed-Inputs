import json
import os
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

# Paths
KB_DIR = "../entities"
EMBEDDING_DIR = "../entities/embeddings"
ENTITY_EMBEDDINGS_FILE = os.path.join(EMBEDDING_DIR, "entity_embeddings.pt")
TRANSE_EMBEDDINGS_FILE = os.path.join(EMBEDDING_DIR, "entity_embeddings_transe.pt")
MAPPER_FILE = os.path.join(EMBEDDING_DIR, "mapper.pth")

# Load all entities
with open(os.path.join(KB_DIR, "entities.json"), "r", encoding="utf-8") as f:
    entities = json.load(f)
entity_ids = [e["id"] for e in entities]

# Load embeddings
sentence_emb = torch.load(ENTITY_EMBEDDINGS_FILE)  # (num_entities, 512)
transe_emb = torch.load(TRANSE_EMBEDDINGS_FILE)    # (num_transe_entities, 50)

# Get entities used in TransE (from triples)
all_entities = set()
with open(os.path.join(KB_DIR, "triples.json"), "r", encoding="utf-8") as f:
    triples = json.load(f)
for t in triples:
    all_entities.add(t["head"])
    all_entities.add(t["tail"])
entity_list = list(all_entities)

# Align embeddings
sentence_emb_aligned = []
transe_emb_aligned = []
for i, ent in enumerate(entity_list):
    if ent in entity_ids:
        idx = entity_ids.index(ent)
        sentence_emb_aligned.append(sentence_emb[idx])
        transe_emb_aligned.append(transe_emb[i])

sentence_emb_aligned = torch.stack(sentence_emb_aligned)
transe_emb_aligned = torch.stack(transe_emb_aligned)

print(f"Aligned {len(sentence_emb_aligned)} entities for mapper training")

# Define mapper
mapper = nn.Linear(512, 50)

# Training
optimizer = torch.optim.Adam(mapper.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

print("[INFO] Training mapper from SentenceTransformer to TransE space...")
for epoch in range(100):
    optimizer.zero_grad()
    pred = mapper(sentence_emb_aligned)
    loss = loss_fn(pred, transe_emb_aligned)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Save mapper
torch.save(mapper.state_dict(), MAPPER_FILE)
print(f"[SUCCESS] Mapper saved to {MAPPER_FILE}")