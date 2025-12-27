import json
import os
import torch
import numpy as np
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline

# Load triples
KB_DIR = "../entities"
TRIPLES_FILE = os.path.join(KB_DIR, "triples.json")

with open(TRIPLES_FILE, "r", encoding="utf-8") as f:
    triples_data = json.load(f)

# Convert to PyKEEN format: numpy array of shape (n, 3)
triples = np.array([[t["head"], t["relation"], t["tail"]] for t in triples_data], dtype=str)

# Create TriplesFactory
tf = TriplesFactory.from_labeled_triples(triples)

# Train TransE model
result = pipeline(
    model='TransE',
    training=tf,
    testing=tf,  # Use same for now; split later if needed
    model_kwargs=dict(embedding_dim=50),
    training_kwargs=dict(num_epochs=10, batch_size=1024),  # Reduced epochs for testing
    random_seed=42,
)

# Save the model
model_path = os.path.join(KB_DIR, "transe_model")
result.save_to_directory(model_path)

# Extract entity embeddings
entity_embeddings = result.model.entity_representations[0](indices=None).detach().cpu()
torch.save(entity_embeddings, os.path.join(KB_DIR, "entity_embeddings_transe.pt"))

print(f"[SUCCESS] TransE trained and saved to {model_path}")
print(f"Entity embeddings shape: {entity_embeddings.shape}")