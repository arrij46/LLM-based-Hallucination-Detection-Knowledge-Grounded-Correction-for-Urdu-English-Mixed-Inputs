import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

from pipeline.utils.clustering import cluster_embeddings
from pipeline.utils.entropy import semantic_entropy

# ================= CONFIG =================

STAGE1_OUTPUT = Path("data/stage1_output.json")
OUTPUT_FILE = Path("data/stage2_output.json")

GEN_MODEL = "bigscience/bloom-560m"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

NUM_GENERATIONS = 5
ENTROPY_THRESHOLD = 1.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================


def load_generator():
    print("Loading multilingual generator (Alif-compatible fallback)...")

    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForCausalLM.from_pretrained(GEN_MODEL)

    model.to(DEVICE)
    model.eval()

    return tokenizer, model


def generate_responses(prompt, tokenizer, model):
    prompt = f"Question: {prompt}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        num_return_sequences=NUM_GENERATIONS,
        pad_token_id=tokenizer.eos_token_id
    )

    responses = []
    for o in outputs:
        text = tokenizer.decode(o, skip_special_tokens=True)
        text = text.split("Answer:")[-1].strip()
        responses.append(text)

    return responses


def detect_hallucination(stage1_items):
    tokenizer, model = load_generator()

    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL)

    results = []

    for item in stage1_items:
        prompt = item["normalized_text"]

        responses = generate_responses(prompt, tokenizer, model)
        embeddings = embedder.encode(responses)

        clusters = cluster_embeddings(embeddings)
        entropy = semantic_entropy(clusters)

        result = {
            "original_text": item["original_text"],
            "normalized_text": prompt,
            "responses": responses,
            "clusters": clusters,
            "entropy_score": round(entropy, 3),
            "hallucination_detected": entropy > ENTROPY_THRESHOLD
        }

        results.append(result)

    return results


if __name__ == "__main__":
    with open(STAGE1_OUTPUT, "r", encoding="utf-8") as f:
        stage1_data = json.load(f)

    output = detect_hallucination(stage1_data)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("✅ Stage-2 completed successfully.")
    print(f"📁 Output saved to {OUTPUT_FILE}")