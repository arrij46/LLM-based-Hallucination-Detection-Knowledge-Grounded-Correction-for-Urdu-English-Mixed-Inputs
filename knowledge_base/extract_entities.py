import os
import json
import spacy

# Input and output paths
INPUT_DIR = "./data_sources/english"
OUTPUT_DIR = "./entities/english"
ALL_ENTITIES_FILE = os.path.join(OUTPUT_DIR, "all_entities.json")
DEDUP_ENTITIES_FILE = os.path.join(OUTPUT_DIR, "entities.json")

# Allowed NER labels
ALLOWED_LABELS = {"PERSON", "ORG", "GPE", "LOC", "EVENT", "DATE", "NORP", "FAC"}

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


def extract_entities(sentence):
    doc = nlp(sentence)
    entities = []
    for ent in doc.ents:
        if ent.label_ in ALLOWED_LABELS:
            entities.append({
                "text": ent.text.strip(),
                "label": ent.label_,
                "id": ent.text.strip().replace(" ", "_")
            })
    return entities


def process_file(file_path, category):
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            sentence = line.strip()
            if not sentence:
                continue

            ents = extract_entities(sentence)
            if not ents:
                continue

            records.append({
                "category": category,
                "sentence": sentence,
                "entities": ents
            })
    return records


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_records = []
    all_entities_set = set()
    dedup_entities_list = []

    for file in os.listdir(INPUT_DIR):
        if file.endswith(".txt"):
            category = file.replace(".txt", "")
            file_path = os.path.join(INPUT_DIR, file)
            records = process_file(file_path, category)
            all_records.extend(records)

            # Collect deduplicated entities
            for record in records:
                for ent in record["entities"]:
                    key = (ent["text"], ent["label"])
                    if key not in all_entities_set:
                        all_entities_set.add(key)
                        dedup_entities_list.append({
                            "id": ent["text"],
                            "type": ent["label"]
                        })

    # Write all sentences with entities
    with open(ALL_ENTITIES_FILE, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    # Write deduplicated entities
    with open(DEDUP_ENTITIES_FILE, "w", encoding="utf-8") as f:
        json.dump(dedup_entities_list, f, ensure_ascii=False, indent=2)

    print(f"[SUCCESS] Combined {len(all_records)} sentences to {ALL_ENTITIES_FILE}")
    print(f"[SUCCESS] Saved {len(dedup_entities_list)} unique entities to {DEDUP_ENTITIES_FILE}")


if __name__ == "__main__":
    main()

# import os
# import json
# import spacy

# INPUT_DIR = "../data_sources/english"
# OUTPUT_DIR = "../entities/english"

# ALLOWED_LABELS = {
#     "PERSON",
#     "ORG",
#     "GPE",
#     "LOC",
#     "EVENT",
#     "DATE",
#     "NORP",
#     "FAC"
# }

# nlp = spacy.load("en_core_web_sm")

# def extract_entities(sentence):
#     doc = nlp(sentence)
#     entities = []

#     for ent in doc.ents:
#         if ent.label_ in ALLOWED_LABELS:
#             entities.append({
#                 "text": ent.text,
#                 "label": ent.label_
#             })

#     return entities

# def process_file(category):
#     in_path = os.path.join(INPUT_DIR, f"{category}.txt")
#     out_path = os.path.join(OUTPUT_DIR, f"{category}.json")

#     records = []

#     with open(in_path, "r", encoding="utf-8") as f:
#         for line in f:
#             sentence = line.strip()
#             if not sentence:
#                 continue

#             ents = extract_entities(sentence)
#             if len(ents) < 1:
#                 continue

#             records.append({
#                 "sentence": sentence,
#                 "entities": ents
#             })

#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     with open(out_path, "w", encoding="utf-8") as f:
#         json.dump(records, f, indent=2, ensure_ascii=False)

#     print(f"[SUCCESS] {category}: {len(records)} sentences with entities")

# def main():
#     for file in os.listdir(INPUT_DIR):
#         if file.endswith(".txt"):
#             category = file.replace(".txt", "")
#             process_file(category)

# if __name__ == "__main__":
#     main()
