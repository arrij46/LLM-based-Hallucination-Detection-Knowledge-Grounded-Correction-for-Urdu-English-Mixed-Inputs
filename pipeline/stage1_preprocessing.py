import json
from transformers import XLMRobertaTokenizer
from pipeline.utils.normalization import load_spelling_variants, normalize_text
from pipeline.utils.lang_id import identify_language, tag_code_switches

# Load spelling variants
load_spelling_variants("data/spelling_variation.csv")

# Load tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")


def preprocess_text(text: str) -> dict:
    normalized_text = normalize_text(text)

    tokens = tokenizer.tokenize(normalized_text)
    clean_tokens = [t.replace("▁", "") for t in tokens]

    language_tags = [identify_language(tok) for tok in clean_tokens]
    code_switch_points = tag_code_switches(language_tags)

    return {
        "original_text": text,
        "normalized_text": normalized_text,
        "tokens": tokens,
        "language_tags": language_tags,
        "code_switch_points": code_switch_points
    }


def run_on_dataset():
    # Load questions dataset
    with open("data/dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    outputs = []

    for question in data["questions"]:
        processed = preprocess_text(question)
        outputs.append(processed)

    # Save output
    with open("data/stage1_output.json", "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(outputs)} questions.")
    print("Output saved to data/stage1_output.json")


if __name__ == "__main__":
    run_on_dataset()
