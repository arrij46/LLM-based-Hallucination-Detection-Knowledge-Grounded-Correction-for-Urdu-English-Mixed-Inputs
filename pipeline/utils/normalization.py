import re
import pandas as pd

SPELLING_MAP = {}

def load_spelling_variants(csv_path: str):
    """
    Load Roman Urdu spelling variants from CSV.
    """
    global SPELLING_MAP

    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        common = str(row["Common"]).lower().strip()

        for col in ["var-1", "var-2", "var-3", "var-4", "var-5"]:
            variant = row.get(col)
            if pd.notna(variant):
                SPELLING_MAP[str(variant).lower().strip()] = common

        SPELLING_MAP[common] = common


def normalize_text(text: str) -> str:
    """
    Normalize Roman Urdu spellings.
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s?.!,]", "", text)
    text = re.sub(r"\s+", " ", text)

    words = text.split()
    normalized = [SPELLING_MAP.get(w, w) for w in words]

    return " ".join(normalized)
