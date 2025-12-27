import requests
import re
import os
import json
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize

WIKI_URLS_LOC = "./data_sources/"
WIKI_URLS = None
HEADERS = {
    "User-Agent": "Mozilla/5.0 (KB-Builder/1.0)"
}

with open(os.path.join(WIKI_URLS_LOC, "wikiURLs.json"), "r", encoding="utf-8") as f:
    WIKI_URLS = json.load(f)


OUTPUT_DIR = "./data_sources"
MIN_LEN = 40
MAX_LEN = 200

def clean_text(text):
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[citation needed\]", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def is_good_sentence(s):
    if len(s) < MIN_LEN or len(s) > MAX_LEN:
        return False
    if s.count(",") > 3:
        return False
    if s.lower().startswith(("this", "these", "it", "they")):
        return False
    return s.endswith(".")

def scrape_url(url):
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    paragraphs = soup.find_all("p")

    sentences = set()

    for p in paragraphs:
        text = clean_text(p.get_text())
        for s in sent_tokenize(text):
            s = s.strip()
            if is_good_sentence(s):
                sentences.add(s)

    return sentences

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for category, urls in WIKI_URLS.items():
        print(f"\n[INFO] Processing category: {category.upper()}")

        all_sentences = set()

        for url in urls:
            try:
                print(f"  → Scraping {url}")
                sents = scrape_url(url)
                all_sentences.update(sents)
            except Exception as e:
                print(f"[ERROR] {category} | {url}: {e}")

        out_path = os.path.join(OUTPUT_DIR, f"{category}.txt")

        with open(out_path, "w", encoding="utf-8") as f:
            for s in sorted(all_sentences):
                f.write(s + "\n")

        print(f"[SUCCESS] Saved {len(all_sentences)} sentences → {out_path}")

if __name__ == "__main__":
    main()
