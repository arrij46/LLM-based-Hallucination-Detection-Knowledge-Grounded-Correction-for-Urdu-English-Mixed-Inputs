import requests
import re
import os
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize

HEADERS = {
    "User-Agent": "Mozilla/5.0 (KB-Builder/1.0)"
}

WIKI_URLS = {

    "culture": [
        "https://en.wikipedia.org/wiki/Culture_of_Pakistan",
        "https://en.wikipedia.org/wiki/Pakistani_clothing",
        "https://en.wikipedia.org/wiki/List_of_festivals_in_Pakistan",
        "https://en.wikipedia.org/wiki/Desi",
        "https://en.wikipedia.org/wiki/National_Heritage_and_Culture_Division_(Pakistan)",
        "https://en.wikipedia.org/wiki/Pakistani_cuisine",
        "https://en.wikipedia.org/wiki/Pakistani_music",
        "https://en.wikipedia.org/wiki/Pakistani_cinema",
        "https://en.wikipedia.org/wiki/Pakistani_literature",
        "https://en.wikipedia.org/wiki/Television_in_Pakistan",
        "https://en.wikipedia.org/wiki/Islam_in_Pakistan",
        "https://en.wikipedia.org/wiki/Languages_of_Pakistan",
        "https://en.wikipedia.org/wiki/Sports_in_Pakistan",
        "https://en.wikipedia.org/wiki/Pakistani_folk_music",
        "https://en.wikipedia.org/wiki/Architecture_of_Pakistan",
        "https://en.wikipedia.org/wiki/Education_in_Pakistan",
        "https://en.wikipedia.org/wiki/Ethnic_groups_in_Pakistan",
        "https://en.wikipedia.org/wiki/Pakistani_art",
        "https://en.wikipedia.org/wiki/Pakistani_poetry",
        "https://en.wikipedia.org/wiki/Sufism_in_Pakistan",
        "https://en.wikipedia.org/wiki/Religion_in_Pakistan",
        "https://en.wikipedia.org/wiki/Media_of_Pakistan",
        "https://en.wikipedia.org/wiki/Pakistani_fashion",
    ],
    "geography":[
        "https://en.wikipedia.org/wiki/Punjab,_Pakistan",
        "https://en.wikipedia.org/wiki/Geography_of_Pakistan",
        "https://en.wikipedia.org/wiki/Quetta",
        "https://en.wikipedia.org/wiki/Azad_Kashmir",
        "https://en.wikipedia.org/wiki/List_of_rivers_of_Pakistan",
        "https://en.wikipedia.org/wiki/Khyber_Pakhtunkhwa",
        "https://en.wikipedia.org/wiki/Sindh",
        "https://en.wikipedia.org/wiki/Mountains_of_Pakistan",
        "https://en.wikipedia.org/wiki/Indus_River",
        "https://en.wikipedia.org/wiki/Thar_Desert",
        "https://en.wikipedia.org/wiki/Balochistan,_Pakistan",
        "https://en.wikipedia.org/wiki/Gilgit-Baltistan",
        "https://en.wikipedia.org/wiki/Karakoram",
        "https://en.wikipedia.org/wiki/List_of_cities_in_Pakistan",
        "https://en.wikipedia.org/wiki/Climate_of_Pakistan",
        "https://en.wikipedia.org/wiki/Arabian_Sea",
        "https://en.wikipedia.org/wiki/Transport_in_Pakistan",
        "https://en.wikipedia.org/wiki/List_of_districts_of_Pakistan",
        "https://en.wikipedia.org/wiki/Administrative_units_of_Pakistan",
        "https://en.wikipedia.org/wiki/Earthquakes_in_Pakistan",
        "https://en.wikipedia.org/wiki/Floods_in_Pakistan",
        "https://en.wikipedia.org/wiki/National_parks_of_Pakistan",
        "https://en.wikipedia.org/wiki/Indus_Basin",
        "https://en.wikipedia.org/wiki/Lakes_of_Pakistan",
        "https://en.wikipedia.org/wiki/Environmental_issues_in_Pakistan"
    ],
    "history":[
        "https://en.wikipedia.org/wiki/Timeline_of_Pakistani_history",
        "https://en.wikipedia.org/wiki/Partition_of_India",
        "https://en.wikipedia.org/wiki/India%E2%80%93Pakistan_war_of_1947%E2%80%931948",
        "https://en.wikipedia.org/wiki/Languages_of_Pakistan",
        "https://en.wikipedia.org/wiki/Economic_history_of_Pakistan",
        "https://en.wikipedia.org/wiki/History_of_Pakistan_(1947%E2%80%93present)",
        "https://en.wikipedia.org/wiki/History_of_East_Pakistan",
        "https://en.wikipedia.org/wiki/Indus_Valley_Civilisation",
        "https://en.wikipedia.org/wiki/Mughal_Empire",
        "https://en.wikipedia.org/wiki/Delhi_Sultanate",
        "https://en.wikipedia.org/wiki/British_Raj",
        "https://en.wikipedia.org/wiki/Pakistan_Movement",
        "https://en.wikipedia.org/wiki/All-India_Muslim_League",
        "https://en.wikipedia.org/wiki/Muhammad_Ali_Jinnah",
        "https://en.wikipedia.org/wiki/Simla_Conference",
        "https://en.wikipedia.org/wiki/Radcliffe_Line",
        "https://en.wikipedia.org/wiki/Objectives_Resolution",
        "https://en.wikipedia.org/wiki/Military_history_of_Pakistan",
         "https://en.wikipedia.org/wiki/1956_Constitution_of_Pakistan",
        "https://en.wikipedia.org/wiki/1965_Indo-Pakistani_War",
        "https://en.wikipedia.org/wiki/1999_Pakistani_coup_d%27%C3%A9tat",
        "https://en.wikipedia.org/wiki/Zulfikar_Ali_Bhutto",
        "https://en.wikipedia.org/wiki/Benazir_Bhutto",
        "https://en.wikipedia.org/wiki/Nuclear_program_of_Pakistan"        
    ],
    "politics":[
        "https://en.wikipedia.org/wiki/Politics_of_Pakistan",
        "https://en.wikipedia.org/wiki/Corruption_in_Pakistan",
        "https://en.wikipedia.org/wiki/2022%E2%80%932024_Pakistan_political_unrest",
        "https://en.wikipedia.org/wiki/Women_in_Pakistani_politics",
        "https://en.wikipedia.org/wiki/Constitution_of_Pakistan",
        "https://en.wikipedia.org/wiki/Prime_Minister_of_Pakistan",
        "https://en.wikipedia.org/wiki/President_of_Pakistan",
        "https://en.wikipedia.org/wiki/Parliament_of_Pakistan",
        "https://en.wikipedia.org/wiki/Pakistan_Army",
        "https://en.wikipedia.org/wiki/Pakistan_Tehreek-e-Insaf",
        "https://en.wikipedia.org/wiki/Pakistan_Muslim_League_(N)",
        "https://en.wikipedia.org/wiki/Elections_in_Pakistan",
        "https://en.wikipedia.org/wiki/Judiciary_of_Pakistan",
        "https://en.wikipedia.org/wiki/Foreign_relations_of_Pakistan",
        "https://en.wikipedia.org/wiki/Government_of_Pakistan",
        "https://en.wikipedia.org/wiki/Civil_Service_of_Pakistan",
        "https://en.wikipedia.org/wiki/Supreme_Court_of_Pakistan",
        "https://en.wikipedia.org/wiki/Inter-Services_Intelligence",
        "https://en.wikipedia.org/wiki/National_Assembly_of_Pakistan",
        "https://en.wikipedia.org/wiki/Senate_of_Pakistan",
        "https://en.wikipedia.org/wiki/Human_rights_in_Pakistan",
        "https://en.wikipedia.org/wiki/Local_government_in_Pakistan",
        "https://en.wikipedia.org/wiki/Foreign_policy_of_Pakistan",
        "https://en.wikipedia.org/wiki/Terrorism_in_Pakistan"
    ]
}
OUTPUT_DIR = "./data_sources/english"
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
