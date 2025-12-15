# pipeline/utils/fallback_search.py

from typing import List, Dict


# -----------------------------
# Public API
# -----------------------------
def fallback_search(query: str) -> List[Dict]:
    """
    Triggered when retrieval confidence is low.
    Returns a list of alternative fact documents.
    """

    results = []

    # Trusted fallback sources (mocked)
    results.extend(search_wikipedia(query))
    results.extend(search_bbc_urdu(query))
    results.extend(search_dawn(query))

    return results


# -----------------------------
# Mock Search Providers
# -----------------------------
def search_wikipedia(query: str) -> List[Dict]:
    return [{
        "text": f"{query} ka verified jawab Wikipedia ke mutabiq",
        "source": "Wikipedia",
        "date": "2024"
    }]


def search_bbc_urdu(query: str) -> List[Dict]:
    return [{
        "text": f"BBC Urdu ke mutabiq {query} ki tafseel",
        "source": "BBC Urdu",
        "date": "2024"
    }]


def search_dawn(query: str) -> List[Dict]:
    return [{
        "text": f"Dawn News report: {query}",
        "source": "Dawn",
        "date": "2024"
    }]
