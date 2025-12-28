"""
Fallback search module for Stage 4 verification.
Uses multiple official, free APIs for comprehensive fact retrieval:
1. Wikipedia (MediaWiki API)
2. World Bank Indicators API
3. REST Countries API (for country information)
4. DuckDuckGo Instant Answer API
5. Open Library API (for books/authors)
"""

from typing import List, Dict
import requests
import json

HEADERS = {
    "User-Agent": "NLP-Hallucination-Detection/1.0 (academic-project)"
}

# ------------------------------------------------------------------
# Main Entry
# ------------------------------------------------------------------
def fallback_search(query: str, max_results: int = 10) -> List[Dict]:
    """
    Comprehensive fallback search using multiple free APIs.
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
    
    Returns:
        List of document dicts with 'text', 'source', 'date' fields
    """

    if not query or not query.strip():
        return []

    results = []

    try:
        # Try all search sources
        results.extend(search_wikipedia(query, limit=5))
        results.extend(search_world_bank(query, limit=3))
        results.extend(search_rest_countries(query, limit=3))
        results.extend(search_duckduckgo(query, limit=3))
        results.extend(search_open_library(query, limit=3))

        # Remove duplicates based on text similarity
        seen = set()
        unique = []
        for doc in results:
            key = doc["text"].lower().strip()
            # Normalize key for better duplicate detection
            key = " ".join(key.split()[:20])  # First 20 words
            if key and key not in seen and len(key) > 10:  # Minimum length
                seen.add(key)
                unique.append(doc)

        return unique[:max_results]

    except Exception as e:
        print(f"[Fallback Search Error] {e}")
        return []


# ------------------------------------------------------------------
# Wikipedia Search (FIXED)
# ------------------------------------------------------------------
def search_wikipedia(query: str, limit: int = 3) -> List[Dict]:
    """
    Wikipedia MediaWiki API (User-Agent REQUIRED)
    """

    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": limit
    }

    response = requests.get(url, params=params, headers=HEADERS, timeout=10)
    response.raise_for_status()

    data = response.json()
    results = []

    for item in data.get("query", {}).get("search", []):
        title = item.get("title")
        page_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"

        snippet = item.get("snippet", "")
        snippet = snippet.replace("<span class=\"searchmatch\">", "").replace("</span>", "")

        results.append({
            "text": snippet,
            "source": "Wikipedia",
            "date": "N/A",
            "url": page_url
        })

    return results


# ------------------------------------------------------------------
# World Bank Search
# ------------------------------------------------------------------
def search_world_bank(query: str, limit: int = 3) -> List[Dict]:

    url = "https://api.worldbank.org/v2/indicator"
    params = {
        "format": "json",
        "per_page": limit
    }

    response = requests.get(url, params=params, headers=HEADERS, timeout=10)
    response.raise_for_status()

    data = response.json()
    results = []

    if not isinstance(data, list) or len(data) < 2:
        return results

    for indicator in data[1]:
        name = indicator.get("name", "")
        if query.lower() in name.lower():
            results.append({
                "text": f"World Bank indicator: {name}",
                "source": "World Bank",
                "date": indicator.get("sourceNote", "N/A"),
                "url": "https://data.worldbank.org"
            })

    return results


# ------------------------------------------------------------------
# REST Countries API
# ------------------------------------------------------------------
def search_rest_countries(query: str, limit: int = 3) -> List[Dict]:
    """Search for country information using REST Countries API"""
    try:
        # Extract potential country name from query
        query_lower = query.lower()
        country_keywords = ['country', 'capital', 'population', 'pakistan', 'india', 'bangladesh', 
                          'afghanistan', 'iran', 'china', 'usa', 'united states', 'uk', 'britain']
        
        if not any(keyword in query_lower for keyword in country_keywords):
            return []
        
        # Try to find country name in query
        url = "https://restcountries.com/v3.1/all"
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        
        countries = response.json()
        results = []
        
        for country in countries[:50]:  # Limit search to first 50 for performance
            name = country.get("name", {}).get("common", "").lower()
            if name in query_lower or query_lower in name:
                capital = country.get("capital", ["N/A"])[0]
                population = country.get("population", 0)
                region = country.get("region", "N/A")
                
                text = f"{name.title()} is a country in {region}. Capital: {capital}. Population: {population:,}"
                results.append({
                    "text": text,
                    "source": "REST Countries",
                    "date": "2024",
                    "url": f"https://restcountries.com/v3.1/name/{name}"
                })
                if len(results) >= limit:
                    break
        
        return results
    except Exception:
        return []


# ------------------------------------------------------------------
# DuckDuckGo Instant Answer API
# ------------------------------------------------------------------
def search_duckduckgo(query: str, limit: int = 3) -> List[Dict]:
    """Search DuckDuckGo Instant Answer API (free, no API key)"""
    try:
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1"
        }
        
        response = requests.get(url, params=params, headers=HEADERS, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        # Extract Abstract
        abstract = data.get("AbstractText", "")
        if abstract:
            results.append({
                "text": abstract,
                "source": "DuckDuckGo",
                "date": "2024",
                "url": data.get("AbstractURL", "")
            })
        
        # Extract Related Topics
        related_topics = data.get("RelatedTopics", [])
        for topic in related_topics[:limit-1]:
            if isinstance(topic, dict) and "Text" in topic:
                results.append({
                    "text": topic["Text"],
                    "source": "DuckDuckGo",
                    "date": "2024",
                    "url": topic.get("FirstURL", "")
                })
        
        return results[:limit]
    except Exception:
        return []


# ------------------------------------------------------------------
# Open Library API
# ------------------------------------------------------------------
def search_open_library(query: str, limit: int = 3) -> List[Dict]:
    """Search Open Library API for books and authors"""
    try:
        # Check if query is about books/authors
        book_keywords = ['book', 'author', 'writer', 'novel', 'poet', 'poetry', 'literature']
        if not any(keyword in query.lower() for keyword in book_keywords):
            return []
        
        url = "https://openlibrary.org/search.json"
        params = {
            "q": query,
            "limit": limit
        }
        
        response = requests.get(url, params=params, headers=HEADERS, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        docs = data.get("docs", [])
        for doc in docs[:limit]:
            title = doc.get("title", "Unknown")
            author = ", ".join(doc.get("author_name", []))
            first_publish_year = doc.get("first_publish_year", "N/A")
            
            text = f"{title}"
            if author:
                text += f" by {author}"
            if first_publish_year != "N/A":
                text += f" (Published: {first_publish_year})"
            
            results.append({
                "text": text,
                "source": "Open Library",
                "date": str(first_publish_year) if first_publish_year != "N/A" else "N/A",
                "url": f"https://openlibrary.org{doc.get('key', '')}"
            })
        
        return results
    except Exception:
        return []
