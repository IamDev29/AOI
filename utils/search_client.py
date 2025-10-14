import os
import time
import requests
from urllib.parse import urlparse


GOOGLE_SERP_API = "https://serpapi.com/search"
_CACHE: dict[str, tuple[float, dict]] = {}


def google_search_marking(query: str, num: int = 5):
    """Query SerpAPI Google Search with the provided query.
    Requires SERPAPI_KEY environment variable.
    """
    key = os.getenv("SERPAPI_KEY")
    if not key:
        return {"error": "SERPAPI_KEY missing"}

    params = {
        "engine": "google",
        "q": query,
        "num": num,
        "api_key": key,
    }
    try:
        r = requests.get(GOOGLE_SERP_API, params=params, timeout=15)
        if not r.ok:
            return {"error": f"HTTP {r.status_code}", "text": r.text}
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def google_search_marking_cached(query: str, num: int = 5, ttl_seconds: int = 600):
    """Cached wrapper for SerpAPI Google Search to avoid duplicate costs.
    Uses a simple in-memory cache keyed by the query string for the session lifetime.
    """
    now = time.time()
    key = f"q={query}&n={num}"
    hit = _CACHE.get(key)
    if hit and (now - hit[0]) < ttl_seconds:
        return hit[1]
    res = google_search_marking(query, num=num)
    _CACHE[key] = (now, res)
    return res


def extract_domains(results: list) -> list:
    domains = []
    for r in results:
        link = r.get("link") or r.get("url")
        if link:
            try:
                domains.append(urlparse(link).netloc)
            except Exception:
                pass
    return domains