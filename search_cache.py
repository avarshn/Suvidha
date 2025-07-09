"""Utility functions for caching Google SERP results from SearchAPI.io.

If the cache file already contains results for a given query, those are
returned immediately. Otherwise, the SearchAPI.io endpoint is called and the
results are persisted for future runs.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any
import requests

CACHE_FILE: Path = Path("search_cache.json")

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _load_cache() -> Dict[str, Any]:
    if CACHE_FILE.exists():
        try:
            with CACHE_FILE.open("r", encoding="utf-8") as fp:
                return json.load(fp)
        except json.JSONDecodeError:
            # Corrupted cache → start fresh
            return {}
    return {}


def _save_cache(cache: Dict[str, Any]) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_FILE.open("w", encoding="utf-8") as fp:
        json.dump(cache, fp, ensure_ascii=False, indent=2)


def _fetch_from_api(query: str, api_key: str, engine: str = "google") -> Dict[str, Any]:
    """Call SearchAPI.io and return parsed JSON response."""
    url = "https://www.searchapi.io/api/v1/search"
    params = {"engine": engine, "q": query, "api_key": api_key}
    response = requests.get(url, params=params, timeout=10)
    logging.info(f"Response: {response}") 
    response.raise_for_status()
    return response.json()


def get_search_results(query: str, api_key: str, engine: str = "google") -> Dict[str, Any]:
    """Return SERP results for query, using local cache when possible."""
    cache = _load_cache()
    if query in cache:
        logging.info("Using cached SERP results for query: '%s'", query)
        return cache[query]

    logging.info("Fetching SERP results from API for query: '%s'", query)
    fresh_results = _fetch_from_api(query, api_key, engine)
    logging.info(f"Got the results: {fresh_results}") 
    cache[query] = fresh_results
    _save_cache(cache)
    logging.info("Saved results to cache for query: '%s'", query)
    return fresh_results 
