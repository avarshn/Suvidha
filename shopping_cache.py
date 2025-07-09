"""Utility functions for caching Google Shopping SERP results from SearchAPI.io.

This module mirrors `search_cache.py` but uses a separate on-disk JSON file
(`shopping_cache.json`) so Reddit/organic lookups and Shopping lookups do not
interfere with each other.  The SearchAPI response structure is identical, so
we simply cache the entire JSON response keyed by query.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any
from serpapi import GoogleSearch
import requests

CACHE_FILE: Path = Path("shopping_cache.json")

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


def _fetch_from_api(query: str, api_key: str, engine: str = "google_shopping") -> Dict[str, Any]:
    """Call SerpAPI GoogleShopping engine and return JSON response."""

    params = {
        "engine": engine,
        "q": query,
        "api_key": api_key,
        "gl": "us",
        "hl": "en",
        "num": 5,
    }

    search = GoogleSearch(params)
    return search.get_dict()


def get_shopping_results(query: str, api_key: str, engine: str = "google_shopping") -> Dict[str, Any]:
    """Return SERP results (including `shopping_results`) for the query.

    Uses an on-disk JSON cache (`shopping_cache.json`) to avoid repeated calls
    for the same query.  If the query is not cached, we hit the SearchAPI.
    """
    # Use normalised key so that 'Sony WH-1000XM5' and 'sony wh-1000xm5 ' map to same cache entry
    key = " ".join(query.lower().strip().split())  # collapse whitespace + lowercase

    cache = _load_cache()
    if key in cache:
        logging.info("Using cached Shopping SERP results for product: '%s'", key)
        return cache[key]

    logging.info("Fetching Shopping SERP results from API for product: '%s'", key)
    fresh_results = _fetch_from_api(query, api_key, engine)
    cache[key] = fresh_results
    _save_cache(cache)
    logging.info("Saved %d Shopping results to cache for product: '%s'", len(fresh_results.get("shopping_results", [])), key)
    return fresh_results 