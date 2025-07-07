"""Utility functions for caching Google SERP results from SearchAPI.io.

If the cache file already contains results for a given query, those are
returned immediately. Otherwise, the SearchAPI.io endpoint is called and the
results are persisted for future runs.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import requests
import os
from fake_useragent import UserAgent

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
    cache[query] = fresh_results
    _save_cache(cache)
    logging.info("Saved results to cache for query: '%s'", query)
    return fresh_results


def get_shopping_results_from_serpapi(query: str, api_key: str, **kwargs) -> Dict[str, Any]:
    """Return Google Shopping results for query, using local cache when possible.
    
    Args:
        query: Search query for shopping results
        api_key: SearchApi.io API key
        **kwargs: Additional parameters:
            - gl: Country code (default "us")
            - hl: Language code (default "en") 
            - location: Geographic location
            - page: Page number (default 1)
            - price_min: Minimum price filter
            - price_max: Maximum price filter
            - condition: Product condition ("new" or "used")
            - shoprs: Encoded filters
    
    Returns:
        Dictionary containing shopping results from SearchApi.io
    """
    cache = _load_cache()
    
    # Use simple query as cache key with shopping prefix
    cache_key = f"shopping_{query}"
    
    if cache_key in cache:
        logging.info("Using cached shopping results for query: '%s'", query)
        return cache[cache_key]

    logging.info("Fetching shopping results from SearchApi.io for query: '%s'", query)
    fresh_results = _fetch_shopping_from_searchapi(query, api_key, **kwargs)
    cache[cache_key] = fresh_results
    _save_cache(cache)
    logging.info("Saved shopping results to cache for query: '%s'", query)
    return fresh_results


def _fetch_shopping_from_searchapi(query: str, api_key: str, **kwargs) -> Dict[str, Any]:
    """Call SearchApi.io Google Shopping API and return parsed JSON response."""
    url = "https://www.searchapi.io/api/v1/search"
    params = {
        "engine": "google_shopping",
        "q": query,
        "api_key": api_key,
        "gl": kwargs.get("gl", "us"),
        "hl": kwargs.get("hl", "en"),
        "page": kwargs.get("page", 1)
    }
    
    # Add optional parameters if provided
    optional_params = ["location", "price_min", "price_max", "condition", "shoprs"]
    for param in optional_params:
        if param in kwargs and kwargs[param] is not None:
            params[param] = kwargs[param]
    
    
    session = requests.session()
    session.proxies = {}
    session.proxies["http"] = "socks5h://localhost:9150"
    session.proxies["https"] = "socks5h://localhost:9150"
    # Update only the User-Agent header
    session.headers.update(
        {"User-Agent": UserAgent().random}
    )  # Empty User-Agent or any custom value

    response = session.get(url, params=params, timeout=15)
    response.raise_for_status()
    return response.json()

def get_shopping_results(product_list: List[dict]) -> Dict[str, Any]:

    final_results = []

    if len(product_list) == 1:

        results = get_shopping_results_from_serpapi(product_list[0]['product_name'], api_key=os.getenv("SERP_API_KEY"))

        top5_results = results['shopping_results'][:5]



        for result in top5_results:
            final_results.append({
                "product_name": product_list[0]['product_name'],
                # "product_id": result['product_id'],
                "title": result['title'],
                "product_link": result['product_link'],
                'price': result['price'],
                'thumbnail': result['thumbnail'],
            })
        
    
    else:


        for product in product_list:

            # Get the first one from the shopping results
            results = get_shopping_results_from_serpapi(product['product_name'], api_key=os.getenv("SERP_API_KEY"))

            
            first_result = results['shopping_results'][0]

            final_results.append({
                "product_name": product['product_name'],
                # "product_id": first_result['product_id'],
                "title": first_result['title'],
                "product_link": first_result['product_link'],
                'price': first_result['price'],
                'thumbnail': first_result['thumbnail'],
            })
    return final_results

if __name__ == "__main__":
    product_list = [
        {
            "product_name": "Bose QuietComfort 45"
        },
        # {
        #     "product_name": "Sony WH-1000XM4"
        # },
        # {
        #     "product_name": "Apple AirPods Pro"
        # }
    ]
    print(get_shopping_results(product_list))