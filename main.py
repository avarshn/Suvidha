import requests
from pprint import pprint
import os
from dotenv import load_dotenv
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from search_cache import get_search_results  # Local caching


def search_google(query: str, api_key: str, engine: str = "google") -> dict:
    """Fetch search results from SearchAPI.io.

    Args:
        query: Search query string.
        api_key: Your SearchAPI.io API key.
        engine: Search engine to use (default is "google").

    Returns:
        Parsed JSON response as a Python dictionary.
    """
    url = "https://www.searchapi.io/api/v1/search"
    params = {
        "engine": engine,
        "q": query,
        "api_key": api_key,
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


@dataclass
class RedditResult:
    position: int
    title: str
    link: str
    source: str
    domain: str
    displayed_link: str
    snippet: str
    # snippet_highlighted_words: List[str]
    favicon: str


@dataclass
class SearchAPIResponse:
    reddit_results: List[RedditResult]

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "SearchAPIResponse":
        """Parse SearchAPI.io JSON response into structured objects."""
        organic_results = data["organic_results"]  # KeyError if missing
        reddit_items: List[RedditResult] = []
        for item in organic_results:
            # Only include Reddit domain
            if "reddit.com" not in item["domain"]:
                continue
            reddit_items.append(
                RedditResult(
                    position=item["position"],
                    title=item["title"],
                    link=item["link"],
                    source=item["source"],
                    domain=item["domain"],
                    displayed_link=item["displayed_link"],
                    snippet=item["snippet"],
                    # snippet_highlighted_words=item["snippet_highlighted_words"],
                    favicon=item["favicon"],
                )
            )
        return cls(reddit_results=reddit_items)


@dataclass
class RedditComment:
    id: str
    author: str
    body: str
    score: int
    created_utc: int
    depth: int


@dataclass
class RedditPost:
    title: str
    description: str  # Self-text or empty for link posts
    link: str
    comments: List[RedditComment]


def fetch_reddit_post(
    post_url: str, limit: Optional[int] = None
) -> RedditPost:
    """Fetch Reddit post metadata and top-level comments.

    Nested comment threads are ignored; only first-level comments are returned.
    """
    json_url = post_url.rstrip("/") + ".json"
    headers = {"User-Agent": "python:reddit.comment.fetch:v0.1 (by /u/anon)"}
    response = requests.get(json_url, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()

    # Post meta information is in the first element
    post_data = data[0]["data"]["children"][0]["data"]
    title = post_data.get("title", "")
    description = post_data.get("selftext", "")

    # Top-level comments
    comments_listing = data[1]["data"]["children"]
    comments: List[RedditComment] = []
    for child in comments_listing:
        if child.get("kind") != "t1":
            continue
        cdata = child["data"]
        comments.append(
            RedditComment(
                id=cdata["id"],
                author=cdata.get("author", "[deleted]"),
                body=cdata.get("body", ""),
                score=cdata.get("score", 0),
                created_utc=cdata.get("created_utc", 0),
                depth=0,
            )
        )
        if limit is not None and len(comments) >= limit:
            break

    return RedditPost(title=title, description=description, link=post_url, comments=comments)


def main() -> None:
    """Run a demo search and print the top organic results."""
    # Load environment variables from .env file
    load_dotenv()

    # Fetch the API key securely from environment variables
    api_key = os.getenv("SERP_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "SEARCH_API_KEY not found in environment variables. Please set it in your .env file."
        )

    # Use Google dorking to restrict search results to Reddit
    query = "site:reddit.com good wireless headphones"

    try:
        raw_results = get_search_results(query, api_key)
    except requests.RequestException as exc:
        print(f"Error fetching search results: {exc}")
        return

    # Transform into structured objects
    search_response = SearchAPIResponse.from_json(raw_results)
    reddit_results = search_response.reddit_results

    if not reddit_results:
        print("No Reddit results found.")
        return

    print(f"Reddit results for '{query}':\n")
    for res in reddit_results:
        # Fetch post metadata and top-level comments
        try:
            post = fetch_reddit_post(res.link)
        except Exception as exc:
            print(f"Failed to fetch post data for {res.link}: {exc}\n")
            continue

        print(f"Title: {post.title}")
        print(f"Description: {post.description[:200]}{'...' if len(post.description) > 200 else ''}\n")

        print(f"Top-level comments ({len(post.comments)}):")
        print("-" * 60)


if __name__ == "__main__":
    main()
