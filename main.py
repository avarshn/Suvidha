import requests
from pprint import pprint
import os
from dotenv import load_dotenv
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from search_cache import get_search_results  # Local caching
import praw
import re
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


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


def get_reddit_instance() -> praw.Reddit:
    """Create and return a Reddit API instance using environment variables."""
    load_dotenv()
    
    reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
    reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    reddit_user_agent = os.getenv("REDDIT_USER_AGENT", "python:suvidha.shopping.assistant:v1.0 (by /u/suvidha_bot)")
    
    if not reddit_client_id or not reddit_client_secret:
        raise EnvironmentError(
            "REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET must be set in environment variables. "
            "Get these from https://www.reddit.com/prefs/apps/"
        )
    
    return praw.Reddit(
        client_id=reddit_client_id,
        client_secret=reddit_client_secret,
        user_agent=reddit_user_agent,
        check_for_async=False
    )

def extract_post_id_from_url(post_url: str) -> str:
    """Extract Reddit post ID from various Reddit URL formats."""
    # Handle different Reddit URL formats
    patterns = [
        r'/comments/([a-zA-Z0-9]+)/',  # Standard format
        r'/r/[^/]+/comments/([a-zA-Z0-9]+)/',  # With subreddit
        r'reddit\.com/([a-zA-Z0-9]+)/?$',  # Short format
    ]
    
    for pattern in patterns:
        match = re.search(pattern, post_url)
        if match:
            return match.group(1)
    
    raise ValueError(f"Could not extract post ID from URL: {post_url}")

def fetch_reddit_post(
    post_url: str, limit: Optional[int] = None
) -> RedditPost:
    """Fetch Reddit post metadata and top-level comments using Reddit Official API.

    Nested comment threads are ignored; only first-level comments are returned.
    """
    try:
        reddit = get_reddit_instance()
        post_id = extract_post_id_from_url(post_url)
        
        # Fetch the submission
        submission = reddit.submission(id=post_id)
        
        # Get post metadata
        title = submission.title
        description = submission.selftext if hasattr(submission, 'selftext') else ""
        
        # Get top-level comments
        submission.comments.replace_more(limit=0)  # Remove "more comments" placeholders
        comments: List[RedditComment] = []
        
        for comment in submission.comments:
            if hasattr(comment, 'body') and hasattr(comment, 'author'):
                # Skip deleted comments
                if comment.body in ['[deleted]', '[removed]'] or not comment.author:
                    continue
                    
                comments.append(
                    RedditComment(
                        id=comment.id,
                        author=str(comment.author) if comment.author else "[deleted]",
                        body=comment.body,
                        score=comment.score,
                        created_utc=int(comment.created_utc),
                        depth=0,
                    )
                )
                
                if limit is not None and len(comments) >= limit:
                    break
        
        return RedditPost(
            title=title, 
            description=description, 
            link=post_url, 
            comments=comments
        )
        
    except Exception as e:
        # Fallback to the old method if Reddit API fails
        print(f"Reddit API failed for {post_url}, falling back to direct method: {e}")
        return fetch_reddit_post_fallback(post_url, limit)

def fetch_reddit_post_fallback(
    post_url: str, limit: Optional[int] = None
) -> RedditPost:
    """Fallback method using direct JSON endpoint access."""
    json_url = post_url.rstrip("/") + ".json"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    response = requests.get(json_url, headers=headers, timeout=10)
    logging.info(f"response from reddit post: {response}")
    response.raise_for_status()
    data = response.json()

    logging.info(f"Data from reddit post: {data}")

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
    api_key = os.getenv("SEARCH_API_KEY")
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
