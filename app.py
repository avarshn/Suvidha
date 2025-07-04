import os
from typing import List

import streamlit as st
from dotenv import load_dotenv

from search_cache import get_search_results
from main import SearchAPIResponse, RedditPost, fetch_reddit_post

# -----------------------------
# Setup
# -----------------------------
load_dotenv()
API_KEY_ENV_VAR = "SERP_API_KEY"
api_key = os.getenv(API_KEY_ENV_VAR, "")

st.set_page_config(page_title="Reddit SERP Explorer", layout="wide")
st.title("🔎 Reddit SERP Explorer")

if not api_key:
    st.error(
        f"Environment variable `{API_KEY_ENV_VAR}` not found. "
        "Please set it in your .env file before running the app."
    )
    st.stop()

st.markdown(
    """
Enter a search query and this tool will:
1. Retrieve Google search results via SearchAPI.io (cached locally).
2. Filter for Reddit posts.
3. Display each post's title, description, and top-level comments.
"""
)

# User inputs
query = st.text_input("Search query", placeholder="e.g., good wireless headphones")

if st.button("Search") and query:
    with st.spinner("Fetching search results..."):
        try:
            raw_results = get_search_results(f"site:reddit.com {query}", api_key)
        except Exception as exc:
            st.error(f"Error fetching SERP results: {exc}")
            st.stop()

    search_response = SearchAPIResponse.from_json(raw_results)

    if not search_response.reddit_results:
        st.info("No Reddit results found.")
        st.stop()

    # Show each Reddit result
    for res in search_response.reddit_results:
        with st.spinner(f"Processing: {res.title}"):
            try:
                post: RedditPost = fetch_reddit_post(res.link)
            except Exception as exc:
                st.warning(f"Failed to fetch post data: {exc}")
                continue

        with st.expander(post.title, expanded=False):
            st.markdown(f"**Post URL:** [{post.link}]({post.link})")
            if post.description:
                st.markdown(f"**Description:** {post.description}")
            else:
                st.markdown("*(No description — link post)*")

            st.markdown("---")
            st.markdown(f"### Top-level comments ({len(post.comments)})")
            for c in post.comments:
                st.markdown(f"*{c.author}* — {c.body}")

else:
    st.info("Enter a query and click Search to begin.") 