import streamlit as st
from typing import Annotated, TypedDict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import os
from dotenv import load_dotenv
from search_cache import get_search_results  # Local caching
from main import SearchAPIResponse, RedditResult, fetch_reddit_post

# Load environment variables
load_dotenv()

# Environment setup
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize LLM
@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7
    )

# State definition
class BotState(TypedDict):
    messages: Annotated[list, add_messages]
    content: dict  # Holds product content in JSON format

# System instruction
SYS_INSTRUCTION = """You are a shopping assistant helping users find products. 
Use the provided product content and help the user in making informed decisions. 
You can give information about products and all from the content provided, if user asks."""

WELCOME_MSG = "Welcome to Suvidha! How can I assist you with your shopping needs today?"

# Product content retrieval
def get_content(query: str) -> dict:
    """Fetches structured product data based on user query"""
    query = "site:reddit.com " + query

    with st.spinner(f"Fetching product context for: {query}"):
        results = get_search_results(query, api_key=os.getenv("SERP_API_KEY"))

        # Transform into structured objects
        search_response = SearchAPIResponse.from_json(results)
        reddit_results = search_response.reddit_results

        product_data = []

        for reddit_result in reddit_results:
            # Fetch post metadata and top-level comments
            try:
                post = fetch_reddit_post(reddit_result.link)
                product_data.append({
                    "title": post.title,
                    "description": post.description[:200] + ('...' if len(post.description) > 200 else ''),
                    "link": post.link,
                    "comments": [
                        {
                            "id": comment.id,
                            "author": comment.author,
                            "body": comment.body[:150] + ('...' if len(comment.body) > 150 else ''),
                            "score": comment.score
                        }
                        for comment in post.comments[:5]  # Limit to first 5 comments
                    ]
                })
            except Exception as exc:
                st.warning(f"Failed to fetch post data for {reddit_result.link}: {exc}")
                continue

        return product_data

# Node: Generate assistant response
def generate_response(state: BotState, user_input: str) -> tuple[str, dict]:
    """Generate assistant response and return updated content"""
    
    # Add user message to state
    state["messages"].append(HumanMessage(content=user_input))
    
    # Check if this is the first user message (after welcome) - fetch content
    human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    
    if len(human_messages) == 1 and not state["content"]:
        state["content"] = get_content(user_input)
        st.success(f"Fetched product context for: {user_input}")

    # Create system message with content
    sys_msg = SystemMessage(content=(
        SYS_INSTRUCTION + 
        f"\n\nProduct Context:\n{json.dumps(state['content'], indent=2)}"
    ) if state["content"] else SYS_INSTRUCTION)

    # Filter only HumanMessages for conversation
    conversation_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]

    # Safety check
    if not conversation_messages:
        response_content = "Please tell me what you're looking for today!"
    else:
        # Generate LLM response
        messages = [sys_msg] + conversation_messages
        response = get_llm().invoke(messages)
        response_content = response.content

    # Add AI response to state
    state["messages"].append(AIMessage(content=response_content))
    
    return response_content, state["content"]

# Initialize session state
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [AIMessage(content=WELCOME_MSG)]
    if "content" not in st.session_state:
        st.session_state.content = {}
    if "bot_state" not in st.session_state:
        st.session_state.bot_state = {
            "messages": st.session_state.messages,
            "content": st.session_state.content
        }

def main():
    st.set_page_config(
        page_title="Suvidha - Shopping Assistant",
        page_icon="🛒",
        layout="wide"
    )
    
    st.title("🛒 Suvidha - Shopping Assistant")
    st.markdown("Get personalized product recommendations based on Reddit discussions!")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for content information
    with st.sidebar:
        st.header("📊 Session Info")
        st.write(f"**Messages:** {len(st.session_state.messages)}")
        st.write(f"**Content Items:** {len(st.session_state.content) if st.session_state.content else 0}")
        
        if st.session_state.content:
            st.subheader("🔍 Current Context")
            st.write(f"**Total Posts:** {len(st.session_state.content)}")
            total_comments = sum(len(item['comments']) for item in st.session_state.content)
            st.write(f"**Total Comments:** {total_comments}")
            
            st.write("**Post Previews:**")
            for i, item in enumerate(st.session_state.content[:3]):  # Show first 3 items
                with st.expander(f"Post {i+1}: {item['title'][:40]}..."):
                    st.write(f"**Description:** {item['description'][:100]}...")
                    st.write(f"**Comments:** {len(item['comments'])}")
                    st.write(f"**Link:** [View on Reddit]({item['link']})")
                    
                    if item['comments']:
                        st.write("**Top Comment:**")
                        top_comment = max(item['comments'], key=lambda x: x['score'])
                        st.write(f"👤 {top_comment['author']} (Score: {top_comment['score']})")
                        st.write(f"💬 {top_comment['body'][:80]}...")
            
            if len(st.session_state.content) > 3:
                st.write(f"... and {len(st.session_state.content) - 3} more posts")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = [AIMessage(content=WELCOME_MSG)]
            st.session_state.content = {}
            st.session_state.bot_state = {
                "messages": st.session_state.messages,
                "content": st.session_state.content
            }
            st.rerun()

    # Display Reddit posts if available
    if st.session_state.content:
        st.subheader("📋 Reddit Posts Found")
        
        # Create tabs for better organization
        tab1, tab2 = st.tabs(["💬 Chat", "📰 Reddit Posts"])
        
        with tab2:
            st.write(f"Found {len(st.session_state.content)} relevant Reddit posts:")
            
            for i, post in enumerate(st.session_state.content):
                with st.expander(f"📝 {post['title']}", expanded=i==0):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Description:** {post['description']}")
                        st.write(f"**Link:** [{post['link']}]({post['link']})")
                    
                    with col2:
                        st.metric("Comments", len(post['comments']))
                    
                    if post['comments']:
                        st.write("**Top Comments:**")
                        for j, comment in enumerate(post['comments'][:3]):  # Show top 3 comments
                            with st.container():
                                st.write(f"👤 **{comment['author']}** (Score: {comment['score']})")
                                st.write(f"💬 {comment['body']}")
                                if j < len(post['comments'][:3]) - 1:
                                    st.divider()
        
        with tab1:
            # Chat interface
            # Display chat messages
            for message in st.session_state.messages:
                if isinstance(message, AIMessage):
                    with st.chat_message("assistant"):
                        st.markdown(message.content)
                elif isinstance(message, HumanMessage):
                    with st.chat_message("user"):
                        st.markdown(message.content)
    else:
        # Chat interface (when no content is available)
        # Display chat messages
        for message in st.session_state.messages:
            if isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.markdown(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(message.content)

    # Chat input
    if prompt := st.chat_input("What are you looking for today?"):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_content, updated_content = generate_response(
                    st.session_state.bot_state, 
                    prompt
                )
                
                # Update session state
                st.session_state.messages = st.session_state.bot_state["messages"]
                st.session_state.content = updated_content
                
                st.markdown(response_content)
        
        # Force rerun to update the chat
        st.rerun()

if __name__ == "__main__":
    main()
