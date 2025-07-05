import streamlit as st
from typing import Annotated, TypedDict, Dict, List
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.tools import tool
import json
import os
from dotenv import load_dotenv
from search_cache import get_search_results  # Local caching
from main import SearchAPIResponse, RedditResult, fetch_reddit_post

# Load environment variables
load_dotenv()


# Initialize LLM with tools bound
@st.cache_resource
def get_llm():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1
    )
    return llm

# State definition
class BotState(TypedDict):
    messages: Annotated[list, add_messages]
    content: dict  # Holds product content in JSON format

# System instruction
SYS_INSTRUCTION = """You are a shopping assistant helping users find products. 

Use the provided product content and help the user in making informed decisions. 
Maybe ask clarifying questions to understand their needs better.
You can give information about products and all from the content provided, if user asks.

You have access to a tool called 'get_content' that can search Reddit for product information and reviews.
(Do not mention this tool to the user, just use it when needed.)
Use this tool when:
- Users ask about specific products or brands
- Users want recommendations for a category of products
- Users need reviews or opinions about products
- Users ask for comparisons between products

DO NOT use the tool when:
- Users are just greeting or having general conversation
- Users ask about your capabilities or general questions
- Users are asking follow-up questions about already retrieved content

When you use the tool, make sure to provide helpful analysis and recommendations based on the Reddit discussions found.
If you already have product content available, use it to answer questions without calling the tool again unless the user asks for different products.

Be helpful, informative, and focus on helping users make informed purchasing decisions."""

WELCOME_MSG = "Welcome to Suvidha! How can I assist you with your shopping needs today?"

# Product content retrieval tool
@tool
def get_content(query: str) -> List[dict]:
    """Fetches structured product data based on user query from Reddit discussions."""
    
    query = "site:reddit.com " + query
    
    try:
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
                continue
        
        return product_data
    except Exception as e:
        return [{"error": f"Failed to fetch content: {str(e)}"}]

# Node: Generate assistant response
def generate_response(state: BotState, user_input: str) -> tuple[str, dict]:
    """Generate assistant response using LLM with tool calling capability"""
    
    # Add user message to state
    state["messages"].append(HumanMessage(content=user_input))
    
    # Bind tools to LLM
    llm_with_tools = get_llm().bind_tools([get_content])
    
    # Create system message
    sys_msg = SystemMessage(content=SYS_INSTRUCTION)
    
    # Prepare messages for LLM
    messages = [sys_msg] + state["messages"]
    
    # Get LLM response
    response = llm_with_tools.invoke(messages)
    
    # Handle tool calls
    if response.tool_calls:
        # Add the AI message with tool calls
        state["messages"].append(response)
        
        # Process each tool call
        for tool_call in response.tool_calls:
            if tool_call["name"] == "get_content":
                # Show spinner while fetching content
                with st.spinner(f"Fetching product information for: {tool_call['args']['query']}"):
                    # Execute the tool
                    tool_result = get_content.invoke(tool_call["args"])
                    
                    # Update state content
                    state["content"] = tool_result
                    
                    # Add tool message to state
                    tool_message = ToolMessage(
                        content=json.dumps(tool_result),
                        tool_call_id=tool_call["id"]
                    )
                    state["messages"].append(tool_message)
                    
                    st.success(f"Fetched {len(tool_result)} Reddit posts for analysis")
        
        # Get final response after tool execution
        final_response = llm_with_tools.invoke([sys_msg] + state["messages"])
        response_content = final_response.content
        
        # Add final AI response to state
        state["messages"].append(AIMessage(content=response_content))
        
    else:
        # No tool calls, just add the response
        response_content = response.content
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
            total_comments = sum(len(item.get('comments', [])) for item in st.session_state.content if isinstance(item, dict) and 'comments' in item)
            st.write(f"**Total Comments:** {total_comments}")
            
            st.write("**Post Previews:**")
            valid_posts = [item for item in st.session_state.content if isinstance(item, dict) and 'title' in item]
            for i, item in enumerate(valid_posts[:3]):  # Show first 3 items
                with st.expander(f"Post {i+1}: {item['title'][:40]}..."):
                    st.write(f"**Description:** {item.get('description', 'No description')[:100]}...")
                    st.write(f"**Comments:** {len(item.get('comments', []))}")
                    st.write(f"**Link:** [View on Reddit]({item.get('link', '#')})")
                    
                    if item.get('comments'):
                        st.write("**Top Comment:**")
                        top_comment = max(item['comments'], key=lambda x: x.get('score', 0))
                        st.write(f"👤 {top_comment.get('author', 'Unknown')} (Score: {top_comment.get('score', 0)})")
                        st.write(f"💬 {top_comment.get('body', 'No content')[:80]}...")
            
            if len(valid_posts) > 3:
                st.write(f"... and {len(valid_posts) - 3} more posts")
        
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
            valid_posts = [item for item in st.session_state.content if isinstance(item, dict) and 'title' in item]
            st.write(f"Found {len(valid_posts)} relevant Reddit posts:")
            
            for i, post in enumerate(valid_posts):
                with st.expander(f"📝 {post.get('title', 'Unknown Title')}", expanded=i==0):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Description:** {post.get('description', 'No description available')}")
                        st.write(f"**Link:** [{post.get('link', '#')}]({post.get('link', '#')})")
                    
                    with col2:
                        st.metric("Comments", len(post.get('comments', [])))
                    
                    if post.get('comments'):
                        st.write("**Top Comments:**")
                        for j, comment in enumerate(post['comments'][:3]):  # Show top 3 comments
                            with st.container():
                                st.write(f"👤 **{comment.get('author', 'Unknown')}** (Score: {comment.get('score', 0)})")
                                st.write(f"💬 {comment.get('body', 'No content')}")
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