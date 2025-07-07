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

class BotState(TypedDict):
    """Shared state flowing through the app graph."""
    messages: Annotated[list, add_messages]
    content: dict

# Product content retrieval tool
@tool
def get_content(query: str) -> List[dict]:
    """Fetches structured product data based on user query from Reddit discussions."""
    
    query = "site:reddit.com " + query
    
    try:
        api_key = os.getenv("SERP_API_KEY")
        if not api_key:
            raise ValueError("SERP_API_KEY environment variable is not set")
        
        results = get_search_results(query, api_key=api_key)
        
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
    
    # Get LLM response with graceful failure handling
    try:
        response = llm_with_tools.invoke(messages)
    except Exception as exc:
        # Log and inform the user without crashing the Streamlit app
        err_msg = f"⚠️ Sorry, I ran into an error while thinking: {exc}"
        state["messages"].append(AIMessage(content=err_msg))
        return err_msg, state["content"]
    
    # Handle tool calls
    tool_calls = getattr(response, "tool_calls", [])
    if tool_calls:
        # Add the AI message with tool calls
        state["messages"].append(response)
        
        # Process each tool call
        for tool_call in tool_calls:
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
        response_content_raw = final_response.content
        response_content = response_content_raw if isinstance(response_content_raw, str) else json.dumps(response_content_raw)
        # Attach TL;DR summary to response
        tldr_text = generate_tldr(response_content)
        if tldr_text:
            response_content = response_content + "\n\n**TL;DR:** " + tldr_text
        
        # Add final AI response to state
        state["messages"].append(AIMessage(content=response_content))
        
    else:
        # No tool calls, just add the response
        response_content_raw = response.content
        response_content = response_content_raw if isinstance(response_content_raw, str) else json.dumps(response_content_raw)
        # Attach TL;DR summary to response
        tldr_text = generate_tldr(response_content)
        if tldr_text:
            response_content = response_content + "\n\n**TL;DR:** " + tldr_text
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
    # Ensure preference store exists
    if "user_preferences" not in st.session_state:
        st.session_state.user_preferences = {}

# ------------------------------
# Preference graph renderer
# ------------------------------

def render_preference_graph() -> None:
    """Display the user preference graph as a Graphviz chart."""
    prefs: dict[str, int] = st.session_state.get("user_preferences", {})
    if not prefs:
        st.info("No preferences detected yet – chat to build the graph.")
        return

    dot = [
        "digraph Preferences {",
        "  rankdir=LR;",
        "  User [shape=ellipse, style=filled, color=lightblue];",
    ]
    for k, w in sorted(prefs.items(), key=lambda x: -x[1])[:25]:
        safe = k.replace("\"", "\\\"")
        dot.append(f'  "{safe}" [shape=box, style=filled, color=lightyellow];')
        dot.append(f'  User -> "{safe}" [label="{w}"];')
    dot.append("}")
    st.graphviz_chart("\n".join(dot))

# ------------------------------
# TL;DR generator
# ------------------------------

def generate_tldr(text: str) -> str:
    """Generate a concise (1-2 sentence) TL;DR for a given text using the LLM."""
    if not text or not isinstance(text, str):
        return ""

    llm = get_llm()
    try:
        summary_prompt = (
            "You are an assistant that summarises a response in 1-2 short sentences as a TL;DR. "
            "Return ONLY the TL;DR without any extra headings or formatting."
        )
        resp = llm.invoke([
            SystemMessage(content=summary_prompt),
            HumanMessage(content=text),
        ])
        tl = resp.content if isinstance(resp.content, str) else str(resp.content)
        return tl.strip()
    except Exception:
        # If summarisation fails, silently skip TL;DR
        return ""

def render_with_tldr(text: str) -> None:
    """Render assistant message with TL;DR inside an expander if present."""
    if not text:
        return
    if "**TL;DR:**" in text:
        main_text, tldr_part = text.split("**TL;DR:**", 1)
        st.markdown(main_text.strip())
        with st.expander("TL;DR"):
            st.markdown(tldr_part.strip())
    else:
        st.markdown(text)

def inject_custom_css() -> None:
    """Inject CSS to make tab headers sticky and tab panels scrollable."""
    st.markdown(
        """
        <style>
        /* Make tab headers sticky */
        div[data-baseweb="tabs"] > div:first-child {
            position: sticky;
            top: 0;
            background: var(--background-color, #ffffff);
            z-index: 998;
        }
        /* Scrollable tab panels (chat, posts, prefs) */
        div[data-baseweb="tab-panel"] {
            max-height: 80vh;
            overflow-y: auto;
        }

        /* Scrollable chat area */
        .chat-scroll {
            max-height: 70vh;
            overflow-y: auto;
            padding-right: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def main():
    st.set_page_config(
        page_title="Suvidha - Shopping Assistant",
        page_icon=":material/support_agent:",
        layout="wide"
    )
    
    # Inject custom CSS for layout tweaks
    inject_custom_css()
    
    st.title(":material/support_agent: Suvidha - Shopping Assistant")
    st.markdown("Get personalized product recommendations based on Reddit discussions!")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for content information
    with st.sidebar:
        if st.session_state.content:
            st.subheader(":material/search: Current Context")
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
        if st.button(":material/delete: Clear Chat"):
            st.session_state.messages = [AIMessage(content=WELCOME_MSG)]
            st.session_state.content = {}
            st.session_state.bot_state = {
                "messages": st.session_state.messages,
                "content": st.session_state.content
            }
            st.rerun()

    # Always show tabs for better organization
    tab1, tab2, tab3 = st.tabs([":material/chat: Chat", ":material/newspaper: Reddit Posts", ":material/neurology: Preferences"])
    
    with tab1:
        # Render chat history inside scrollable area (about 70% of viewport)
        with st.container(height=0):  # placeholder to attach CSS class
            st.markdown(
                """
                <div class="chat-scroll">
                """,
                unsafe_allow_html=True,
            )
            for msg in st.session_state.messages:
                if isinstance(msg, AIMessage):
                    if msg.content and str(msg.content).strip():
                        with st.chat_message("assistant"):
                            render_with_tldr(str(msg.content))
                elif isinstance(msg, HumanMessage):
                    with st.chat_message("user"):
                        st.markdown(msg.content)
            st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        if st.session_state.content:
            valid_posts = [item for item in st.session_state.content if isinstance(item, dict) and 'title' in item]
            st.write(f"Found {len(valid_posts)} relevant Reddit posts:")
            
            # Wrap posts list in a scrollable container – mirrors chat scrolling behaviour
            with st.container(height=600):
                for i, post in enumerate(valid_posts):
                    with st.expander(f":material/article: {post.get('title', 'Unknown Title')}", expanded=i==0):
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
                                    st.write(f":material/person: **{comment.get('author', 'Unknown')}** (Score: {comment.get('score', 0)})")
                                    st.write(f":material/chat: {comment.get('body', 'No content')}")
                                    if j < len(post['comments'][:3]) - 1:
                                        st.divider()
        else:
            st.info(":material/search: **No Reddit posts found yet**")
            st.markdown("Ask questions about products and I'll search Reddit for relevant discussions and reviews!")
            st.markdown("**Try asking about:**")
            st.markdown("- 'Best wireless headphones under $200'")
            st.markdown("- 'Sony camera vs Canon for beginners'")
            st.markdown("- 'Gaming laptop recommendations'")
    
    with tab3:
        render_preference_graph()

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
                
                # Update preference graph with the latest user query
                update_user_preferences(prompt)
                
                render_with_tldr(response_content)
        
        # Force rerun to update the chat
        st.rerun()

def update_user_preferences(user_query: str) -> None:
    """Extract preferences from the latest user query via the LLM and merge into the graph."""
    if not user_query or not isinstance(user_query, str):
        return

    system_prompt = (
        "You are an assistant that extracts a shopper's preference keywords from ONE message. "
        "Return ONLY a JSON object mapping concise lowercase keywords (1-3 words) to an integer weight 1-5. "
        "Example: {\"mirrorless camera\": 5, \"sony\": 4}. No extra text."
    )

    llm = get_llm()
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query),
        ])
        txt_resp = response.content if isinstance(response.content, str) else str(response.content)
        txt_resp = txt_resp.strip()
        import re, json
        if not txt_resp.startswith("{"):
            match = re.search(r"\{[\s\S]*\}", txt_resp)
            txt_resp = match.group(0) if match else "{}"
        prefs_fragment: dict[str, int] = json.loads(txt_resp)

        # Merge into existing store
        prefs_store: dict[str, int] = st.session_state.get("user_preferences", {})
        for k, v in prefs_fragment.items():
            try:
                weight = int(v)
            except Exception:
                continue
            prefs_store[k] = max(prefs_store.get(k, 0), weight)
        st.session_state.user_preferences = prefs_store
    except Exception as exc:
        st.warning(f"Preference extraction failed: {exc}")

if __name__ == "__main__":
    main()