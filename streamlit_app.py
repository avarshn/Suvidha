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
from shopping_cache import get_shopping_results
from streamlit_product_card import product_card
import streamlit.components.v1 as components
from streamlit_mic_recorder import mic_recorder
from streamlit_agraph import agraph, Node, Edge, Config
from groq import Groq
import io
import logging
# Load environment variables
load_dotenv()

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


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
    products: list

# Product content retrieval tool
@tool
def get_content(query: str) -> List[dict]:
    """Fetch structured product data from Reddit discussions and Google Shopping.

    The result is a unified list of dictionaries so downstream components (chat UI, summariser)
    can treat every item uniformly. For Google Shopping items we include an empty ``comments``
    list so existing code that iterates over that key continues to work without changes.
    """

    user_query = query.strip()

    # Build separate query strings for Reddit (dorked) and Shopping
    reddit_query = f"site:reddit.com {user_query}"

    try:
        api_key = os.getenv("SEARCH_API_KEY")
        if not api_key:
            raise ValueError("SEARCH_API_KEY environment variable is not set")

        # -----------------------------
        # 1) Reddit Posts via Google
        # -----------------------------
        reddit_serp = get_search_results(reddit_query, api_key=api_key)

        logging.info(f"Reddit SERP: {reddit_serp}")
        # Transform into structured objects
        search_response = SearchAPIResponse.from_json(reddit_serp)
        reddit_results = search_response.reddit_results[:5]


        logging.info(f"Reddit Results: {reddit_results}")

        product_data: List[dict] = []

        for reddit_result in reddit_results:
            # Fetch post metadata and top-level comments
            try:

                logging.info(f"each reddit_result: {reddit_result}")
                post = fetch_reddit_post(reddit_result.link)
                product_data.append({
                    "title": post.title,
                    "description": post.description[:200] + ("..." if len(post.description) > 200 else ""),
                    "link": post.link,
                    "comments": [
                        {
                            "id": comment.id,
                            "author": comment.author,
                            "body": comment.body[:150] + ("..." if len(comment.body) > 150 else ""),
                            "score": comment.score,
                        }
                        for comment in post.comments[:5]  # Limit to first 5 comments
                    ],
                    "source": "reddit",
                })
            except Exception:

                logging.info(f"error in getting attributes from reddit posts")
                # Skip individual post failures without aborting entire run
                continue
        logging.info(f"product_data: {product_data}")
        
        # Return ONLY Reddit-based product discussion data
        return product_data
    except Exception as e:
        return [{"error": f"Failed to fetch content: {str(e)}"}]

# -------------------------------------------------
# New tool: Fetch product offers from Google Shopping
# -------------------------------------------------

def get_products_from_gshopping(query: str, max_results: int = 1) -> List[dict]:
    """Fetch product offers from Google Shopping SearchAPI.

    Returns a list of product dictionaries compatible with the UI card renderer.
    max_results: how many offers to return (default 1)
    """
    user_query = query.strip()
    api_key = os.getenv("SERP_API_KEY")
    if not api_key:
        return [{"error": "SERP_API_KEY environment variable is not set"}]
    try:
        print(f"Fetching products for query: {user_query}\t results: {max_results}")
        shopping_raw = get_shopping_results(user_query, api_key, engine="google_shopping")
        product_list: List[dict] = []
        for item in shopping_raw.get("shopping_results", [])[:max_results]:
            title_val = item.get("title") or item.get("name") or item.get("product_title")
            product_list.append({
                "title": title_val or "Unknown Product",
                "description": item.get("description") or item.get("source", ""),
                "link": item.get("product_link", "#"),
                "price": item.get("price"),
                "rating": item.get("rating"),
                "product_image": item.get("thumbnail") or item.get("image"),
            })
        return product_list
    except Exception as exc:
        return [{"error": f"Failed to fetch products: {str(exc)}"}]

# Node: Generate assistant response
def generate_response(state: BotState, user_input: str) -> tuple[str, dict, str]:
    """Generate assistant response using LLM with tool calling capability"""
    
    # Add user message to state
    state["messages"].append(HumanMessage(content=user_input))
    
    # Bind tools to LLM (Reddit + Shopping)
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
        err_msg = f":material/error: Sorry, I ran into an error while thinking: {exc}"
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
                with st.spinner(f"Fetching Reddit comments for: {tool_call['args']['query']}"):
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

    # Add final AI response to state with TL;DR metadata
    ai_message = AIMessage(content=response_content)
    if tldr_text:
        ai_message.additional_kwargs = {"tldr": tldr_text}
    state["messages"].append(ai_message)
    
    # -----------------------------
    # Auto-fetch product offers mentioned in the reply
    # -----------------------------
    products = get_product_entities(response_content)

    products = [p.get("product_name") for p in products]

    print(products)

    if products:

        st.session_state.products = []
        
        # If only one product, fetch up to 5; else, fetch only 1 per product
        max_results = 5 if len(products) == 1 else 1
        for q in products:

            # with st.spinner(f"🛒 Searching offers for '{q}'"):
            prod_results = get_products_from_gshopping(q, max_results)
            if isinstance(prod_results, list):
                st.session_state.products.extend(prod_results)
                state["messages"].append(ToolMessage(
                    content=json.dumps(prod_results),
                    tool_call_id=f"auto_products_{q}"
                ))
        if state.get("products"):
            st.toast("Product offers updated!", icon=":material/local_mall:")
        

    return response_content, state["content"], tldr_text

# Initialize session state
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [AIMessage(content=WELCOME_MSG)]
    if "content" not in st.session_state:
        st.session_state.content = {}
    if "bot_state" not in st.session_state:
        st.session_state.bot_state = {
            "messages": st.session_state.messages,
            "content": st.session_state.content,
            "products": [],
        }
    # Ensure preference store exists
    if "user_preferences" not in st.session_state:
        st.session_state.user_preferences = {}
    # Ensure a top-level products cache for UI convenience
    if "products" not in st.session_state:
        st.session_state.products = []
    # Add speech processing state
    if "processing_speech" not in st.session_state:
        st.session_state.processing_speech = False
    if "is_recording" not in st.session_state:
        st.session_state.is_recording = False
    if "groq_client" not in st.session_state:
        if os.getenv("GROQ_API_KEY"):
            st.session_state.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        else:
            st.session_state.groq_client = None

# ------------------------------
# Preference graph renderer
# ------------------------------

def render_preference_graph() -> None:
    """Display the user preference graph as an interactive agraph chart."""
    prefs: dict[str, int] = st.session_state.get("user_preferences", {})
    if not prefs:
        st.info("No preferences detected yet – chat to build the graph.")
        return

    nodes = []
    edges = []
    
    # Create the central User node positioned at the center
    nodes.append(Node(id="User", 
                     label="User", 
                     size=30, 
                     shape="dot",
                     color="#87CEEB",  # Light blue
                     x=300,  # Center position
                     y=300))
    
    # Create nodes for each preference and edges from User to preferences
    import math
    preferences = sorted(prefs.items(), key=lambda x: -x[1])[:25]  # Top 25 preferences
    num_prefs = len(preferences)
    
    for i, (k, w) in enumerate(preferences):
        # Calculate position in a circle around the center
        angle = (2 * math.pi * i) / num_prefs
        radius = 50  # Distance from center
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        
        # Preference node
        nodes.append(Node(id=k,
                         label=k,
                         size=15 + (w * 3),  # Size based on weight
                         shape="box",
                         color="#FFFFE0",  # Light yellow
                         ))
        
        # Edge from User to preference with weight as label
        edges.append(Edge(source="User",
                         target=k,
                         label=str(w),
                         color="#808080"))  # Gray

    # Center the graph with moderate height
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        config = Config(width=750,
                        height=500,
                        directed=True, 
                        physics=True, 
                        hierarchical=False,
                        nodeHighlightBehavior=False,
                        highlightColor="#F7A7A6",
                        collapsible=False,
                        initialZoom=1.5,
                        fit=True,
                        dragNodes=False,
                        dragView=False,
                        zoomView=False,
                        selectConnectedEdges=False)

        # Render the graph
        return_value = agraph(nodes=nodes, 
                              edges=edges, 
                              config=config)

# ------------------------------
# TL;DR generator
# ------------------------------

def generate_tldr(response_content: str) -> str:
    """Generate a 2-line TLDR summary of the main response"""
    try:
        llm = get_llm()
        
        tldr_prompt = f"""Please summarize the following shopping assistant response in exactly 2 lines.
Make it concise and capture the key recommendations or insights. The main goal is to add TTS support for this summary, because the main response is too long for TTS to handle effectively.
So you shouldn't loose any important details, but make sure the summary is short enough to be read aloud in a reasonable time. And keep it as friendly and engaging as possible.

If the response is already concise, just return it as is.
Response to summarize:
{response_content}

TLDR (2 lines):"""
        
        tldr_response = llm.invoke([HumanMessage(content=tldr_prompt)])
        tldr_content = tldr_response.content.strip()
        
        # Ensure it's exactly 2 lines
        lines = tldr_content.split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        
        if len(lines) >= 2:
            return f"{lines[0]}\n{lines[1]}"
        elif len(lines) == 1:
            # If only one line, split it or add a second line
            if len(lines[0]) > 80:
                mid_point = lines[0].rfind(' ', 0, 80)
                if mid_point > 0:
                    return f"{lines[0][:mid_point]}\n{lines[0][mid_point+1:]}"
            return f"{lines[0]}\nBased on Reddit discussions and user experiences."
        else:
            return "Key insights from Reddit discussions.\nRecommendations based on user experiences."
    
    except Exception as e:
        # Return a fallback TLDR if API fails
        return "Product recommendations summary.\nBased on Reddit user discussions."

def render_with_tldr(text: str, tldr_text: str = None) -> None:
    """Render assistant message with TL;DR inside an expander if present."""
    if not text:
        return
    
    # Render main text
    st.markdown(text.strip())
    
    # Render TL;DR if provided
    if tldr_text and tldr_text.strip():
        with st.expander("TL;DR"):
            st.markdown(tldr_text.strip())

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

            # Show detected products list
            if st.session_state.products:
                st.subheader(":material/local_mall: Products Detected")
                for p in st.session_state.products:
                    st.markdown(f"- [{p.get('title','Unknown')}]({p.get('link','#')})")
        
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
    tab_chat, tab_products, tab_posts, tab_prefs = st.tabs([
        ":material/chat: Chat", 
        ":material/local_mall: Products", 
        ":material/newspaper: Reddit Posts", 
        ":material/neurology: Preferences"
    ])
    
    with tab_chat:
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
                            # Get TL;DR from message metadata if available
                            tldr_text = msg.additional_kwargs.get("tldr") if hasattr(msg, 'additional_kwargs') and msg.additional_kwargs else None
                            render_with_tldr(str(msg.content), tldr_text)
                elif isinstance(msg, HumanMessage):
                    with st.chat_message("user"):
                        st.markdown(msg.content)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Speech input processing (only in chat tab)
        audio = mic_recorder(
            start_prompt="🎤",
            stop_prompt="⏹️",
            just_once=True,
            use_container_width=False,
            format="webm",
            key='speech_recorder'
        )
        
        # Process speech input if available
        if audio and not st.session_state.processing_speech:
            transcribed_text = process_speech_input(audio)
            if transcribed_text:
                # Use transcribed text as prompt
                prompt = transcribed_text
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(f"{prompt}")
                
                # Generate and display assistant response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response_content, updated_content, tldr_text = generate_response(
                            st.session_state.bot_state, 
                            prompt
                        )
                        
                        # Update session state
                        st.session_state.messages = st.session_state.bot_state["messages"]
                        st.session_state.content = updated_content
                        
                        # Update preference graph with the latest user query
                        update_user_preferences(prompt)
                        
                        render_with_tldr(response_content, tldr_text)
                
                # Force rerun to update the chat
                st.rerun()

        # Chat input (only in chat tab)
        if prompt := st.chat_input("What are you looking for today?"):
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response_content, updated_content, tldr_text = generate_response(
                        st.session_state.bot_state, 
                        prompt
                    )
                    
                    # Update session state
                    st.session_state.messages = st.session_state.bot_state["messages"]
                    st.session_state.content = updated_content
                    
                    # Update preference graph with the latest user query
                    update_user_preferences(prompt)
                    
                    render_with_tldr(response_content, tldr_text)
            
            # Force rerun to update the chat
            st.rerun()
    
    with tab_products:
        products = st.session_state.products
        if products:
            st.subheader(":material/local_mall: Products")
            rows = [products[i:i+3] for i in range(0, len(products), 3)]
            for i, row in enumerate(rows):
                cols = st.columns(len(row))
                for idx, prod in enumerate(row):
                    with cols[idx]:
                        product_card(
                            product_name=prod.get("title", "Unknown Product"),
                            description=prod.get("description", ""),
                            product_image=prod.get("product_image"),
                            price=prod.get("price"),
                            button_text="",
                            on_button_click=lambda url=prod.get("link", "#"): open_url(url),
                            key=f"card_{hash(prod.get('link', '') + str(idx) + str(i))}"
                        )
        else:
            st.info("No product offers fetched yet. Ask about products to see offers here.")
    
    with tab_posts:
        if st.session_state.content:
            # Separate reddit posts and shopping items
            reddit_posts = [item for item in st.session_state.content if item.get("source") == "reddit"]

            if reddit_posts:
                st.subheader(":material/newspaper: Reddit Posts")
                st.write(f"Found {len(reddit_posts)} relevant Reddit posts:")
                with st.container(height=600):
                    for i, post in enumerate(reddit_posts):
                        with st.expander(f":material/article: {post.get('title', 'Unknown Title')}", expanded=False):
                            col1, col2 = st.columns([3, 1])

                            with col1:
                                st.write(f"**Description:** {post.get('description', 'No description available')}")
                                st.write(f"**Link:** [{post.get('link', '#')}]({post.get('link', '#')})")

                            with col2:
                                st.metric("Comments", len(post.get('comments', [])))

                            if post.get('comments'):
                                st.write("**Top Comments:**")
                                for j, comment in enumerate(post['comments'][:3]):
                                    with st.container():
                                        st.write(f":material/person: **{comment.get('author', 'Unknown')}** (Score: {comment.get('score', 0)})")
                                        st.write(f":material/chat: {comment.get('body', 'No content')}")
                                        if j < len(post['comments'][:3]) - 1:
                                            st.divider()
        else:
            st.info(":material/search: **No Reddit posts found yet**")
            st.markdown("Ask questions about products and I'll search Reddit for relevant discussions and reviews!")
    
    with tab_prefs:
        render_preference_graph()




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

# -------------------------------------------------
# Helper: Extract product keywords from assistant reply
# -------------------------------------------------

def get_product_entities(response_text: str) -> List[dict]:
    """Extract product entities from the LLM response."""
    try:

        
        # Skip if response is empty or too short
        if not response_text or len(response_text.strip()) < 10:
            return []
        
        # Skip error messages
        if any(error_indicator in response_text for error_indicator in ["🚫", "⚠️", "⏰", "⏱️", "Error"]):
            return []
        
        system_prompt = """
        You are an assistant that extracts product entities from shopping assistant responses.
        Extract specific product names, brands, and categories mentioned in the text.
        Return ONLY a JSON array of objects with this format:
        [
            {
                "product_name": "Sony WH-1000XM4",
                "brand": "Sony",
            }
        ]
        
        Only extract actual products that are specifically mentioned or recommended.
        Do not extract generic terms or vague references.
        If no specific products are found, return an empty array: []
        """
        
        llm = get_llm()
        extraction_response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Extract product entities from this text:\n\n{response_text}")
        ])
        
        # Parse the JSON response
        txt_resp = extraction_response.content if isinstance(extraction_response.content, str) else str(extraction_response.content)
        txt_resp = txt_resp.strip()

        print(f"Extraction response: {extraction_response.content}")
        
        # Handle JSON extraction
        import re, json
        if not txt_resp.startswith("["):
            match = re.search(r"\[[\s\S]*\]", txt_resp)
            txt_resp = match.group(0) if match else "[]"
        
        product_entities = json.loads(txt_resp)
        
        # Validate the structure
        if not isinstance(product_entities, list):
            return []
        
        # Filter and validate each entity
        valid_entities = []
        for entity in product_entities:
            if (isinstance(entity, dict) and 
                entity.get("product_name") and 
                entity.get("brand")):
                valid_entities.append(entity)
        
        print(f"Valid entities: {valid_entities}")
        return valid_entities
        
    except Exception as e:
        # Silently fail and return empty list
        return []

def speech_to_text(audio_bytes):
    """Convert speech to text using Groq Speech API."""
    if not st.session_state.groq_client:
        st.error("Groq API not available. Please set GROQ_API_KEY environment variable.")
        return None
    
    try:
        # Create a BytesIO object from audio bytes
        audio_bio = io.BytesIO(audio_bytes)
        audio_bio.name = 'audio.webm'  # Set filename for Groq API
        
        # Use Groq's speech-to-text API
        response = st.session_state.groq_client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=audio_bio,
            response_format="verbose_json"
        )
        return response.text
    except Exception as e:
        st.error(f"Error in speech-to-text: {e}")
        return None

def process_speech_input(audio_data):
    """Process recorded audio and convert to text input."""
    if not audio_data:
        return None
    
    st.session_state.processing_speech = True
    
    with st.spinner("🔄 Converting speech to text..."):
        # Convert speech to text
        user_text = speech_to_text(audio_data['bytes'])
        
        if user_text:
            st.session_state.processing_speech = False
            return user_text
        else:
            st.error("Could not transcribe your speech. Please try again.")
            st.session_state.processing_speech = False
            return None

def open_url(url: str):
    """Open a URL in a new browser tab from Streamlit callback."""
    js = f"window.open('{url}', '_blank')"
    components.html(f"<script>{js}</script>", height=0)

if __name__ == "__main__":
    main()
