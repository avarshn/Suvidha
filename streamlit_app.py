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
from search_cache import get_search_results, get_shopping_results # Local caching
from main import SearchAPIResponse, RedditResult, fetch_reddit_post
from groq import Groq
import base64
from io import BytesIO
import re, json

# Load environment variables
load_dotenv()


# Initialize LLM with tools bound
@st.cache_resource
def get_llm():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",  #"qwen/qwen3-32b", # ,
        temperature=0.1,
        max_retries=3,  # Add retry logic
        timeout=30  # Add timeout
    )
    return llm

# Initialize LLM with tools bound
@st.cache_resource
def get_budget_llm():
    llm = ChatGroq(
        model="llama-3.1-8b-instant",  #"qwen/qwen3-32b", # ,
        temperature=0.1,
        max_retries=3,  # Add retry logic
        timeout=30  # Add timeout
    )
    return llm

# System instruction
def get_system_instruction(user_preferences: dict) -> str:
    """Generate system instruction with user preferences included."""
    base_instruction = """You are a helpful shopping assistant focused on providing clear, concise product recommendations.

CORE GUIDELINES:
- Give direct, user-friendly answers without technical jargon
- Focus on practical benefits and real-world usage
- Never include debug information, system messages, or technical processes in your response
- Keep responses conversational and helpful

TOOL USAGE:
- Use the 'get_content' tool only when you need specific product information
- Don't use it for vague queries - ask for clarification first
- Don't mention the tool or technical processes to the user

- Use the user's known preferences to ask more targeted questions.

RESPONSE FORMAT:
- Start with direct recommendations
- Use clear headings for different products
- Always show each recommendation in a new line, separarted by "\n".
- Include Reddit citations as inline links: [[1]](https://reddit.com/link)
- Never add a "Sources:" section
- Keep descriptions concise but informative

WHAT TO AVOID:
- Debug messages or system information
- Technical processes or tool mentions
- Overly verbose explanations
- Redirecting users to external research
- Try not to tell user to go to internet or other sources, its your job to find the best products for them based on Reddit discussions and reviews.

Example good response:
"**Sony WH-1000XM4** - Excellent noise cancellation and 30-hour battery life make these perfect for travel [1](https://reddit.com/r/headphones/...)

**Bose QuietComfort 45** - Superior comfort for long listening sessions with great sound quality [2](https://reddit.com/r/bose/...)"
"""

    # Add user preferences to the context if they exist
    if user_preferences:
        pref_text = "\n\nUSER PREFERENCES (use these to personalize recommendations):\n"
        sorted_prefs = sorted(user_preferences.items(), key=lambda x: x[1], reverse=True)
        for pref, weight in sorted_prefs:
            pref_text += f"- {pref}: {weight}/5 importance\n"
        pref_text += "\nUse these preferences to provide more targeted recommendations and ask fewer clarifying questions."
        base_instruction += pref_text
    else:
        base_instruction += "\n\nNOTE: This user has no known preferences yet, so you may need to ask more clarifying questions to understand their needs."

    return base_instruction

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
                
                # Extract subreddit from the link for better citations
                subreddit = "unknown"
                if "/r/" in post.link:
                    try:
                        subreddit = post.link.split("/r/")[1].split("/")[0]
                    except:
                        pass
                
                product_data.append({
                    "title": post.title,
                    "description": post.description[:200] + ('...' if len(post.description) > 200 else ''),
                    "link": post.link,
                    "subreddit": subreddit,
                    "source_citation": f"{post.title} - r/{subreddit} (Reddit discussion)",
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



def play_audio_hidden(audio_bytes, audio_id="current_audio"):
    """Play audio automatically without showing controls."""
    if audio_bytes:
        # Handle BytesIO objects by extracting the bytes
        if hasattr(audio_bytes, 'getvalue'):
            audio_data = audio_bytes.getvalue()
        else:
            audio_data = audio_bytes
            
        # Convert audio bytes to base64
        audio_base64 = base64.b64encode(audio_data).decode()
        
        # Create HTML5 audio element with autoplay and controls for stopping
        audio_html = f"""
        <audio id="{audio_id}" autoplay style="display: none;" 
               onended="this.setAttribute('data-ended', 'true'); window.audioEnded('{audio_id}');"
               onloadstart="this.setAttribute('data-loading', 'true');"
               oncanplay="this.setAttribute('data-ready', 'true');">
            <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
            Your browser does not support the audio element.
        </audio>
        <script>
            // Aggressively stop ALL audio elements first
            document.querySelectorAll('audio').forEach(function(audio) {{
                audio.pause();
                audio.currentTime = 0;
                audio.muted = true;  // Mute first to prevent any sound
            }});
            
            // Wait a moment then unmute and play only the new audio
            setTimeout(() => {{
                const currentAudio = document.getElementById('{audio_id}');
                if (currentAudio) {{
                    currentAudio.muted = false;
                    currentAudio.play().catch(function(error) {{
                        // Audio play failed silently
                    }});
                }}
            }}, 50);
            
            // Function to stop specific audio
            function stopAudio(audioId) {{
                const audio = document.getElementById(audioId);
                if (audio) {{
                    audio.pause();
                    audio.currentTime = 0;
                    audio.muted = true;
                    audio.setAttribute('data-stopped', 'true');
                }}
            }}
            
            // Global stop function for current audio
            window.stopCurrentAudio = function(audioId) {{
                if (audioId) {{
                    stopAudio(audioId);
                }} else {{
                    // Stop all audio if no specific ID
                    document.querySelectorAll('audio').forEach(function(audio) {{
                        audio.pause();
                        audio.currentTime = 0;
                        audio.muted = true;
                    }});
                }}
            }};
            
            // Handle audio ending naturally
            window.audioEnded = function(audioId) {{
                // The ended state will be handled by Streamlit app logic
            }};
        </script>
        """
        
        # Display the HTML
        st.markdown(audio_html, unsafe_allow_html=True)
        
        # Optional: Show a subtle indicator
        st.markdown("🔊 *AI response is playing...*")

def stop_audio(audio_id=None):
    """Stop currently playing audio."""
    target_audio = audio_id or st.session_state.currently_playing_audio
    if target_audio:
        stop_html = f"""
        <script>
            if (typeof window.stopCurrentAudio !== 'undefined') {{
                window.stopCurrentAudio('{target_audio}');
            }}
        </script>
        """
        st.markdown(stop_html, unsafe_allow_html=True)
        
        # Clear the state if we stopped the currently playing audio
        if target_audio == st.session_state.currently_playing_audio:
            st.session_state.currently_playing_audio = None

def text_to_speech(text):
    """Convert text to speech using Groq Speech API."""
    try:
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = groq_client.audio.speech.create(
            model="playai-tts",
            voice="Arista-PlayAI",
            response_format="wav",
            input=text
        )
        
        # Return the audio bytes directly
        return BytesIO(response.read())
        
    except Exception as e:
        st.error(f"Error in text-to-speech: {e}")
        return None

def get_product_entities(response) -> List[dict]:
    """Extract product entities from the LLM response."""
    try:

        response_text = response.content if isinstance(response.content, str) else str(response.content)
        
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

def format_response_with_citations(response_content: str, sources: List[dict]) -> str:
    """Format response content to make inline links more visible."""
    
    if not sources:
        return response_content
    
    # The LLM should already include inline hyperlinks, so we just return the content
    # This function is kept for potential future enhancements
    return response_content

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
        if "headphone" in response_content.lower():
            return "Headphone recommendations found.\nBased on Reddit user discussions."
        elif "laptop" in response_content.lower():
            return "Laptop recommendations found.\nBased on Reddit user discussions."
        else:
            return "Product recommendations summary.\nBased on Reddit user discussions."
    
# Node: Generate assistant response
def generate_response(state: BotState, user_input: str) -> tuple[str, dict, str, BytesIO]:
    """Generate assistant response using LLM with tool calling capability"""
    
    # Add user message to state
    state["messages"].append(HumanMessage(content=user_input))
    
    # Bind tools to LLM
    llm_with_tools = get_llm().bind_tools([get_content])
    
    # Get user preferences from session state
    user_preferences = st.session_state.get("user_preferences", {})
    
    # Create system message with user preferences
    sys_msg = SystemMessage(content=get_system_instruction(user_preferences))
    
    # Prepare messages for LLM
    messages = [sys_msg] + state["messages"]
    
    # Get LLM response with graceful failure handling
    try:
        response = llm_with_tools.invoke(messages)
        # Extract product entities from the response
        product_list = get_product_entities(response)
        # Store product entities in session state for tracking
        if product_list:
            st.session_state.product_entities.extend(product_list)
            # Keep only unique entities (by product_name)
            seen = set()
            unique_entities = []
            for entity in st.session_state.product_entities:
                if entity.get("product_name") not in seen:
                    seen.add(entity.get("product_name"))
                    unique_entities.append(entity)
            st.session_state.product_entities = unique_entities
            
            # Fetch shopping results with images for the new products
            try:
                with st.spinner("🛒 Fetching current prices and product images..."):
                    shopping_results = get_shopping_results(product_list)
                    if shopping_results:
                        # Add to session state shopping results
                        st.session_state.shopping_results.extend(shopping_results)
                        # Keep only unique shopping results (by product_name)
                        seen_shopping = set()
                        unique_shopping = []
                        for result in st.session_state.shopping_results:
                            if result.get("product_name") not in seen_shopping:
                                seen_shopping.add(result.get("product_name"))
                                unique_shopping.append(result)
                        st.session_state.shopping_results = unique_shopping
                        st.success(f"✅ Found {len(shopping_results)} products with current pricing!")
            except Exception as e:
                print(f"Error fetching shopping results: {e}")
                st.warning("⚠️ Could not fetch current pricing data. Please check the Shopping tab later.")
    except Exception as exc:
        # Check if it's a Groq API issue
        error_msg = str(exc).lower()
        if "503" in error_msg or "service unavailable" in error_msg or "upstream" in error_msg:
            err_msg = "🚫 **Groq API is currently experiencing issues.** This is a temporary service outage. Please try again in a few minutes, or check [Groq's status page](https://status.groq.com/) for updates."
        elif "rate limit" in error_msg or "429" in error_msg:
            err_msg = "⏰ **Rate limit reached.** Please wait a moment before trying again."
        elif "timeout" in error_msg:
            err_msg = "⏱️ **Request timeout.** The API took too long to respond. Please try again."
        else:
            err_msg = f"⚠️ **API Error:** {exc}\n\nThis appears to be a temporary issue. Please try again in a moment."
        
        state["messages"].append(AIMessage(content=err_msg))
        return err_msg, state["content"], "", None
    
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
        try:
            final_response = llm_with_tools.invoke([sys_msg] + state["messages"])
            response_content = final_response.content
            # Extract product entities from the final response
            product_list = get_product_entities(final_response)
            if product_list:
                st.session_state.product_entities.extend(product_list)
                # Keep only unique entities (by product_name)
                seen = set()
                unique_entities = []
                for entity in st.session_state.product_entities:
                    if entity.get("product_name") not in seen:
                        seen.add(entity.get("product_name"))
                        unique_entities.append(entity)
                st.session_state.product_entities = unique_entities
                
                # Fetch shopping results with images for the new products
                try:
                    with st.spinner("🛒 Fetching current prices and product images..."):
                        shopping_results = get_shopping_results(product_list)
                        if shopping_results:
                            # Add to session state shopping results
                            st.session_state.shopping_results.extend(shopping_results)
                            # Keep only unique shopping results (by product_name)
                            seen_shopping = set()
                            unique_shopping = []
                            for result in st.session_state.shopping_results:
                                if result.get("product_name") not in seen_shopping:
                                    seen_shopping.add(result.get("product_name"))
                                    unique_shopping.append(result)
                            st.session_state.shopping_results = unique_shopping
                            st.success(f"✅ Found {len(shopping_results)} products with current pricing!")
                except Exception as e:
                    print(f"Error fetching shopping results: {e}")
                    st.warning("⚠️ Could not fetch current pricing data. Please check the Shopping tab later.")
        except Exception as exc:
            # Handle API errors in final response generation
            error_msg = str(exc).lower()
            if "503" in error_msg or "service unavailable" in error_msg or "upstream" in error_msg:
                response_content = "🚫 **Groq API is currently experiencing issues.** I was able to fetch the product information, but couldn't generate the final response. Please try asking your question again in a few minutes."
            else:
                response_content = f"⚠️ **Error generating response:** {exc}\n\nI fetched the product data successfully, but couldn't process the final response. Please try again."
        
        # Format response with better citations
        if state["content"]:
            response_content = format_response_with_citations(response_content, state["content"])
        
        # Add final AI response to state
        state["messages"].append(AIMessage(content=response_content))
        
    else:
        # No tool calls, just add the response
        response_content = response.content
        state["messages"].append(AIMessage(content=response_content))
    
    # Generate TLDR for the response
    tldr_content = generate_tldr(response_content)
    
    # Generate audio response using TLDR if TTS is enabled
    audio_response = None
    if st.session_state.get("tts_enabled", True) and tldr_content:
        # Only generate audio for non-error responses
        if not any(error_indicator in response_content for error_indicator in ["🚫", "⚠️", "⏰", "⏱️"]):
            with st.spinner("🔊 Generating audio summary..."):
                audio_response = text_to_speech(tldr_content)
    
    return response_content, state["content"], tldr_content, audio_response

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
    # Store TLDRs for each AI response
    if "tldrs" not in st.session_state:
        st.session_state.tldrs = {}
    # Store audio responses and TTS settings
    if "audio_responses" not in st.session_state:
        st.session_state.audio_responses = {}
    if "tts_enabled" not in st.session_state:
        st.session_state.tts_enabled = True
    if "play_latest_audio" not in st.session_state:
        st.session_state.play_latest_audio = False
    if "currently_playing_audio" not in st.session_state:
        st.session_state.currently_playing_audio = None
    if "audio_control_states" not in st.session_state:
        st.session_state.audio_control_states = {}
    if "last_played_audio" not in st.session_state:
        st.session_state.last_played_audio = None
    # Store product entities for tracking
    if "product_entities" not in st.session_state:
        st.session_state.product_entities = []
    # Store shopping results with images
    if "shopping_results" not in st.session_state:
        st.session_state.shopping_results = []

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
        st.write(f"**User Preferences:** {len(st.session_state.user_preferences)}")
        st.write(f"**Product Entities:** {len(st.session_state.product_entities)}")
        st.write(f"**Shopping Results:** {len(st.session_state.shopping_results)}")
        
        if st.session_state.user_preferences:
            st.subheader("🎯 Active Preferences")
            st.info("These preferences are automatically included in the AI context")
            sorted_prefs = sorted(st.session_state.user_preferences.items(), key=lambda x: x[1], reverse=True)
            for pref, weight in sorted_prefs[:5]:  # Show top 5
                st.write(f"• {pref}: {weight}/5")
            if len(sorted_prefs) > 5:
                st.write(f"... and {len(sorted_prefs) - 5} more")
        
        if st.session_state.product_entities:
            st.subheader("🛍️ Discussed Products")
            st.info("Products mentioned in our conversation")
            for entity in st.session_state.product_entities[:5]:  # Show top 5
                st.write(f"• **{entity.get('product_name')}** ({entity.get('brand')})")
            if len(st.session_state.product_entities) > 5:
                st.write(f"... and {len(st.session_state.product_entities) - 5} more")
        
        if st.session_state.shopping_results:
            st.subheader("🛒 Shopping Preview")
            st.info("Current prices and availability")
            for result in st.session_state.shopping_results[:3]:  # Show top 3
                with st.container():
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        if result.get('thumbnail'):
                            st.image(result['thumbnail'], width=60)
                    with col2:
                        st.write(f"**{result.get('title', 'Unknown')[:30]}...**")
                        st.write(f"💲 {result.get('price', 'N/A')}")
                    st.divider()
            if len(st.session_state.shopping_results) > 3:
                st.write(f"... and {len(st.session_state.shopping_results) - 3} more in Shopping tab")
        
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
        
        # TTS Controls
        st.subheader("🔊 Text-to-Speech")
        st.session_state.tts_enabled = st.checkbox("Enable TTS", value=st.session_state.tts_enabled, help="Enable automatic text-to-speech for AI responses")
        
        # API Status section
        st.subheader("🌐 API Status")
        st.info("💡 If you see API errors, it means Groq's servers are temporarily busy. Try again in a few minutes!")
        if st.button("🔄 Check Groq Status"):
            st.markdown("[Visit Groq Status Page](https://status.groq.com/)")
        
        # Debug info
        if st.session_state.currently_playing_audio:
            st.info(f"🎵 Currently playing: {st.session_state.currently_playing_audio}")
        else:
            st.info("🔇 No audio playing")
        
        if st.button("🎵 Test TTS"):
            test_audio = text_to_speech("Text to speech is working perfectly!")
            if test_audio:
                # Use a unique ID for test audio that won't conflict
                test_id = f"test_audio_{hash(str(test_audio))}"
                play_audio_hidden(test_audio, test_id)
                st.session_state.currently_playing_audio = test_id
                st.success("TTS Test successful!")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = [AIMessage(content=WELCOME_MSG)]
            st.session_state.content = {}
            st.session_state.tldrs = {}
            st.session_state.audio_responses = {}
            st.session_state.play_latest_audio = False
            st.session_state.currently_playing_audio = None
            st.session_state.audio_control_states = {}
            st.session_state.last_played_audio = None
            st.session_state.product_entities = []
            st.session_state.shopping_results = []
            st.session_state.bot_state = {
                "messages": st.session_state.messages,
                "content": st.session_state.content
            }
            st.rerun()

    # Always show tabs for better organization
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📰 Sources", "🛒 Shopping", "🧠 Preferences"])
    
    with tab1:
        # Render chat history
        for i, msg in enumerate(st.session_state.messages):
            if isinstance(msg, AIMessage):
                if msg.content and str(msg.content).strip():
                    with st.chat_message("assistant"):
                        st.markdown(msg.content)
                        
                        # Add controls for assistant messages (except welcome message)
                        if msg.content != WELCOME_MSG:
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                # Add audio replay/stop toggle button
                                audio_key = f"audio_{i}"
                                if audio_key in st.session_state.audio_responses:
                                    # Check if this audio is currently playing
                                    is_playing = st.session_state.currently_playing_audio == audio_key
                                    
                                    if is_playing:
                                        if st.button(f"⏹️ Stop", key=f"stop_{i}", help="Stop audio playback"):
                                            stop_audio(audio_key)
                                            st.session_state.currently_playing_audio = None
                                            st.session_state.audio_control_states[audio_key] = "stopped"
                                            st.rerun()
                                    else:
                                        if st.button(f"🔊 Replay", key=f"replay_{i}", help="Replay this response"):
                                            # Stop any currently playing audio first
                                            if st.session_state.currently_playing_audio:
                                                stop_audio(st.session_state.currently_playing_audio)
                                            
                                            # Reset states and play new audio
                                            st.session_state.last_played_audio = None
                                            st.session_state.currently_playing_audio = audio_key
                                            st.session_state.audio_control_states[audio_key] = "playing"
                                            play_audio_hidden(st.session_state.audio_responses[audio_key], audio_key)
                                            st.rerun()
                            
                            with col2:
                                # Add TLDR expander (non-interactive to avoid stopping audio)
                                tldr_key = f"tldr_{i}"
                                if tldr_key in st.session_state.tldrs:
                                    with st.expander(f"📝 TLDR", expanded=False):
                                        st.info(st.session_state.tldrs[tldr_key])
                            
            elif isinstance(msg, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(msg.content)
    
    with tab2:
        if st.session_state.content:
            valid_posts = [item for item in st.session_state.content if isinstance(item, dict) and 'title' in item]
            st.write(f"📚 **Source Material:** {len(valid_posts)} Reddit discussions referenced in the chat")
            st.info("💡 Click on any hyperlink in the chat responses to jump directly to these Reddit discussions!")
            
            for i, post in enumerate(valid_posts):
                subreddit = post.get('subreddit', 'unknown')
                
                with st.expander(f"📝 {post.get('title', 'Unknown Title')} - r/{subreddit}", expanded=False):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Description:** {post.get('description', 'No description available')}")
                        st.write(f"**Subreddit:** r/{subreddit}")
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
        else:
            st.info("🔍 **No Reddit posts found yet**")
            st.markdown("Ask questions about products and I'll search Reddit for relevant discussions and reviews!")
            st.markdown("**Try asking about:**")
            st.markdown("- 'Best wireless headphones under $200'")
            st.markdown("- 'Sony camera vs Canon for beginners'")
            st.markdown("- 'Gaming laptop recommendations'")
    
    with tab3:
        # Shopping Results with Images
        if st.session_state.shopping_results:
            st.write(f"🛒 **Shopping Results:** {len(st.session_state.shopping_results)} products found")
            st.info("💰 Current prices and availability from various retailers")
            
            # Display shopping results in a grid
            cols_per_row = 3
            shopping_results = st.session_state.shopping_results
            
            for i in range(0, len(shopping_results), cols_per_row):
                cols = st.columns(cols_per_row)
                
                for j, result in enumerate(shopping_results[i:i+cols_per_row]):
                    with cols[j]:
                        # Product image
                        if result.get('thumbnail'):
                            st.image(result['thumbnail'], width=200, use_container_width=True)
                        else:
                            st.empty()  # Placeholder if no image
                        
                        # Product details
                        st.markdown(f"**{result.get('title', 'Unknown Product')}**")
                        
                        # Price with styling
                        price = result.get('price', 'Price not available')
                        st.markdown(f"💲 **{price}**")
                        
                        # Product links
                        if result.get('product_link'):
                            st.markdown(f"[🔗 View Product]({result['product_link']})")
                        
                        # Extra spacing
                        st.markdown("---")
        else:
            st.info("🛒 **No shopping results found yet**")
            st.markdown("Ask about specific products and I'll find current prices and availability!")
            st.markdown("**Try asking about:**")
            st.markdown("- 'Show me iPhone 15 prices'")
            st.markdown("- 'Find Sony headphones deals'")
            st.markdown("- 'Best laptop under $1000'")
    
    with tab4:
        render_preference_graph()

    # Check if we need to play the latest audio response
    if st.session_state.play_latest_audio and st.session_state.audio_responses:
        latest_audio_key = max(st.session_state.audio_responses.keys())
        latest_audio = st.session_state.audio_responses[latest_audio_key]
        
        # Only play if we haven't already played this audio and no other audio is playing
        if (latest_audio and 
            st.session_state.currently_playing_audio != latest_audio_key and
            st.session_state.last_played_audio != latest_audio_key and
            st.session_state.currently_playing_audio is None):  # Extra check to prevent duplicates
            
            # Ensure no audio is playing before starting new one
            if st.session_state.currently_playing_audio:
                stop_audio(st.session_state.currently_playing_audio)
                # Add a small delay to ensure the previous audio stops
                import time
                time.sleep(0.1)
            
            play_audio_hidden(latest_audio, latest_audio_key)
            st.session_state.last_played_audio = latest_audio_key
            st.session_state.currently_playing_audio = latest_audio_key
            
        st.session_state.play_latest_audio = False
    
    # Add periodic audio state check (JavaScript-based)
    if st.session_state.currently_playing_audio:
        st.markdown(f"""
        <script>
            // Check if currently playing audio has ended
            const currentAudio = document.getElementById('{st.session_state.currently_playing_audio}');
            if (currentAudio && currentAudio.ended) {{
                // Audio has ended naturally, will be handled by the app
            }}
        </script>
        """, unsafe_allow_html=True)

    # Chat input with microphone button
    col1, col2 = st.columns([1, 12])
    
    with col1:
        if st.button(":material/mic:", key="mic_button", help="Click to use voice input"):
            st.info("🎙️ Voice input feature - Coming soon! For now, please type your message in the chat box.")
    
    with col2:
        if prompt := st.chat_input("What are you looking for today?"):
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response_content, updated_content, tldr_content, audio_response = generate_response(
                        st.session_state.bot_state, 
                        prompt
                    )
                    
                    # Update session state
                    st.session_state.messages = st.session_state.bot_state["messages"]
                    st.session_state.content = updated_content
                    
                    # Update preference graph with the latest user query
                    update_user_preferences(prompt)
                    
                    st.markdown(response_content)
                    
                    # Store audio response and play it
                    msg_index = len(st.session_state.messages) - 1
                    msg_key = f"audio_{msg_index}"
                    
                    if audio_response:
                        # Stop any currently playing audio first to prevent duplicates
                        if st.session_state.currently_playing_audio:
                            stop_audio(st.session_state.currently_playing_audio)
                            st.session_state.currently_playing_audio = None
                        
                        st.session_state.audio_responses[msg_key] = audio_response
                        # Only set play flag if this is a new audio response and no audio is currently playing
                        if msg_key not in st.session_state.audio_control_states and not st.session_state.currently_playing_audio:
                            st.session_state.play_latest_audio = True
                            st.session_state.audio_control_states[msg_key] = "playing"
                    
                    # Store TLDR and show button
                    tldr_key = f"tldr_{msg_index}"
                    
                    # Store the TLDR
                    if tldr_content:
                        st.session_state.tldrs[tldr_key] = tldr_content
                        
                        # Show TLDR button
                        if st.button("📝 TLDR", key=f"tldr_btn_{msg_index}"):
                            pass  # Just to trigger rerun for expander state
                        with st.expander("📄 TLDR Summary", expanded=False):
                            st.info(tldr_content)
            
            # Force rerun to update the chat
            st.rerun()

def update_user_preferences(user_query: str) -> None:
    """Extract preferences from the latest user query via the LLM and merge into the graph."""
    if not user_query or not isinstance(user_query, str):
        return

    system_prompt = (
        """
        You are a smart shopping assistant that helps in extracting updated user's shopping or lifestyle preferences from user queries.
        You are given the user's **Previous Preferences** and a new **User Query**
        You extract user preferences from shopping conversations in the form of (subject, relation, object) triplets.
        Note if the preferences is conflicting with any other preference, you update the user's previous preferences and keep it dynamic.

        Always use "user" as the subject, if user is talking about themselves or their lifestyle.
        Respond only with a JSON using this triplet extraction format:
        { user_preferences : [ {"subject": "user", "relation": "bought", "object": "lenovo thinkpad"}]}
        
        Example:
        From a user shopping conversation like:
        "I recently bought a Lenovo ThinkPad and I’m looking for a mirrorless camera under $800."

        Respond only with a JSON using this triplet extraction format:
        { user_preferences : [ {"subject": "user", "relation": "bought", "object": "lenovo thinkpad"},
        {"subject": "user", "relation": "interested in", "object": "mirrorless camera"}]}
        
        No extra explanation or text.
"""
    )
    
    user_prompt = f"""
    User Query: {user_query}
    Previous Preferences: {st.session_state.user_preferences}
    """

    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            response_format={"type": "json_object"}
        )
        
        # response = llm.invoke([
        #     SystemMessage(content=system_prompt),
        #     HumanMessage(content=user_prompt),
        # ])
        txt_resp = response.content if isinstance(response.content, str) else str(response.content)
        txt_resp = txt_resp.strip()
        
        if not txt_resp.startswith("{"):
            match = re.search(r"\{[\s\S]*\}", txt_resp)
            txt_resp = match.group(0) if match else "{}"
        prefs_fragment: dict[str, int] = json.loads(txt_resp)

        # # Merge into existing store
        # prefs_store: dict[str, int] = st.session_state.get("user_preferences", {})
        # for k, v in prefs_fragment.items():
        #     try:
        #         weight = int(v)
        #     except Exception:
        #         continue
        #     prefs_store[k] = max(prefs_store.get(k, 0), weight)
        st.session_state.user_preferences = prefs_fragment
    except Exception as exc:
        # Silently fail for preference extraction to not interrupt the main flow
        pass  # Log failures silently to avoid interrupting user experience

if __name__ == "__main__":
    main()
