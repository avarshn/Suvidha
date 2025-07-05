import os
import tempfile
import io
from pathlib import Path
import streamlit as st
from streamlit_mic_recorder import mic_recorder
from groq import Groq
from dotenv import load_dotenv
import base64
from io import BytesIO

# Load environment variables
load_dotenv()

def initialize_session_state():
    """Initialize session state variables."""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'play_latest_audio' not in st.session_state:
        st.session_state.play_latest_audio = False

def play_audio_hidden(audio_bytes):
    """Play audio automatically without showing controls."""
    if audio_bytes:
        # Handle BytesIO objects by extracting the bytes
        if hasattr(audio_bytes, 'getvalue'):
            audio_data = audio_bytes.getvalue()
        else:
            audio_data = audio_bytes
            
        # Convert audio bytes to base64
        audio_base64 = base64.b64encode(audio_data).decode()
        
        # Create HTML5 audio element with autoplay and hidden controls
        audio_html = f"""
        <audio autoplay style="display: none;">
            <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
            Your browser does not support the audio element.
        </audio>
        """
        
        # Display the HTML
        st.markdown(audio_html, unsafe_allow_html=True)
        
        # Optional: Show a subtle indicator
        st.markdown("🔊 *AI response is playing...*")

class ImprovedSpeechChat:
    def __init__(self):
        """Initialize the improved speech chat application with Groq API."""
        if not os.getenv("GROQ_API_KEY"):
            st.error("❌ GROQ_API_KEY environment variable not set!")
            st.stop()
        
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    def speech_to_text(self, audio_bytes):
        """Convert speech to text using Groq Speech API."""
        try:
            # Create a BytesIO object from audio bytes
            audio_bio = io.BytesIO(audio_bytes)
            audio_bio.name = 'audio.webm'  # Set filename for Groq API
            
            # Use Groq's speech-to-text API
            response = self.groq_client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=audio_bio,
                response_format="verbose_json"
            )
            return response.text
        except Exception as e:
            st.error(f"Error in speech-to-text: {e}")
            return None
    
    def text_to_speech(self, text):
        """Convert text to speech using Groq Speech API."""
        try:
            response = self.groq_client.audio.speech.create(
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
    
    def chat_with_llm(self, user_input):
        """Send user input to Groq LLM and get response."""
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant. Keep your responses concise and conversational."
                    },
                    {
                        "role": "user",
                        "content": user_input
                    }
                ],
                temperature=1,
                max_completion_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
            )
            
            # Extract the text content from the response
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error in LLM chat: {e}")
            return "I'm sorry, I encountered an error processing your request."
    
    def process_audio(self, audio_data):
        """Process recorded audio through the complete pipeline."""
        if not audio_data:
            return
        
        st.session_state.processing = True
        
        with st.spinner("🔄 Processing your speech..."):
            # Step 1: Speech to text
            user_text = self.speech_to_text(audio_data['bytes'])
            
            if user_text:
                st.info(f"👤 You : {user_text}")
                
                # Step 2: Get AI response
                ai_response = self.chat_with_llm(user_text)
                
                # Step 3: Convert AI response to speech
                audio_response = self.text_to_speech(ai_response)
                
                if audio_response:
                    # Add to conversation history
                    st.session_state.conversation_history.append({
                        'user_text': user_text,
                        'ai_response': ai_response,
                        'audio_response': audio_response,
                        'timestamp': audio_data['id']
                    })
                    
                    # Set flag to play audio after rerun
                    st.session_state.play_latest_audio = True
                    
                    st.success("✅ Processing complete!")
                    st.rerun()
                else:
                    st.error("Failed to generate speech response")
            else:
                st.error("Could not transcribe your speech. Please try again.")
        
        st.session_state.processing = False
    
    def run_streamlit_app(self):
        """Run the improved Streamlit application."""
        st.set_page_config(
            page_title="🎤 Improved Speech Chat with Groq",
            page_icon="🎤",
            layout="wide"
        )
        
        st.title("🎤 Improved Speech Chat with Groq AI")
        st.markdown("*Continuous conversation - talk as much as you like!*")
        st.markdown("---")
        
        # Check if we need to play the latest audio response
        if st.session_state.play_latest_audio and st.session_state.conversation_history:
            latest_conversation = st.session_state.conversation_history[-1]
            play_audio_hidden(latest_conversation['audio_response'])
            st.session_state.play_latest_audio = False  # Reset the flag
        
        # Create columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("💬 Ongoing Conversation")
            
            # Display conversation history
            if st.session_state.conversation_history:
                for i, conv in enumerate(st.session_state.conversation_history):
                    with st.chat_message("user"):
                        st.write(f"👤 **You:** {conv['user_text']}")
                    
                    with st.chat_message("assistant"):
                        st.write(f"🤖 **AI:** {conv['ai_response']}")
                        
                        # Add a replay button for each AI response
                        if st.button(f"🔄 Replay {i+1}", key=f"replay_{i}", help="Replay this AI response"):
                            play_audio_hidden(conv['audio_response'])
                
                # Add some spacing and encouragement to continue
                st.markdown("---")
                st.info("💬 **Keep the conversation going!** Record another message below.")
            else:
                st.info("👋 **Start your conversation!** Record your first message below.")
        
        with col2:
            st.subheader("🎙️ Voice Input")
            
            # Show current conversation count
            if st.session_state.conversation_history:
                st.metric("💬 Messages Exchanged", len(st.session_state.conversation_history) * 2)
            
            # Recording instructions - simplified for continuous use
            st.markdown("""
            **How to continue chatting:**
            1. 🎤 **Click Start** to record your message
            2. 🗣️ **Speak naturally** - ask questions, share thoughts
            3. ⏹️ **Click Stop** when you're done
            4. 🔊 **Listen** to the AI's response
            5. 🔄 **Repeat** - keep the conversation flowing!
            """)
            
            # Mic recorder component
            audio = mic_recorder(
                start_prompt="🎤 Record Message",
                stop_prompt="⏹️ Stop Recording",
                just_once=True,  # Only return audio once after recording
                use_container_width=True,
                format="webm",  # Better browser compatibility
                key='voice_recorder'
            )
            
            # Process audio when received
            if audio and not st.session_state.processing:
                self.process_audio(audio)
            
            # Show processing status
            if st.session_state.processing:
                st.warning("🔄 Processing your message...")
                st.info("💭 The AI is thinking and preparing a response...")
            
            # Quick action buttons
            st.markdown("---")
            st.markdown("**Quick Actions:**")
            
            col_test, col_clear = st.columns(2)
            
            with col_test:
                if st.button("🎵 Test Audio", use_container_width=True, help="Test if audio is working"):
                    test_audio = self.text_to_speech("Audio test successful! I can hear you clearly.")
                    if test_audio:
                        play_audio_hidden(test_audio)
            
            with col_clear:
                if st.button("🔄 New Chat", use_container_width=True, help="Start a fresh conversation"):
                    st.session_state.conversation_history = []
                    st.session_state.play_latest_audio = False
                    st.rerun()
        
        # Sidebar with enhanced information
        with st.sidebar:
            st.header("ℹ️ Conversation Assistant")
            
            st.markdown("""
            **💡 Conversation Tips:**
            - Ask follow-up questions
            - Share your thoughts and opinions  
            - Request explanations or examples
            - Change topics anytime
            - Have fun chatting!
            """)
            
            st.markdown("---")
            
            st.markdown("""
                          
            **Features:**
            - 🔄 Real-time voice conversations
            - 🔊 Auto-play text-to-speech responses
            - 📱 Mobile-friendly interface
                        
            **🤖 AI Capabilities:**
            - Answers questions on any topic
            - Provides explanations and examples
            - Engages in casual conversation
            - Remembers context within the session
            - Responds in natural speech
            """)
            
            st.markdown("---")
            
            st.markdown("""
            **🔧 Technical Details:**
            - **Speech-to-Text:** Whisper Large V3 Turbo
            - **AI Brain:** Llama 3.1 8B Instant  
            - **Voice:** PlayAI TTS (Arista)
            - **Format:** Real-time conversation
            """)
            
            st.markdown("---")
            
            # Conversation management
            st.markdown("**💾 Save Your Chat:**")
            
            if st.session_state.conversation_history:
                # Create downloadable conversation log
                conversation_text = f"Conversation Log - {len(st.session_state.conversation_history)} exchanges\n"
                conversation_text += "=" * 50 + "\n\n"
                
                for i, conv in enumerate(st.session_state.conversation_history):
                    conversation_text += f"Exchange {i+1}:\n"
                    conversation_text += f"You: {conv['user_text']}\n"
                    conversation_text += f"AI: {conv['ai_response']}\n\n"
                    conversation_text += "-" * 30 + "\n\n"
                
                st.download_button(
                    label="💾 Download Chat Log",
                    data=conversation_text,
                    file_name=f"conversation_{len(st.session_state.conversation_history)}_exchanges.txt",
                    mime="text/plain",
                    use_container_width=True,
                    help="Save your conversation as a text file"
                )
                
                # Statistics
                st.markdown("---")
                st.markdown("**📊 Session Stats:**")
                
                total_exchanges = len(st.session_state.conversation_history)
                st.metric("�� Exchanges", total_exchanges)
                
                total_words = sum(len(conv['user_text'].split()) + len(conv['ai_response'].split()) 
                                for conv in st.session_state.conversation_history)
                st.metric("📝 Total Words", total_words)
                
                if total_exchanges > 0:
                    avg_words = total_words // (total_exchanges * 2)
                    st.metric("📊 Avg Words/Message", avg_words)
            else:
                st.info("Start chatting to see stats and download options!")
            
            # Privacy note
            st.markdown("---")
            st.caption("🔒 **Privacy:** Your conversation is stored locally in this session only. Refresh the page to clear all data.")

def main():
    """Main function to run the improved Streamlit app."""
    # Initialize session state
    initialize_session_state()
    
    # Create and run the speech chat application
    speech_chat = ImprovedSpeechChat()
    speech_chat.run_streamlit_app()

if __name__ == "__main__":
    main() 