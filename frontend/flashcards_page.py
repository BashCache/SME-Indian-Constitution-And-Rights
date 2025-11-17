"""
Dedicated Flashcards Study Page
Provides a clean, focused interface for studying with flashcards
"""

import streamlit as st
import json
import requests
from typing import Dict, List, Any
import os
import sys

# Add root directory to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from frontend.flashcard_component import display_flashcards, parse_flashcard_data_from_response, reset_flashcard_session, handle_flashcard_response

# Configure page
st.set_page_config(
    page_title="🎴 Flashcards Study - Constitutional AI",
    page_icon="🎴",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Backend URL
BACKEND_URL = "http://localhost:8000"

def create_session_for_flashcards(username: str = "flashcard_user") -> str:
    """Create a session for flashcard study"""
    try:
        payload = {
            "username": username,
            "title": f"Flashcard Study {st.session_state.get('study_session_count', 1)}"
        }
        response = requests.post(f"{BACKEND_URL}/sessions/create", json=payload, timeout=10)
        if response.status_code == 200:
            return response.json()["session_id"]
        else:
            # Fallback session ID
            return f"flashcard_{username}_{hash(str(payload))}"
    except:
        # Fallback session ID
        import uuid
        return f"flashcard_session_{str(uuid.uuid4())[:8]}"

def generate_flashcards(topic: str, num_cards: int = 10, difficulty: str = "medium", card_type: str = "mixed") -> Dict[str, Any]:
    """Generate flashcards using the backend API"""
    try:
        # Ensure we have a valid session
        if "session_id" not in st.session_state or not st.session_state.session_id:
            st.session_state.session_id = create_session_for_flashcards()
            st.info(f"Created new session for flashcard study: {st.session_state.session_id}")
        
        session_id = st.session_state.session_id
        
        payload = {
    "message": f"""
Return flashcards ONLY in the following JSON format:

{{
  "topic": "{topic}",
  "difficulty": "{difficulty}",
  "card_type": "{card_type}",
  "cards": [
    {{ "question": "...", "answer": "..." }},
    {{ "question": "...", "answer": "..." }}
  ]
}}

NO MARKDOWN. NO TEXT. ONLY VALID JSON.

Create {num_cards} flashcards about {topic} with {difficulty} difficulty and {card_type} card type.
""",
    "session_id": session_id,
    "filepath": None
}

        print(f"Flashcards: {session_id}")
        st.write(f"🔍 Debug: Sending request with session_id: {session_id}")  # Debug info
        
        response = requests.post(f"{BACKEND_URL}/chat/langchain", json=payload, timeout=90)
        st.write(f"🔍 Debug: Response status: {response.status_code}")  # Debug info
        
        if response.status_code == 200:
            data = response.json()
            response_text = data.get("response", "")
            
            st.write(f"🔍 Debug: Response text length: {len(response_text)}")  # Debug info
            if len(response_text) > 100:
                st.write(f"🔍 Debug: Response preview: {response_text[:200]}...")  # Debug info
            
            # Parse flashcard data from response
            flashcard_data = parse_flashcard_data_from_response(response_text)

            if flashcard_data:
                st.write("🔍 Parsed structure:", flashcard_data.keys())
                st.write("🔍 Number of cards:", len(flashcard_data.get('cards', [])))
                if flashcard_data.get('cards'):
                    st.write("🔍 First card keys:", flashcard_data['cards'][0].keys())

            print(f"FLashcard data: {flashcard_data}")
            if flashcard_data:
                st.write(f"🔍 Debug: Successfully parsed flashcard data with {len(flashcard_data.get('cards', []))} cards")
            else:
                st.write("🔍 Debug: Failed to parse flashcard data from response")
                
            return flashcard_data
        else:
            error_text = response.text
            st.error(f"❌ Backend Error: {response.status_code}")
            st.error(f"Response: {error_text}")
            return None
    except requests.exceptions.Timeout:
        st.error("⏰ Flashcard generation is taking longer than expected. Please try again.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Connection error: {str(e)}")
        return None

def flashcards_study_page():
    """Main flashcards study interface"""
    
    # Page header
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; margin-bottom: 2rem; border-radius: 10px;">
        <h1>🎴 Interactive Flashcards</h1>
        <p>Study Constitutional Law with Interactive Q&A Cards</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("🏠 Back to Chat", use_container_width=True):
            st.switch_page("frontend/streamlit_app.py")
    with col2:
        st.markdown("### 📚 Study Mode")
    with col3:
        if st.button("🔄 New Study Set", use_container_width=True):
            # Clear current flashcard data
            if 'flashcard_data' in st.session_state:
                del st.session_state.flashcard_data
            reset_flashcard_session()
            st.rerun()
    
    st.markdown("---")
    
    # Flashcard generation form (only show if no active flashcards)

    # In flashcards_page.py, add before the form:
    if st.button("🧪 Test Backend Connection"):
        try:
            response = requests.get(f"{BACKEND_URL}/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ Backend is reachable!")
            else:
                st.error(f"❌ Backend returned: {response.status_code}")
        except Exception as e:
            st.error(f"❌ Cannot reach backend: {str(e)}")

    if 'flashcard_data' not in st.session_state or st.session_state.flashcard_data is None:
        st.markdown("### 🎯 Create New Flashcard Set")
        
        with st.form("flashcard_generator"):
            col1, col2 = st.columns(2)
            
            with col1:
                topic = st.text_input(
                    "📝 Topic",
                    placeholder="e.g., Article 21, Fundamental Rights, Constitutional Amendments",
                    help="Enter the constitutional law topic you want to study"
                )
                
                difficulty = st.selectbox(
                    "🎚️ Difficulty Level",
                    options=["easy", "medium", "hard"],
                    index=1,
                    help="Choose the difficulty level for your study session"
                )
            
            with col2:
                num_cards = st.slider(
                    "🎴 Number of Cards",
                    min_value=2,
                    max_value=20,
                    value=10,
                    help="How many flashcards do you want to study?"
                )
                
                card_type = st.selectbox(
                    "🎭 Card Type",
                    options=["mixed", "definitions", "articles", "cases"],
                    help="Focus on specific types of constitutional knowledge"
                )
            
            # Submit button
            submitted = st.form_submit_button("🎴 Generate Flashcards", use_container_width=True, type="primary")
            
            if submitted:
                if not topic.strip():
                    st.error("Please enter a topic for your flashcards.")
                else:
                    with st.spinner(f"🎴 Generating {num_cards} flashcards about {topic}..."):
                        flashcard_data = generate_flashcards(topic, num_cards, difficulty, card_type)
                        print(f"flashcard in flashcards study page: {flashcard_data}")
                        
                        if flashcard_data:
                            # Use the new response handler for the updated flashcard format
                            handle_flashcard_response(flashcard_data)
                            st.session_state.flashcard_data = flashcard_data
                            reset_flashcard_session()
                        else:
                            st.error("❌ Failed to generate flashcards. Please try again.")
        
        # Example topics section
        st.markdown("---")
        st.markdown("### 💡 Popular Study Topics")
        
        example_topics = [
            "Fundamental Rights",
            "Directive Principles",
            "Article 21 - Right to Life",
            "Constitutional Amendments",
            "Supreme Court Judgments",
            "Separation of Powers",
            "Federal Structure",
            "Emergency Provisions"
        ]
        
        cols = st.columns(4)
        for i, topic in enumerate(example_topics):
            with cols[i % 4]:
                if st.button(f"📚 {topic}", key=f"topic_{i}", use_container_width=True):
                    # Auto-generate flashcards for the selected topic
                    with st.spinner(f"🎴 Generating flashcards for {topic}..."):
                        flashcard_data = generate_flashcards(topic, 10, "medium", "mixed")
                        print(f"Flashcard in flashcards study page: {flashcard_data}")
                        if flashcard_data:
                            # Use the new response handler for the updated flashcard format
                            handle_flashcard_response(flashcard_data)
                            st.session_state.flashcard_data = flashcard_data
                            reset_flashcard_session()
    
    # Display active flashcards
    if 'flashcard_data' in st.session_state and st.session_state.flashcard_data:
        st.markdown("---")
        
        # Flashcard controls
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("📊 Study Stats", use_container_width=True):
                show_study_stats()
        with col2:
            flashcard_data = st.session_state.flashcard_data
            st.markdown(f"**Current Set:** {flashcard_data.get('topic', 'Unknown Topic')}")
        with col3:
            if st.button("🎯 New Topic", use_container_width=True):
                if 'flashcard_data' in st.session_state:
                    del st.session_state.flashcard_data
                reset_flashcard_session()
                st.rerun()
        
        # Display the interactive flashcards using the new handler
        if 'flashcard_data' in st.session_state and st.session_state.flashcard_data:
            # Check if this is the new format with flashcard_file
            if isinstance(st.session_state.flashcard_data, dict) and st.session_state.flashcard_data.get('flashcard_file'):
                handle_flashcard_response(st.session_state.flashcard_data)
            else:
                # Fallback to old display method
                display_flashcards(st.session_state.flashcard_data)
        
    else:
        # Welcome message when no flashcards are active
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 10px; margin: 2rem 0;">
            <h2>🎴 Welcome to Interactive Flashcards!</h2>
            <p style="font-size: 1.2em; color: #666;">
                Create your personalized study set to begin learning constitutional law concepts
                through interactive Q&A cards.
            </p>
            <p style="color: #888;">
                📝 Enter a topic above or choose from popular topics to get started
            </p>
        </div>
        """, unsafe_allow_html=True)

def show_study_stats():
    """Display study statistics in a modal-like interface"""
    if 'flashcard_data' in st.session_state and st.session_state.flashcard_data:
        flashcard_data = st.session_state.flashcard_data
        cards = flashcard_data.get('cards', [])
        progress = st.session_state.get('flashcard_progress', [])
        
        # Study statistics
        total_cards = len(cards)
        completed_cards = len(progress)
        completion_rate = (completed_cards / total_cards * 100) if total_cards > 0 else 0
        
        st.markdown("### 📊 Study Session Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📚 Total Cards", total_cards)
        with col2:
            st.metric("✅ Completed", completed_cards)
        with col3:
            st.metric("📈 Progress", f"{completion_rate:.1f}%")
        with col4:
            st.metric("🎯 Remaining", total_cards - completed_cards)
        
        # Progress bar
        if total_cards > 0:
            st.progress(completion_rate / 100, text=f"Study Progress: {completion_rate:.1f}%")
        
        # Topic information
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**📝 Topic:** {flashcard_data.get('topic', 'Unknown')}")
            st.markdown(f"**🎚️ Difficulty:** {flashcard_data.get('difficulty', 'medium').title()}")
        with col2:
            st.markdown(f"**🎭 Type:** {flashcard_data.get('card_type', 'mixed').title()}")
            current_card = st.session_state.get('current_card_index', 0)
            st.markdown(f"**📍 Current Card:** {current_card + 1} of {total_cards}")

def main():
    """Main entry point for the flashcards page"""
    # Initialize session state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "session_id" not in st.session_state:
        st.session_state.session_id = "flashcard_session"
    
    # Check if user is authenticated (you might want to add this)
    flashcards_study_page()

if __name__ == "__main__":
    main()
