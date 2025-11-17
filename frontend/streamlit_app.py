import streamlit as st
import requests
import json
from typing import Dict, List, Optional
import os
from datetime import datetime
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
from db_models.crud_operations import fetch_conversation_messages
from frontend.flashcard_component import display_flashcards, parse_flashcard_data_from_response, reset_flashcard_session
from frontend.quiz_component import display_interactive_quiz, parse_quiz_data_from_response
import uuid

# Configure page
st.set_page_config(
    page_title="Constitutional AI Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for ChatGPT-like styling
def load_css():
    """Load CSS from external file"""
    css_file = os.path.join(os.path.dirname(__file__), "styles.css")
    if os.path.exists(css_file):
        with open(css_file, "r") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    else:
        # Fallback CSS if file not found
        st.markdown("""
        <style>
            .main-header {
                text-align: center;
                padding: 2rem 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                margin-bottom: 2rem;
                border-radius: 10px;
            }
            .chat-container {
                max-width: 800px;
                margin: 0 auto;
                padding: 1rem;
            }
        </style>
        """, unsafe_allow_html=True)

# Load the CSS
load_css()

# Backend URL
BACKEND_URL = "http://localhost:8000"

def check_backend_status() -> bool:
    """Quick check if backend is available"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=1)
        return response.status_code == 200
    except:
        return False

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "session_id" not in st.session_state:
    st.session_state.session_id = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "current_flashcard_data" not in st.session_state:
    st.session_state.current_flashcard_data = None
if "current_quiz_data" not in st.session_state:
    st.session_state.current_quiz_data = None

def authenticate_user(username: str, password: str) -> bool:
    """Authenticate user with existing auth endpoint"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/auth/login",
            json={"username": username, "password": password},
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("success", False)
        return False
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        return False

def create_session(username: str) -> str:
    """Create new chat session using existing sessions endpoint (with auto-generated title)"""
    title = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    return create_session_with_title(username, title)

def create_session_with_title(username: str, title: str) -> str:
    """Create new chat session with custom title using existing sessions endpoint"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/sessions/create",
            json={"username": username, "title": title}
        )
        if response.status_code == 200:
            return response.json()["session_id"]
    except requests.exceptions.RequestException as e:
        pass
    
    # Fallback session ID
    return f"{username}_{uuid.uuid4().hex[:8]}"

def upload_file(file_data, filename: str) -> Dict:
    """Upload file using existing upload endpoint"""
    try:
        files = {"file": (filename, file_data, "application/octet-stream")}
        data = {
            "username": st.session_state.username,
            "session_id": st.session_state.session_id
        }
        
        response = requests.post(f"{BACKEND_URL}/upload", files=files, data=data, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"message": f"Upload failed with status {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"message": f"Upload error: {str(e)}"}

def save_message_to_session(session_id: str, user_message: str, ai_response: str) -> bool:
    """Save a message exchange to the session"""
    try:
        payload = {
            "session_id": session_id,
            "user_message": user_message,
            "ai_response": ai_response
        }
        response = requests.post(f"{BACKEND_URL}/sessions/messages", json=payload, timeout=30)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        # In demo mode, we'll rely on local session state
        return False

def send_message(message: str, history: List[Dict], uploaded_files: List = None) -> str:
    """Send message using existing chat endpoint"""
    try:
        payload = {
            "message": message,
            "session_id": st.session_state.session_id,
            "filepath": None  # Add filepath if available from uploaded files
        }
        
        # Detect if this might be a special generation request (longer processing)
        video_keywords = ['video', 'create video', 'make video', 'generate video', 'video explanation', 'visual explanation']
        quiz_keywords = ['quiz', 'test', 'questions', 'mcq', 'generate', 'create quiz', 'export', 'pdf']
        flashcard_keywords = ['flashcard', 'flashcards', 'flash card', 'study cards', 'create flashcards', 'make flashcards']
        
        is_video_request = any(keyword in message.lower() for keyword in video_keywords)
        is_quiz_request = any(keyword in message.lower() for keyword in quiz_keywords)
        is_flashcard_request = any(keyword in message.lower() for keyword in flashcard_keywords)
        
        # Use longer timeout for generation requests
        timeout = 180 if is_video_request else (180 if (is_quiz_request or is_flashcard_request) else 180)
        
        response = requests.post(f"{BACKEND_URL}/chat/langchain", json=payload, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            response_text = data.get("response", data.get("message", "No response received"))
            
            # Check if this is a flashcard response
            if is_flashcard_request and '🎴' in response_text:
                # Store flashcard data in session state for interactive display
                flashcard_data = parse_flashcard_data_from_response(response_text)
                if flashcard_data:
                    st.session_state.current_flashcard_data = flashcard_data
                    reset_flashcard_session()  # Reset any previous flashcard session
            
            # Check if this is an interactive quiz response
            if is_quiz_request and '🎯' in response_text:
                # Store quiz data in session state for interactive display
                quiz_data = parse_quiz_data_from_response(response_text)
                if quiz_data:
                    st.session_state.current_quiz_data = quiz_data
            
            return response_text
        else:
            return f"❌ Error: {response.status_code} - {response.text}"
    except requests.exceptions.Timeout:
        if is_video_request:
            return "⏰ Video generation is taking longer than expected. Your video may still be processing in the background. Please check the generated_videos folder for the completed video."
        elif is_quiz_request:
            return "⏰ Quiz generation is taking longer than expected. The quiz may still be processing in the background. Please check your documents folder for exported files."
        elif is_flashcard_request:
            return "⏰ Flashcard generation is taking longer than expected. Please try again in a moment."
        else:
            return "⏰ Request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        # Demo mode response when backend is unavailable
        return f"❌ Connection error: {str(e)}"

def get_user_sessions(username: str) -> List[Dict]:
    """Get user sessions using existing sessions endpoint"""
    try:
        response = requests.get(f"{BACKEND_URL}/sessions/{username}")
        if response.status_code == 200:
            return response.json()
        return []
    except requests.exceptions.RequestException:
        # Return demo sessions when backend unavailable
        return [
            {
                "session_id": f"{username}_demo_1", 
                "title": "Demo Session 1", 
                "started_at": "2024-11-14T10:00:00"
            },
            {
                "session_id": f"{username}_demo_2", 
                "title": "Constitutional Rights Chat", 
                "started_at": "2024-11-14T11:30:00"
            }
        ]

@st.cache_data(ttl=5)  # Cache for 5 seconds
def get_user_sessions_cached(username: str) -> List[Dict]:
    """Get user sessions with Streamlit caching to reduce API calls"""
    return get_user_sessions(username)

def get_session_messages(session_id: str) -> List[Dict]:
    """Get messages for a specific session using get_session_details_and_messages endpoint"""
    try:
        session_conversation_deets = fetch_conversation_messages(session_id)
        response = {
            "session_id": session_id,
            "username": session_id.split('_')[0],
            "messages": [
                {"role": m["role"], "content": m["content"]}
                for m in session_conversation_deets
            ]
        }
        if response:
            data = response
            # Extract messages from the response
            messages = data.get("messages", [])
            formatted_messages = []
            
            for msg in messages:
                if isinstance(msg, dict):
                    # Handle different possible backend formats
                    if "role" in msg and "content" in msg:
                        formatted_messages.append(msg)
                    elif "user_message" in msg and "ai_response" in msg:
                        formatted_messages.append({"role": "user", "content": msg["user_message"]})
                        formatted_messages.append({"role": "assistant", "content": msg["ai_response"]})
                    elif "message" in msg and "sender" in msg:
                        role = "user" if msg["sender"] == "user" else "assistant"
                        formatted_messages.append({"role": role, "content": msg["message"]})
                    elif "user_input" in msg and "assistant_response" in msg:
                        formatted_messages.append({"role": "user", "content": msg["user_input"]})
                        formatted_messages.append({"role": "assistant", "content": msg["assistant_response"]})
            
            return formatted_messages
        return []
    except requests.exceptions.RequestException:
        # Return demo messages for demo sessions
        if "demo" in session_id.lower():
            return [
                {"role": "user", "content": "What are fundamental rights?"},
                {"role": "assistant", "content": "Fundamental rights are basic human rights enshrined in the Constitution that protect citizens from arbitrary state action. They include rights to equality, freedom, protection against exploitation, freedom of religion, cultural and educational rights, and the right to constitutional remedies."},
                {"role": "user", "content": "Tell me about Article 21"},
                {"role": "assistant", "content": "Article 21 of the Indian Constitution guarantees the right to life and personal liberty. It states that 'No person shall be deprived of his life or personal liberty except according to procedure established by law.' This is one of the most important fundamental rights and has been interpreted broadly by the Supreme Court to include the right to live with dignity, right to privacy, right to clean environment, and many other rights essential for meaningful existence."}
            ]
        return []

def delete_session(session_id: str) -> bool:
    """Delete a session using existing sessions endpoint"""
    try:
        response = requests.delete(f"{BACKEND_URL}/sessions/{session_id}")
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def rename_session(session_id: str, new_title: str) -> bool:
    """Rename a session using the rename endpoint"""
    try:
        response = requests.patch(
            f"{BACKEND_URL}/sessions/{session_id}/rename",
            json={"title": new_title}
        )
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def login_page():
    """Login page UI"""
    st.markdown('<div class="main-header"><h1>🏛️ Constitutional AI Assistant</h1><p>Your intelligent legal research companion</p></div>', unsafe_allow_html=True)
    
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:            
            st.markdown("### 🔐 Login")
            st.markdown("---")
            
            # Backend status indicator
            backend_online = check_backend_status()
            if backend_online:
                st.success("🟢 Backend Connected")
            else:
                st.warning("🟡 Backend Offline - Demo Mode Available")
            
            username = st.text_input("👤 Username", placeholder="Enter your username")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
            
            if st.button("🚀 Login", use_container_width=True):
                if username and password:
                    if authenticate_user(username, password):
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.session_id = create_session(username)
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials")
                else:
                    st.error("⚠️ Please enter both username and password")
            
            st.markdown('</div>', unsafe_allow_html=True)

def chat_page():
    """Main chat interface"""
    
    # Header with session title and logout
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.markdown(f"👋 **{st.session_state.username}**")
        # Get current session title for display
        current_title = "Active Session"
        user_sessions = get_user_sessions_cached(st.session_state.username)
        if user_sessions:
            for s in user_sessions:
                if s.get('session_id') == st.session_state.session_id:
                    current_title = s.get('title', 'Untitled Session')
                    break
        st.caption(f"📝 {current_title}")
    with col2:
        st.markdown("### 🏛️ Constitutional AI Assistant")
    with col3:
        if st.button("🚀 Logout"):
            # Clear session state
            st.session_state.authenticated = False
            st.session_state.username = ""
            st.session_state.session_id = ""
            st.session_state.messages = []
            st.session_state.uploaded_files = []
            if 'current_flashcard_data' in st.session_state:
                del st.session_state.current_flashcard_data
            if 'current_quiz_data' in st.session_state:
                del st.session_state.current_quiz_data
            reset_flashcard_session()
            st.rerun()
    
    st.markdown("---")
    
    # Sidebar for file uploads and session management
    with st.sidebar:
        st.markdown("### 🗂️ Session Management")
        
        # Load existing sessions
        user_sessions = get_user_sessions_cached(st.session_state.username)
        
        if user_sessions:
            # Format sessions with better display
            session_options = {}
            for s in user_sessions:
                session_id = s.get('session_id', 'Unknown')
                title = s.get('title', 'Untitled Session')
                started_at = s.get('started_at', '')
                if started_at:
                    date_part = started_at[:10]  # Get YYYY-MM-DD
                    display_name = f"📝 {title} ({date_part})"
                else:
                    display_name = f"📝 {title}"
                session_options[display_name] = session_id
            
            selected_display = st.selectbox("💾 Existing Sessions", list(session_options.keys()) if session_options else ["No sessions found"])
            
            if selected_display in session_options and st.button("📂 Load Selected Session"):
                session_id = session_options[selected_display]
                
                # Load session messages
                with st.spinner("Loading session..."):
                    session_messages = get_session_messages(session_id)
                
                # Update session state
                st.session_state.session_id = session_id
                st.session_state.messages = session_messages
                
                st.success(f"✅ Loaded: {selected_display} ({len(session_messages)} messages)")
                st.rerun()
        
        # New session with custom title
        st.markdown("---")
        st.markdown("**🆕 Create New Session**")
        
        session_title = st.text_input(
            "Session Title",
            placeholder="e.g., Constitutional Rights Research",
            help="Give your session a descriptive title"
        )
        
        if st.button("🆕 New Session", use_container_width=True):
            title = session_title.strip() if session_title.strip() else f"Chat {datetime.now().strftime('%m/%d %H:%M')}"
            new_session_id = create_session_with_title(st.session_state.username, title)
            st.session_state.session_id = new_session_id
            st.session_state.messages = []
            # Clear sessions cache after creating new session
            get_user_sessions_cached.clear()
            st.success(f"✅ Created: {title}")
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📁 File Upload")
        
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=['pdf', 'docx', 'pptx', 'txt'],
            help="Upload constitutional documents, legal texts, case studies, etc."
        )
        
        st.markdown("---")
        st.markdown("### 🎬 Features Available")
        st.markdown("""
        **📝 Content Generation:**
        - Constitutional Q&A
        - Quiz generation
        - Document export (PDF/DOCX)
        
        **🎴 Interactive Learning:**
        - Flashcard generation
        - Study sessions with Q&A cards
        - Progress tracking
        
        **🎯 Interactive Quiz:**
        - Take quizzes with immediate feedback
        - Multiple question types (MCQ, T/F, Fill-blank)
        - Automatic scoring & explanations
        
        **🎥 Video Generation:**
        - Educational videos (2-2.5 min)
        - Constitutional topic explanations
        - Automatic narration & slides
        
        **💡 Example requests:**
        - "Create flashcards for Article 21"
        - "Take a quiz on fundamental rights"
        - "Quiz me about constitutional amendments"
        - "Generate a quiz on constitutional amendments"
        - "Create a video about the right to education"
        """)
        
        if uploaded_file is not None:
            if st.button("📤 Process Upload"):
                with st.spinner("Processing file..."):
                    result = upload_file(uploaded_file.read(), uploaded_file.name)
                    
                    if result:
                        st.session_state.uploaded_files.append({
                            "filename": uploaded_file.name,
                            "uploaded_at": datetime.now().isoformat(),
                            "document_id": result.get("document_id"),
                            "size_bytes": result.get("size_bytes", 0),
                            "extracted_text_length": result.get("extracted_text_length", 0)
                        })
                        
                        # Show upload success with extraction info
                        st.success(result.get("message", "File processed"))
                        
                        if result.get("extracted_text_length", 0) > 0:
                            st.info(f"📄 Extracted {result['extracted_text_length']:,} characters of text")
                        else:
                            st.warning("⚠️ No text could be extracted from this file")
                    else:
                        st.error("❌ Failed to upload file")
        
        # Session info
        st.markdown("---")
        st.markdown("### ℹ️ Current Session")
        
        # Try to find current session title
        current_session_title = "Unknown Session"
        if user_sessions:
            for s in user_sessions:
                if s.get('session_id') == st.session_state.session_id:
                    current_session_title = s.get('title', 'Untitled Session')
                    break
        
        st.markdown(f"**📝 Title:** {current_session_title}")
        st.markdown(f"**🔗 Session ID:** `{st.session_state.session_id}`")
        st.markdown(f"**💬 Messages:** {len(st.session_state.messages)}")
        
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            # Clear interactive components when clearing chat
            if 'current_flashcard_data' in st.session_state:
                del st.session_state.current_flashcard_data
            if 'current_quiz_data' in st.session_state:
                del st.session_state.current_quiz_data
            st.rerun()
        
        # Display active flashcard set info if available
        if 'current_flashcard_data' in st.session_state and st.session_state.current_flashcard_data:
            st.markdown("---")
            st.markdown("### 🎴 Active Flashcard Set")
            flashcard_data = st.session_state.current_flashcard_data
            st.markdown(f"**Topic:** {flashcard_data.get('topic', 'Unknown')}")
            
            cards = flashcard_data.get('flashcards', [])
            if cards:
                st.markdown(f"**Cards:** {len(cards)}")
                if 'current_flashcard_index' in st.session_state:
                    current_idx = st.session_state.current_flashcard_index
                    st.markdown(f"**Current:** {current_idx + 1}/{len(cards)}")
        
        # Display active quiz info if available
        if 'current_quiz_data' in st.session_state and st.session_state.current_quiz_data:
            st.markdown("---")
            st.markdown("### 🎯 Active Quiz")
            quiz_data = st.session_state.current_quiz_data
            st.markdown(f"**Topic:** {quiz_data.get('topic', 'Unknown')}")
            
            questions = quiz_data.get('questions', [])
            if questions:
                st.markdown(f"**Questions:** {len(questions)}")
                
                # Show quiz progress if started
                if 'quiz_started' in st.session_state and st.session_state.quiz_started:
                    answered = len(st.session_state.get('quiz_answers', {}))
                    st.markdown(f"**Progress:** {answered}/{len(questions)} answered")
                    
                    if 'current_question_index' in st.session_state:
                        current_q = st.session_state.current_question_index + 1
                        st.markdown(f"**Current:** Question {current_q}")
        
        # Session deletion
        st.markdown("---")
        st.markdown("**🗑️ Danger Zone**")
        
        if st.button("🗂️ Manage Sessions", use_container_width=True):
            if "show_session_manager" not in st.session_state:
                st.session_state.show_session_manager = False
            st.session_state.show_session_manager = not st.session_state.show_session_manager
            st.rerun()
        
        # Show session management interface if toggled
        if st.session_state.get("show_session_manager", False):
            st.markdown("**Select session to delete:**")
            
            if user_sessions:
                delete_options = {}
                for s in user_sessions:
                    session_id = s.get('session_id', 'Unknown')
                    title = s.get('title', 'Untitled Session')
                    # Don't show current session in delete list
                    if session_id != st.session_state.session_id:
                        delete_options[f"🗑️ {title}"] = session_id
                
                if delete_options:
                    selected_to_delete = st.selectbox(
                        "Sessions to delete",
                        list(delete_options.keys()),
                        help="Select a session to delete (current session is not shown)"
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("❌ Delete", use_container_width=True):
                            session_to_delete = delete_options[selected_to_delete]
                            if delete_session(session_to_delete):
                                # Clear the sessions cache after successful delete
                                get_user_sessions_cached.clear()
                                st.success(f"✅ Deleted: {selected_to_delete}")
                                st.rerun()
                            else:
                                st.error("❌ Failed to delete session")
                    
                    with col2:
                        if st.button("🔒 Cancel", use_container_width=True):
                            st.session_state.show_session_manager = False
                            st.rerun()
                else:
                    st.info("No other sessions to delete")
            else:
                st.info("No sessions found")
        
        # Add option to rename current session
        if not st.session_state.get("show_session_manager", False):
            st.markdown("---")
            st.markdown("**✏️ Rename Session**")
            new_title = st.text_input(
                "New title",
                value=current_session_title,
                placeholder="Enter new session title"
            )
            if st.button("💾 Save Title") and new_title.strip() != current_session_title:
                if new_title.strip():
                    if rename_session(st.session_state.session_id, new_title.strip()):
                        # Clear sessions cache to show updated name
                        get_user_sessions_cached.clear()
                        st.success(f"✅ Session renamed to: {new_title.strip()}")
                        st.rerun()
                    else:
                        st.error("❌ Failed to rename session")
                else:
                    st.error("⚠️ Session title cannot be empty")
    
    # Main chat area
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display chat messages
    if not st.session_state.messages:
        st.markdown("""
        ### 👋 How can I help you today?
        
        I can assist you with:
        - **Constitutional Law Research** 📚
        - **Legal Document Analysis** 🔍  
        - **Case Study Reviews** ⚖️
        - **Rights & Duties Explanation** 📋
        - **Legal Precedent Search** 🔎
        
        Upload documents and ask me anything about constitutional law!
        """)
    
    # Chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">👤 <strong>You:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message">🤖 <strong>Assistant:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display interactive components
    # Interactive flashcard display
    if 'current_flashcard_data' in st.session_state and st.session_state.current_flashcard_data:
        st.markdown("---")
        display_flashcards(st.session_state.current_flashcard_data)
    
    # Interactive quiz display
    if 'current_quiz_data' in st.session_state and st.session_state.current_quiz_data:
        st.markdown("---")
        display_interactive_quiz(st.session_state.current_quiz_data)
    
    # Chat input
    st.markdown("---")
    
    # Text input for messages
    user_message = st.chat_input("Ask me anything about constitutional law...")
    
    # Check if we're currently processing a message
    if st.session_state.get("processing_message", False):
        # Determine the type of processing
        message_to_process = st.session_state.get("message_to_process", "")
        quiz_keywords = ['quiz', 'test', 'questions', 'mcq', 'generate', 'create quiz', 'export', 'pdf']
        is_quiz_request = any(keyword in message_to_process.lower() for keyword in quiz_keywords)
        
        # Show appropriate processing state
        # if is_quiz_request:
        #     with st.spinner("🎯 Generating quiz... This may take a minute for document exports."):
        #         ai_response = send_message(message_to_process, st.session_state.messages, st.session_state.uploaded_files)
        # else:
        #     with st.spinner("🤔 Thinking..."):
        #         ai_response = send_message(message_to_process, st.session_state.messages, st.session_state.uploaded_files)
        with st.spinner("🤔 Thinking..."):
            ai_response = send_message(message_to_process, st.session_state.messages, st.session_state.uploaded_files)

        # Add AI response to session state
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        
        # Clear processing state
        st.session_state.processing_message = False
        st.session_state.message_to_process = ""
        
        # Rerun to update display with AI response
        st.rerun()
    
    if user_message:
        # Add user message immediately
        st.session_state.messages.append({"role": "user", "content": user_message})
        
        # Set processing state
        st.session_state.processing_message = True
        st.session_state.message_to_process = user_message
        
        # Rerun to show user message and start processing
        st.rerun()

# Main app logic
def main():
    if not st.session_state.authenticated:
        login_page()
    else:
        chat_page()

if __name__ == "__main__":
    main()
