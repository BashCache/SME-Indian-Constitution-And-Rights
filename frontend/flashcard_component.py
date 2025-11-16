"""
Interactive Flashcard Component for Streamlit
Provides basic flashcard viewing and interaction
"""

import streamlit as st
import json
from typing import Dict, List, Any

def display_flashcards(flashcard_data: Dict[str, Any]):
    """
    Display interactive flashcards in Streamlit
    
    Args:
        flashcard_data: Dictionary containing flashcard information
    """
    if not flashcard_data or 'cards' not in flashcard_data:
        st.error("No flashcard data available")
        return
    
    cards = flashcard_data['cards']
    if not cards:
        st.error("No cards found in flashcard data")
        return
    
    # Initialize session state for flashcard navigation
    if 'current_card_index' not in st.session_state:
        st.session_state.current_card_index = 0
    if 'show_answer' not in st.session_state:
        st.session_state.show_answer = False
    if 'flashcard_progress' not in st.session_state:
        st.session_state.flashcard_progress = []
    
    # Ensure current index is valid
    if st.session_state.current_card_index >= len(cards):
        st.session_state.current_card_index = 0
    
    current_card = cards[st.session_state.current_card_index]
    total_cards = len(cards)
    
    # Flashcard header
    st.markdown(f"### 🎴 {flashcard_data.get('topic', 'Constitutional Law')} Flashcards")
    
    # Progress indicator
    progress = (st.session_state.current_card_index + 1) / total_cards
    st.progress(progress)
    st.markdown(f"**Card {st.session_state.current_card_index + 1} of {total_cards}**")
    
    # Card difficulty and category
    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"📊 Difficulty: {flashcard_data.get('difficulty', 'medium').title()}")
    with col2:
        st.caption(f"📝 Category: {current_card.get('category', 'general').title()}")
    
    # Main card display
    st.markdown("---")
    
    # Create card container with styling
    with st.container():
        # Apply custom CSS for card styling
        st.markdown("""
        <style>
        .flashcard-container {
            background-color: #f8f9fa;
            border: 2px solid #dee2e6;
            border-radius: 15px;
            padding: 30px;
            margin: 20px 0;
            text-align: center;
            min-height: 250px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .flashcard-question {
            font-size: 1.4em;
            font-weight: bold;
            color: #495057;
            margin-bottom: 20px;
        }
        .flashcard-answer {
            font-size: 1.1em;
            color: #6c757d;
            line-height: 1.6;
        }
        .flashcard-reference {
            font-style: italic;
            color: #28a745;
            margin-top: 15px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Question side
        if not st.session_state.show_answer:
            st.markdown(f"""
            <div class="flashcard-container">
                <div>
                    <div class="flashcard-question">{current_card.get('question', 'Sample question')}</div>
                    <p style="color: #6c757d; margin-top: 20px;">🤔 Think about your answer, then click "Show Answer" below</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Answer side
        else:
            reference = current_card.get('article_reference', 'Reference')
            st.markdown(f"""
            <div class="flashcard-container">
                <div>
                    <div class="flashcard-answer">{current_card.get('answer', 'Sample answer')}</div>
                    <div class="flashcard-reference">📖 Reference: {reference}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Card interaction buttons
    st.markdown("---")
    
    # Center the flip button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if not st.session_state.show_answer:
            if st.button("🔄 Show Answer", use_container_width=True, type="primary"):
                st.session_state.show_answer = True
                st.rerun()
        else:
            if st.button("🔄 Show Question", use_container_width=True, type="secondary"):
                st.session_state.show_answer = False
                st.rerun()
    
    # Navigation buttons
    st.markdown("")
    nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([1, 1, 1, 1, 1])
    
    with nav_col1:
        if st.button("⏮️ First", disabled=(st.session_state.current_card_index == 0)):
            st.session_state.current_card_index = 0
            st.session_state.show_answer = False
            st.rerun()
    
    with nav_col2:
        if st.button("◀️ Previous", disabled=(st.session_state.current_card_index == 0)):
            st.session_state.current_card_index -= 1
            st.session_state.show_answer = False
            st.rerun()
    
    with nav_col3:
        # Study progress button
        if st.session_state.show_answer:
            if st.button("✅ Got it!", type="primary"):
                # Mark as known and go to next
                if st.session_state.current_card_index not in st.session_state.flashcard_progress:
                    st.session_state.flashcard_progress.append(st.session_state.current_card_index)
                
                if st.session_state.current_card_index < total_cards - 1:
                    st.session_state.current_card_index += 1
                    st.session_state.show_answer = False
                st.rerun()
    
    with nav_col4:
        if st.button("▶️ Next", disabled=(st.session_state.current_card_index == total_cards - 1)):
            st.session_state.current_card_index += 1
            st.session_state.show_answer = False
            st.rerun()
    
    with nav_col5:
        if st.button("⏭️ Last", disabled=(st.session_state.current_card_index == total_cards - 1)):
            st.session_state.current_card_index = total_cards - 1
            st.session_state.show_answer = False
            st.rerun()
    
    # Study progress
    if st.session_state.flashcard_progress:
        progress_percent = len(st.session_state.flashcard_progress) / total_cards * 100
        st.markdown("---")
        st.markdown(f"📈 **Study Progress:** {len(st.session_state.flashcard_progress)}/{total_cards} cards completed ({progress_percent:.1f}%)")
        
        if len(st.session_state.flashcard_progress) == total_cards:
            st.success("🎉 Congratulations! You've completed all flashcards!")
            if st.button("🔄 Reset Progress"):
                st.session_state.flashcard_progress = []
                st.session_state.current_card_index = 0
                st.session_state.show_answer = False
                st.rerun()

def parse_flashcard_data_from_response(response_text: str) -> Dict[str, Any]:
    """
    Extract flashcard data from LLM response text
    
    Args:
        response_text: The response text containing JSON flashcard data
    
    Returns:
        Parsed flashcard data dictionary
    """
    try:
        # Look for JSON in the response
        if '```json' in response_text:
            # Extract JSON from code block
            start = response_text.find('```json') + 7
            end = response_text.find('```', start)
            json_str = response_text[start:end].strip()
        elif '{' in response_text and '}' in response_text:
            # Find JSON object in response
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            json_str = response_text[start:end]
        else:
            return None
        
        # Parse JSON
        flashcard_data = json.loads(json_str)
        return flashcard_data
        
    except Exception as e:
        st.error(f"Error parsing flashcard data: {e}")
        return None

def reset_flashcard_session():
    """
    Reset all flashcard session state
    """
    keys_to_reset = ['current_card_index', 'show_answer', 'flashcard_progress']
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]