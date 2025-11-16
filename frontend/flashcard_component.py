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
    st.markdown(f"**Progress: {st.session_state.current_card_index + 1} of {total_cards}**")
    
    # Main card display
    st.markdown("")
    
    # Create card container with styling
    with st.container():
        # Apply custom CSS for enhanced card styling
        st.markdown("""
        <style>
        .flashcard-main-container {
            perspective: 1000px;
            margin: 30px auto;
            max-width: 600px;
        }
        .flashcard-container {
            background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
            border: none;
            border-radius: 20px;
            padding: 40px;
            margin: 20px 0;
            text-align: center;
            min-height: 320px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            box-shadow: 
                0 10px 30px rgba(0, 0, 0, 0.15),
                0 6px 20px rgba(0, 0, 0, 0.1),
                inset 0 1px 0 rgba(255, 255, 255, 0.8);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            transform: translateZ(0);
            position: relative;
            overflow: hidden;
        }
        .flashcard-container:before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 20px 20px 0 0;
        }
        .flashcard-container:hover {
            transform: translateY(-5px) translateZ(0);
            box-shadow: 
                0 15px 40px rgba(0, 0, 0, 0.2),
                0 10px 25px rgba(0, 0, 0, 0.15),
                inset 0 1px 0 rgba(255, 255, 255, 0.9);
        }
        .flashcard-question {
            font-size: 1.6em;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 25px;
            line-height: 1.4;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
        }
        .flashcard-answer {
            font-size: 1.2em;
            color: #34495e;
            line-height: 1.7;
            text-align: justify;
            margin-bottom: 20px;
        }
        .flashcard-reference {
            font-style: italic;
            font-weight: 500;
            color: #27ae60;
            margin-top: 20px;
            padding: 8px 16px;
            background: rgba(39, 174, 96, 0.1);
            border-radius: 20px;
            border: 1px solid rgba(39, 174, 96, 0.2);
        }
        .flashcard-hint {
            color: #7f8c8d;
            font-style: italic;
            margin-top: 25px;
            opacity: 0.8;
        }
        .flashcard-card-indicator {
            position: absolute;
            top: 15px;
            right: 20px;
            background: rgba(102, 126, 234, 0.1);
            color: #667eea;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.9em;
            font-weight: 500;
        }
        .flashcard-difficulty-badge {
            position: absolute;
            top: 15px;
            left: 20px;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.85em;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .difficulty-easy {
            background: rgba(46, 204, 113, 0.1);
            color: #2ecc71;
            border: 1px solid rgba(46, 204, 113, 0.2);
        }
        .difficulty-medium {
            background: rgba(241, 196, 15, 0.1);
            color: #f1c40f;
            border: 1px solid rgba(241, 196, 15, 0.2);
        }
        .difficulty-hard {\n            background: rgba(231, 76, 60, 0.1);\n            color: #e74c3c;\n            border: 1px solid rgba(231, 76, 60, 0.2);\n        }\n        \n        /* Enhanced button styling */\n        .stButton > button {\n            border-radius: 12px;\n            border: none;\n            font-weight: 500;\n            transition: all 0.3s ease;\n            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);\n        }\n        .stButton > button:hover {\n            transform: translateY(-2px);\n            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);\n        }\n        </style>
        """, unsafe_allow_html=True)
        
        # Question side
        if not st.session_state.show_answer:
            difficulty = flashcard_data.get('difficulty', 'medium').lower()
            st.markdown(f"""
            <div class="flashcard-main-container">
                <div class="flashcard-container">
                    <div class="flashcard-difficulty-badge difficulty-{difficulty}">
                        {flashcard_data.get('difficulty', 'medium').title()}
                    </div>
                    <div class="flashcard-card-indicator">
                        Card {st.session_state.current_card_index + 1} of {total_cards}
                    </div>
                    <div class="flashcard-question">{current_card.get('question', 'Sample question')}</div>
                    <div class="flashcard-hint">🤔 Think about your answer, then click "Show Answer" below</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Answer side
        else:
            reference = current_card.get('article_reference', 'Reference')
            difficulty = flashcard_data.get('difficulty', 'medium').lower()
            st.markdown(f"""
            <div class="flashcard-main-container">
                <div class="flashcard-container">
                    <div class="flashcard-difficulty-badge difficulty-{difficulty}">
                        {flashcard_data.get('difficulty', 'medium').title()}
                    </div>
                    <div class="flashcard-card-indicator">
                        Card {st.session_state.current_card_index + 1} of {total_cards}
                    </div>
                    <div class="flashcard-answer">{current_card.get('answer', 'Sample answer')}</div>
                    <div class="flashcard-reference">📖 {reference}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Card interaction buttons
    st.markdown("")
    
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