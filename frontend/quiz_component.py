"""
Interactive Quiz Component for Streamlit
Provides quiz taking interface with immediate feedback
"""

import streamlit as st
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

def display_interactive_quiz(quiz_data: Dict[str, Any]):
    """
    Display interactive quiz interface in Streamlit
    
    Args:
        quiz_data: Dictionary containing quiz information
    """
    if not quiz_data or 'questions' not in quiz_data:
        st.error("No quiz data available")
        return
    
    questions = quiz_data['questions']
    if not questions:
        st.error("No questions found in quiz data")
        return
    
    # Initialize session state for quiz
    if 'quiz_started' not in st.session_state:
        st.session_state.quiz_started = False
    if 'current_question_index' not in st.session_state:
        st.session_state.current_question_index = 0
    if 'quiz_answers' not in st.session_state:
        st.session_state.quiz_answers = {}
    if 'quiz_completed' not in st.session_state:
        st.session_state.quiz_completed = False
    if 'quiz_score' not in st.session_state:
        st.session_state.quiz_score = 0
    if 'quiz_start_time' not in st.session_state:
        st.session_state.quiz_start_time = None
    
    # Quiz header
    st.markdown(f"### 🎯 {quiz_data.get('title', 'Constitutional Quiz')}")
    st.markdown(f"**{quiz_data.get('description', 'Test your constitutional knowledge')}**")
    
    # Quiz not started - show introduction
    if not st.session_state.quiz_started:
        _display_quiz_intro(quiz_data)
        return
    
    # Quiz completed - show results
    if st.session_state.quiz_completed:
        _display_quiz_results(quiz_data)
        return
    
    # Quiz in progress - show current question
    _display_current_question(quiz_data)

def _display_quiz_intro(quiz_data: Dict[str, Any]):
    """
    Display quiz introduction and start button
    """
    questions = quiz_data.get('questions', [])
    total_questions = len(questions)
    total_points = quiz_data.get('total_points', 0)
    difficulty = quiz_data.get('difficulty', 'medium')
    
    st.markdown("---")
    
    # Quiz info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📝 Questions", total_questions)
    with col2:
        st.metric("🎯 Total Points", total_points)
    with col3:
        st.metric("📈 Difficulty", difficulty.title())
    
    # Question type breakdown
    type_counts = {}
    for q in questions:
        q_type = q.get('type', 'unknown')
        type_counts[q_type] = type_counts.get(q_type, 0) + 1
    
    if type_counts:
        st.markdown("**Question Types:**")
        type_cols = st.columns(len(type_counts))
        for i, (q_type, count) in enumerate(type_counts.items()):
            type_name = {
                'mcq': '🅰️ Multiple Choice',
                'true_false': '✅ True/False',
                'fill_blank': '📝 Fill in Blank'
            }.get(q_type, q_type.title())
            with type_cols[i]:
                st.write(f"{type_name}: {count}")
    
    st.markdown("---")
    
    # Instructions
    st.markdown("""
    **📖 Instructions:**
    - Read each question carefully
    - Select your answer and click "Submit Answer"
    - You can navigate between questions using Previous/Next
    - Submit the entire quiz when you're done
    - You'll get immediate feedback and explanations
    """)
    
    # Start quiz button
    st.markdown("")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Start Quiz", type="primary", use_container_width=True):
            st.session_state.quiz_started = True
            st.session_state.quiz_start_time = datetime.now()
            st.session_state.current_question_index = 0
            st.session_state.quiz_answers = {}
            st.session_state.quiz_completed = False
            st.rerun()

def _display_current_question(quiz_data: Dict[str, Any]):
    """
    Display current question with answer interface
    """
    questions = quiz_data.get('questions', [])
    current_index = st.session_state.current_question_index
    
    if current_index >= len(questions):
        st.session_state.current_question_index = 0
        current_index = 0
    
    current_question = questions[current_index]
    total_questions = len(questions)
    
    # Progress indicator
    progress = (current_index + 1) / total_questions
    st.progress(progress)
    
    # Question header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**Question {current_index + 1} of {total_questions}**")
    with col2:
        st.markdown(f"**Points: {current_question.get('points', 1)}**")
    
    st.markdown("---")
    
    # Question content
    st.markdown(f"### {current_question.get('question', 'Sample question')}")
    
    question_type = current_question.get('type', 'mcq')
    question_id = current_question.get('id', current_index + 1)
    
    # Answer interface based on question type
    user_answer = None
    
    if question_type == 'mcq':
        options = current_question.get('options', [])
        if options:
            # Get previously selected answer
            previous_answer = st.session_state.quiz_answers.get(question_id)
            default_index = previous_answer if previous_answer is not None else 0
            
            user_answer = st.radio(
                "Choose your answer:",
                options=list(range(len(options))),
                format_func=lambda x: f"{chr(65+x)}. {options[x]}",
                index=default_index,
                key=f"q_{question_id}_mcq"
            )
    
    elif question_type == 'true_false':
        previous_answer = st.session_state.quiz_answers.get(question_id)
        default_value = previous_answer if previous_answer is not None else True
        
        user_answer = st.radio(
            "Choose your answer:",
            options=[True, False],
            format_func=lambda x: "True" if x else "False",
            index=0 if default_value else 1,
            key=f"q_{question_id}_tf"
        )
    
    elif question_type == 'fill_blank':
        previous_answer = st.session_state.quiz_answers.get(question_id, "")
        
        user_answer = st.text_input(
            "Your answer:",
            value=previous_answer,
            placeholder="Type your answer here...",
            key=f"q_{question_id}_fill"
        )
    
    # Submit answer button
    st.markdown("")
    if st.button("✅ Submit Answer", type="primary"):
        if user_answer is not None and str(user_answer).strip():
            st.session_state.quiz_answers[question_id] = user_answer
            st.success("✅ Answer submitted!")
        else:
            st.warning("⚠️ Please select an answer before submitting.")
    
    # Navigation buttons
    st.markdown("---")
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 1, 1])
    
    with nav_col1:
        if st.button("◀️ Previous", disabled=(current_index == 0)):
            st.session_state.current_question_index = max(0, current_index - 1)
            st.rerun()
    
    with nav_col2:
        if st.button("▶️ Next", disabled=(current_index == total_questions - 1)):
            st.session_state.current_question_index = min(total_questions - 1, current_index + 1)
            st.rerun()
    
    with nav_col3:
        # Show progress
        answered = len(st.session_state.quiz_answers)
        st.write(f"📈 {answered}/{total_questions} answered")
    
    with nav_col4:
        if st.button("🏁 Finish Quiz", type="secondary"):
            if len(st.session_state.quiz_answers) == 0:
                st.warning("⚠️ Please answer at least one question before finishing.")
            else:
                # Calculate score
                _calculate_quiz_score(quiz_data)
                st.session_state.quiz_completed = True
                st.rerun()

def _calculate_quiz_score(quiz_data: Dict[str, Any]):
    """
    Calculate quiz score based on answers
    """
    questions = quiz_data.get('questions', [])
    user_answers = st.session_state.quiz_answers
    
    total_score = 0
    max_score = 0
    correct_answers = 0
    
    for question in questions:
        question_id = question.get('id')
        points = question.get('points', 1)
        max_score += points
        
        if question_id not in user_answers:
            continue
        
        user_answer = user_answers[question_id]
        question_type = question.get('type', 'mcq')
        
        is_correct = False
        
        if question_type == 'mcq':
            correct_index = question.get('correct_answer', 0)
            is_correct = user_answer == correct_index
        
        elif question_type == 'true_false':
            correct_answer = question.get('correct_answer', True)
            is_correct = user_answer == correct_answer
        
        elif question_type == 'fill_blank':
            correct_answer = question.get('correct_answer', '').lower().strip()
            alternative_answers = [ans.lower().strip() for ans in question.get('alternative_answers', [])]
            user_answer_clean = str(user_answer).lower().strip()
            
            is_correct = (user_answer_clean == correct_answer or 
                         user_answer_clean in alternative_answers)
        
        if is_correct:
            total_score += points
            correct_answers += 1
    
    st.session_state.quiz_score = {
        'total_score': total_score,
        'max_score': max_score,
        'correct_answers': correct_answers,
        'total_questions': len(questions),
        'percentage': (total_score / max_score * 100) if max_score > 0 else 0
    }

def _display_quiz_results(quiz_data: Dict[str, Any]):
    """
    Display quiz results and explanations
    """
    score_data = st.session_state.quiz_score
    questions = quiz_data.get('questions', [])
    user_answers = st.session_state.quiz_answers
    
    # Results header
    st.markdown("### 🏆 Quiz Results")
    
    # Score display
    percentage = score_data['percentage']
    grade = _get_grade(percentage)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Score", f"{score_data['total_score']}/{score_data['max_score']}")
    with col2:
        st.metric("📊 Percentage", f"{percentage:.1f}%")
    with col3:
        st.metric("✅ Correct", f"{score_data['correct_answers']}/{score_data['total_questions']}")
    with col4:
        st.metric("🏅 Grade", grade)
    
    # Time taken
    if st.session_state.quiz_start_time:
        time_taken = datetime.now() - st.session_state.quiz_start_time
        minutes = int(time_taken.total_seconds() // 60)
        seconds = int(time_taken.total_seconds() % 60)
        st.markdown(f"**⏱️ Time taken:** {minutes}m {seconds}s")
    
    st.markdown("---")
    
    # Grade message
    grade_messages = {
        'A+': '🎆 Outstanding! Excellent constitutional knowledge!',
        'A': '🎉 Very good understanding of constitutional concepts!',
        'B+': '🙌 Good grasp of constitutional principles!',
        'B': '👍 Satisfactory knowledge, keep learning!',
        'C+': '📚 Basic understanding, focus on weak areas!',
        'C': '📝 Below average, significant improvement needed!',
        'D': '🔄 Needs focused study on constitutional topics!'
    }
    
    st.success(grade_messages.get(grade, 'Quiz completed!'))
    
    # Question-by-question review
    st.markdown("### 📝 Question Review")
    
    for i, question in enumerate(questions):
        question_id = question.get('id')
        user_answer = user_answers.get(question_id)
        question_type = question.get('type', 'mcq')
        
        # Determine if answer was correct
        is_correct = False
        correct_display = ""
        user_display = ""
        
        if question_type == 'mcq':
            correct_index = question.get('correct_answer', 0)
            options = question.get('options', [])
            
            if user_answer is not None:
                is_correct = user_answer == correct_index
                user_display = options[user_answer] if user_answer < len(options) else "Invalid"
            
            correct_display = options[correct_index] if correct_index < len(options) else "Unknown"
        
        elif question_type == 'true_false':
            correct_answer = question.get('correct_answer', True)
            is_correct = user_answer == correct_answer
            user_display = "True" if user_answer else "False"
            correct_display = "True" if correct_answer else "False"
        
        elif question_type == 'fill_blank':
            correct_answer = question.get('correct_answer', '')
            alternative_answers = question.get('alternative_answers', [])
            
            if user_answer:
                user_answer_clean = str(user_answer).lower().strip()
                correct_answer_clean = correct_answer.lower().strip()
                alternative_answers_clean = [ans.lower().strip() for ans in alternative_answers]
                
                is_correct = (user_answer_clean == correct_answer_clean or 
                             user_answer_clean in alternative_answers_clean)
            
            user_display = str(user_answer) if user_answer else "No answer"
            correct_display = correct_answer
        
        # Display question result
        with st.expander(f"{'✅' if is_correct else '❌'} Question {i+1} - {question.get('points', 1)} point(s)"):
            st.markdown(f"**Q: {question.get('question', 'Unknown question')}**")
            
            if user_answer is not None:
                if is_correct:
                    st.success(f"✅ **Your answer:** {user_display}")
                else:
                    st.error(f"❌ **Your answer:** {user_display}")
                    st.info(f"✅ **Correct answer:** {correct_display}")
            else:
                st.warning("⚠️ No answer provided")
                st.info(f"✅ **Correct answer:** {correct_display}")
            
            # Explanation
            explanation = question.get('explanation', 'No explanation provided')
            article_ref = question.get('article_reference', '')
            
            st.markdown(f"**📝 Explanation:** {explanation}")
            if article_ref and article_ref != 'General':
                st.markdown(f"**📖 Reference:** {article_ref}")
    
    # Action buttons
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Retake Quiz", use_container_width=True):
            _reset_quiz_session()
            st.rerun()
    
    with col2:
        if st.button("❌ Close Quiz", use_container_width=True):
            _reset_quiz_session()
            if 'current_quiz_data' in st.session_state:
                del st.session_state.current_quiz_data
            st.rerun()

def _get_grade(percentage: float) -> str:
    """
    Convert percentage to letter grade
    """
    if percentage >= 95: return 'A+'
    elif percentage >= 85: return 'A'
    elif percentage >= 75: return 'B+'
    elif percentage >= 65: return 'B'
    elif percentage >= 55: return 'C+'
    elif percentage >= 45: return 'C'
    else: return 'D'

def parse_quiz_data_from_response(response_text: str) -> Dict[str, Any]:
    """
    Extract quiz data from LLM response text
    
    Args:
        response_text: The response text containing JSON quiz data
    
    Returns:
        Parsed quiz data dictionary
    """
    try:
        # Look for JSON in the response
        if '```json' in response_text:
            start = response_text.find('```json') + 7
            end = response_text.find('```', start)
            json_str = response_text[start:end].strip()
        elif '{' in response_text and '}' in response_text:
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            json_str = response_text[start:end]
        else:
            return None
        
        quiz_data = json.loads(json_str)
        return quiz_data
        
    except Exception as e:
        st.error(f"Error parsing quiz data: {e}")
        return None

def _reset_quiz_session():
    """
    Reset all quiz session state
    """
    keys_to_reset = [
        'quiz_started', 'current_question_index', 'quiz_answers',
        'quiz_completed', 'quiz_score', 'quiz_start_time'
    ]
    
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]