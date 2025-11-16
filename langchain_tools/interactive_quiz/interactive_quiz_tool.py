"""
Interactive Quiz Tool for LangChain Integration
Generates interactive quizzes about constitutional topics with immediate feedback
"""

import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

class InteractiveQuizInput(BaseModel):
    """Input schema for Interactive Quiz Tool"""
    topic: str = Field(..., description="The constitutional topic to create quiz about")
    num_questions: Optional[int] = Field(default=10, description="Number of quiz questions (default: 10)")
    difficulty: Optional[str] = Field(default="medium", description="Difficulty level: easy, medium, or hard")
    question_types: Optional[str] = Field(default="mixed", description="Question types: mcq, true_false, fill_blank, or mixed")

class InteractiveQuizTool:
    """
    A tool for generating interactive quizzes about constitutional topics.
    
    Features:
    - LLM-powered question generation
    - Multiple question types (MCQ, True/False, Fill blanks)
    - Interactive answer selection
    - Immediate scoring and feedback
    - Constitutional law focus
    """
    
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        # Initialize LLM for quiz generation
        if self.gemini_api_key:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",
                google_api_key=self.gemini_api_key,
                temperature=0.2  # Lower temperature for consistent questions
            )
        else:
            self.llm = None
            print("Warning: GEMINI_API_KEY not found. Quiz generation will be limited.")
    
    def generate_interactive_quiz(self, topic: str, num_questions: int = 10, difficulty: str = "medium", question_types: str = "mixed") -> Dict[str, Any]:
        """
        Main method to generate an interactive quiz
        
        Args:
            topic: Constitutional topic to cover
            num_questions: Number of questions to generate
            difficulty: Difficulty level (easy, medium, hard)
            question_types: Types of questions (mcq, true_false, fill_blank, mixed)
        
        Returns:
            Dictionary with quiz data
        """
        try:
            print(f"🎯 Starting interactive quiz generation for: {topic}")
            start_time = datetime.now()
            
            # Generate quiz content
            if self.llm:
                quiz_data = self._generate_quiz_with_llm(topic, num_questions, difficulty, question_types)
            else:
                quiz_data = self._generate_fallback_quiz(topic, num_questions, difficulty)
            
            if not quiz_data or not quiz_data.get('questions'):
                return {
                    'success': False,
                    'error': 'Failed to generate quiz content',
                    'topic': topic
                }
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            print(f"✅ Interactive quiz generation completed in {processing_time:.1f} seconds")
            
            return {
                'success': True,
                'quiz_data': quiz_data,
                'topic': topic,
                'processing_time': processing_time,
                'created_at': end_time.isoformat()
            }
            
        except Exception as e:
            print(f"❌ Error in quiz generation: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'topic': topic
            }
    
    def _generate_quiz_with_llm(self, topic: str, num_questions: int, difficulty: str, question_types: str) -> Dict[str, Any]:
        """
        Generate quiz using LLM
        """
        try:
            prompt = f"""
Create an interactive quiz with {num_questions} questions about "{topic}" related to the Indian Constitution.

Requirements:
- Difficulty level: {difficulty}
- Question types: {question_types}
- Each question should test constitutional knowledge
- Provide clear, unambiguous questions
- Include detailed explanations for correct answers
- Reference relevant constitutional articles where applicable

Structure the response as a JSON object with this exact format:
{{
    "quiz_id": "quiz_{uuid.uuid4().hex[:8]}",
    "title": "Quiz on {topic}",
    "description": "Test your knowledge of {topic}",
    "topic": "{topic}",
    "total_questions": {num_questions},
    "difficulty": "{difficulty}",
    "total_points": 0,
    "questions": [
        {{
            "id": 1,
            "type": "mcq",
            "question": "What does Article 21 of the Indian Constitution guarantee?",
            "options": [
                "Right to equality",
                "Right to life and personal liberty",
                "Right to freedom of speech", 
                "Right to constitutional remedies"
            ],
            "correct_answer": 1,
            "explanation": "Article 21 guarantees the right to life and personal liberty. It states that no person shall be deprived of his life or personal liberty except according to procedure established by law.",
            "points": 2,
            "article_reference": "Article 21"
        }},
        {{
            "id": 2,
            "type": "true_false",
            "question": "Article 14 applies only to Indian citizens.",
            "correct_answer": false,
            "explanation": "Article 14 applies to all persons, not just citizens. It guarantees equality before law and equal protection of laws to every person within the territory of India.",
            "points": 1,
            "article_reference": "Article 14"
        }},
        {{
            "id": 3,
            "type": "fill_blank",
            "question": "The right to constitutional remedies is guaranteed under Article ____.",
            "correct_answer": "32",
            "alternative_answers": ["thirty-two", "thirty two"],
            "explanation": "Article 32 is known as the right to constitutional remedies. Dr. B.R. Ambedkar called it 'the heart and soul' of the Constitution.",
            "points": 1,
            "article_reference": "Article 32"
        }}
        // ... continue for all {num_questions} questions
    ]
}}

Guidelines for question types:
- "mcq": Multiple choice with 4 options (index 0-3), specify correct_answer as index number
- "true_false": True/False questions, specify correct_answer as boolean
- "fill_blank": Fill in the blank, provide correct_answer and alternative_answers array
- "mixed": Include variety of all question types

Guidelines for difficulty levels:
- "easy": Basic constitutional concepts, well-known articles and rights
- "medium": Detailed provisions, case law applications, lesser-known articles
- "hard": Complex interpretations, detailed case analysis, constitutional history

Ensure:
1. Questions test real understanding, not just memorization
2. Explanations are educational and comprehensive
3. Include relevant constitutional article numbers
4. Points: Easy=1, Medium=2, Hard=3 points per question
5. Cover diverse aspects of the topic
6. Language is clear for students and legal learners

Calculate total_points as sum of all question points.

Respond with ONLY the JSON object, no additional text."""
            
            response = self.llm.invoke(prompt)
            quiz_text = response.content if hasattr(response, 'content') else str(response)
            
            # Clean and parse JSON
            quiz_text = quiz_text.strip()
            if quiz_text.startswith('```json'):
                quiz_text = quiz_text[7:-3].strip()
            elif quiz_text.startswith('```'):
                quiz_text = quiz_text[3:-3].strip()
            
            quiz_data = json.loads(quiz_text)
            
            # Validate and fix quiz data
            quiz_data = self._validate_quiz_data(quiz_data, num_questions)
            
            print(f"✅ Generated {len(quiz_data.get('questions', []))} quiz questions")
            return quiz_data
            
        except Exception as e:
            print(f"Error generating quiz with LLM: {e}")
            return self._generate_fallback_quiz(topic, num_questions, difficulty)
    
    def _generate_fallback_quiz(self, topic: str, num_questions: int, difficulty: str) -> Dict[str, Any]:
        """
        Generate basic fallback quiz when LLM is not available
        """
        fallback_questions = []
        
        # Generate basic questions
        for i in range(min(num_questions, 5)):  # Limit fallback to 5 questions
            if i % 3 == 0:  # MCQ
                fallback_questions.append({
                    "id": i + 1,
                    "type": "mcq",
                    "question": f"What is an important aspect of {topic}?",
                    "options": [
                        f"Basic concept of {topic}",
                        f"Advanced principle of {topic}",
                        f"Constitutional provision about {topic}",
                        f"Legal interpretation of {topic}"
                    ],
                    "correct_answer": 2,
                    "explanation": f"This question tests understanding of {topic} in constitutional context.",
                    "points": 2,
                    "article_reference": "General"
                })
            elif i % 3 == 1:  # True/False
                fallback_questions.append({
                    "id": i + 1,
                    "type": "true_false",
                    "question": f"{topic} is mentioned in the Indian Constitution.",
                    "correct_answer": True,
                    "explanation": f"Yes, {topic} is an important constitutional concept.",
                    "points": 1,
                    "article_reference": "General"
                })
            else:  # Fill blank
                fallback_questions.append({
                    "id": i + 1,
                    "type": "fill_blank",
                    "question": f"The concept of {topic} is important for ____.",
                    "correct_answer": "constitutional law",
                    "alternative_answers": ["constitution", "legal system"],
                    "explanation": f"{topic} plays a crucial role in constitutional framework.",
                    "points": 1,
                    "article_reference": "General"
                })
        
        total_points = sum(q['points'] for q in fallback_questions)
        
        return {
            "quiz_id": f"fallback_{uuid.uuid4().hex[:8]}",
            "title": f"Sample Quiz on {topic}",
            "description": f"Basic quiz about {topic} (configure GEMINI_API_KEY for AI-generated content)",
            "topic": topic,
            "total_questions": len(fallback_questions),
            "difficulty": difficulty,
            "total_points": total_points,
            "questions": fallback_questions
        }
    
    def _validate_quiz_data(self, quiz_data: Dict[str, Any], expected_questions: int) -> Dict[str, Any]:
        """
        Validate and fix quiz data
        """
        questions = quiz_data.get('questions', [])
        
        # Ensure all questions have required fields
        total_points = 0
        for i, question in enumerate(questions):
            if 'id' not in question:
                question['id'] = i + 1
            if 'type' not in question:
                question['type'] = 'mcq'
            if 'question' not in question:
                question['question'] = f"Sample question {i + 1}"
            if 'points' not in question:
                question['points'] = 1
            if 'explanation' not in question:
                question['explanation'] = "Sample explanation"
            if 'article_reference' not in question:
                question['article_reference'] = "General"
            
            # Validate based on question type
            if question['type'] == 'mcq':
                if 'options' not in question or len(question['options']) < 2:
                    question['options'] = ["Option A", "Option B", "Option C", "Option D"]
                if 'correct_answer' not in question:
                    question['correct_answer'] = 0
            elif question['type'] == 'true_false':
                if 'correct_answer' not in question:
                    question['correct_answer'] = True
            elif question['type'] == 'fill_blank':
                if 'correct_answer' not in question:
                    question['correct_answer'] = "sample answer"
                if 'alternative_answers' not in question:
                    question['alternative_answers'] = []
            
            total_points += question['points']
        
        quiz_data['questions'] = questions
        quiz_data['total_questions'] = len(questions)
        quiz_data['total_points'] = total_points
        
        # Set quiz_id if missing
        if 'quiz_id' not in quiz_data:
            quiz_data['quiz_id'] = f"quiz_{uuid.uuid4().hex[:8]}"
        
        return quiz_data

# Create the tool instance
interactive_quiz_tool_instance = InteractiveQuizTool()

@tool("interactive_quiz_tool", args_schema=InteractiveQuizInput, return_direct=True)
def interactive_quiz_tool(topic: str, num_questions: int = 10, difficulty: str = "medium", question_types: str = "mixed") -> str:
    """
    Generate an interactive quiz about constitutional topics.
    
    Creates educational quizzes with multiple question types, immediate feedback,
    and scoring. Perfect for testing knowledge of constitutional law, rights, and legal principles.
    
    Args:
        topic: The constitutional topic to create quiz about
        num_questions: Number of questions to generate (default: 10)
        difficulty: Difficulty level - easy, medium, or hard
        question_types: Question types - mcq, true_false, fill_blank, or mixed
    
    Returns:
        JSON string with quiz data for interactive display
    """
    result = interactive_quiz_tool_instance.generate_interactive_quiz(
        topic=topic,
        num_questions=num_questions,
        difficulty=difficulty,
        question_types=question_types
    )
    
    if result['success']:
        quiz_data = result.get('quiz_data', {})
        questions = quiz_data.get('questions', [])
        
        response = f"""🎯 **Interactive Quiz Generated Successfully!**

📚 **Quiz Details:**
• **Topic:** {result['topic']}
• **Questions:** {len(questions)}
• **Difficulty:** {quiz_data.get('difficulty', difficulty).title()}
• **Total Points:** {quiz_data.get('total_points', 0)}
• **Quiz ID:** {quiz_data.get('quiz_id', 'unknown')}

⏱️ **Processing Time:** {result.get('processing_time', 0):.1f} seconds

🎯 **Question Types:**"""
        
        # Count question types
        type_counts = {}
        for q in questions:
            q_type = q.get('type', 'unknown')
            type_counts[q_type] = type_counts.get(q_type, 0) + 1
        
        for q_type, count in type_counts.items():
            type_name = {
                'mcq': 'Multiple Choice',
                'true_false': 'True/False', 
                'fill_blank': 'Fill in the Blank'
            }.get(q_type, q_type.title())
            response += f"\n• {type_name}: {count} questions"
        
        # Show first question as preview
        if questions:
            first_q = questions[0]
            response += f"\n\n📝 **Sample Question:**\n**{first_q.get('question', 'Sample question')}**"
            
            if first_q.get('type') == 'mcq' and first_q.get('options'):
                for i, option in enumerate(first_q['options'][:2]):  # Show first 2 options
                    response += f"\n{chr(65+i)}. {option}"
                if len(first_q['options']) > 2:
                    response += f"\n... and {len(first_q['options'])-2} more options"
        
        response += f"\n\n✅ **Your interactive quiz about '{result['topic']}' is ready to take!**"
        response += f"\n\n🎯 **Quiz Data:**\n```json\n{json.dumps(quiz_data, indent=2)}\n```"
        
    else:
        response = f"❌ **Quiz Generation Failed**\n\n**Topic:** {result['topic']}\n**Error:** {result.get('error', 'Unknown error')}\n\nPlease try again or check the logs for more details."
    
    return response