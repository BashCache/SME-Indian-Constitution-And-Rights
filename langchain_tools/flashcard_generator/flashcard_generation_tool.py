"""
Flashcard Generation Tool for LangChain Integration
Generates interactive flashcards about constitutional topics
"""

import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

class FlashcardGenerationInput(BaseModel):
    """Input schema for Flashcard Generation Tool"""
    topic: str = Field(..., description="The constitutional topic to create flashcards about")
    num_cards: Optional[int] = Field(default=10, description="Number of flashcards to generate (default: 10)")
    difficulty: Optional[str] = Field(default="medium", description="Difficulty level: easy, medium, or hard")
    card_type: Optional[str] = Field(default="mixed", description="Type: definitions, articles, cases, or mixed")

class FlashcardGenerationTool:
    """
    A tool for generating educational flashcards about constitutional topics.
    
    Features:
    - LLM-powered Q&A generation
    - Constitutional law focus
    - Interactive card data structure
    - JSON output for frontend consumption
    """
    
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        # Initialize LLM for flashcard generation
        if self.gemini_api_key:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",
                google_api_key=self.gemini_api_key,
                temperature=0.2  # Lower temperature for more consistent Q&A pairs
            )
        else:
            self.llm = None
            print("Warning: GEMINI_API_KEY not found. Flashcard generation will be limited.")
    
    def generate_flashcards(self, topic: str, num_cards: int = 10, difficulty: str = "medium", card_type: str = "mixed") -> Dict[str, Any]:
        """
        Main method to generate flashcards
        
        Args:
            topic: Constitutional topic to cover
            num_cards: Number of flashcards to generate
            difficulty: Difficulty level (easy, medium, hard)
            card_type: Type of cards (definitions, articles, cases, mixed)
        
        Returns:
            Dictionary with flashcard data
        """
        try:
            print(f"🎴 Starting flashcard generation for: {topic}")
            start_time = datetime.now()
            
            # Generate flashcard content
            if self.llm:
                flashcard_data = self._generate_flashcards_with_llm(topic, num_cards, difficulty, card_type)
            else:
                flashcard_data = self._generate_fallback_flashcards(topic, num_cards, difficulty)
            
            if not flashcard_data or not flashcard_data.get('cards'):
                return {
                    'success': False,
                    'error': 'Failed to generate flashcard content',
                    'topic': topic
                }
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            print(f"✅ Flashcard generation completed in {processing_time:.1f} seconds")
            
            return {
                'success': True,
                'flashcard_data': flashcard_data,
                'topic': topic,
                'processing_time': processing_time,
                'created_at': end_time.isoformat()
            }
            
        except Exception as e:
            print(f"❌ Error in flashcard generation: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'topic': topic
            }
    
    def _generate_flashcards_with_llm(self, topic: str, num_cards: int, difficulty: str, card_type: str) -> Dict[str, Any]:
        """
        Generate flashcards using LLM
        """
        try:
            prompt = f"""
Create {num_cards} educational flashcards about "{topic}" related to the Indian Constitution.

Requirements:
- Difficulty level: {difficulty}
- Card type focus: {card_type}
- Each card should test important constitutional knowledge
- Questions should be clear and concise
- Answers should be comprehensive but not too long
- Include constitutional articles, rights, principles, or case law as relevant

Structure the response as a JSON object with this exact format:
{{
    "topic": "{topic}",
    "total_cards": {num_cards},
    "difficulty": "{difficulty}",
    "card_type": "{card_type}",
    "cards": [
        {{
            "id": 1,
            "question": "What does Article 21 of the Indian Constitution guarantee?",
            "answer": "Article 21 guarantees the right to life and personal liberty. It states that no person shall be deprived of his life or personal liberty except according to procedure established by law. This has been interpreted broadly to include right to livelihood, right to privacy, right to clean environment, etc.",
            "category": "fundamental_rights",
            "article_reference": "Article 21"
        }},
        {{
            "id": 2,
            "question": "What is the difference between Fundamental Rights and Directive Principles?",
            "answer": "Fundamental Rights are justiciable (enforceable in courts) and are negative obligations on the state, while Directive Principles are non-justiciable guidelines for state policy and are positive obligations. Fundamental Rights are found in Part III (Articles 12-35) and Directive Principles in Part IV (Articles 36-51).",
            "category": "constitutional_principles",
            "article_reference": "Part III & IV"
        }}
        // ... continue for all {num_cards} cards
    ]
}}

Guidelines for different card types:
- "definitions": Focus on constitutional terms, concepts, and legal definitions
- "articles": Focus on specific constitutional articles and their provisions
- "cases": Focus on landmark Supreme Court cases and their significance
- "mixed": Include a variety of definitions, articles, and important cases

Guidelines for difficulty levels:
- "easy": Basic constitutional concepts, well-known rights and articles
- "medium": Detailed provisions, lesser-known articles, important case law
- "hard": Complex constitutional interpretations, detailed case analysis, amendments

Ensure:
1. Questions are specific and test real understanding
2. Answers are accurate and educational
3. Include relevant constitutional article numbers where applicable
4. Cover diverse aspects of the topic
5. Language is clear for students and legal learners

Respond with ONLY the JSON object, no additional text."""
            
            response = self.llm.invoke(prompt)
            flashcard_text = response.content if hasattr(response, 'content') else str(response)
            
            # Clean and parse JSON
            flashcard_text = flashcard_text.strip()
            if flashcard_text.startswith('```json'):
                flashcard_text = flashcard_text[7:-3].strip()
            elif flashcard_text.startswith('```'):
                flashcard_text = flashcard_text[3:-3].strip()
            
            flashcard_data = json.loads(flashcard_text)
            
            # Validate flashcard data
            flashcard_data = self._validate_flashcard_data(flashcard_data, num_cards)
            
            print(f"✅ Generated {len(flashcard_data.get('cards', []))} flashcards")
            return flashcard_data
            
        except Exception as e:
            print(f"Error generating flashcards with LLM: {e}")
            return self._generate_fallback_flashcards(topic, num_cards, difficulty)
    
    def _generate_fallback_flashcards(self, topic: str, num_cards: int, difficulty: str) -> Dict[str, Any]:
        """
        Generate basic fallback flashcards when LLM is not available
        """
        fallback_cards = [
            {
                "id": 1,
                "question": f"What is the main focus of {topic}?",
                "answer": f"{topic} is an important concept in the Indian Constitution that relates to citizens' rights and governmental responsibilities.",
                "category": "general",
                "article_reference": "General"
            },
            {
                "id": 2,
                "question": f"Where is {topic} mentioned in the Constitution?",
                "answer": f"{topic} is addressed in various parts of the Indian Constitution, particularly in the chapters dealing with fundamental rights and directive principles.",
                "category": "constitutional_reference",
                "article_reference": "Multiple Articles"
            }
        ]
        
        # Extend with basic cards up to num_cards
        while len(fallback_cards) < num_cards:
            card_id = len(fallback_cards) + 1
            fallback_cards.append({
                "id": card_id,
                "question": f"Question {card_id} about {topic}",
                "answer": f"This is a sample answer about {topic}. Please configure GEMINI_API_KEY for AI-generated content.",
                "category": "sample",
                "article_reference": "Sample"
            })
        
        return {
            "topic": topic,
            "total_cards": num_cards,
            "difficulty": difficulty,
            "card_type": "fallback",
            "cards": fallback_cards[:num_cards]
        }
    
    def _validate_flashcard_data(self, flashcard_data: Dict[str, Any], expected_cards: int) -> Dict[str, Any]:
        """
        Validate and fix flashcard data
        """
        cards = flashcard_data.get('cards', [])
        
        # Ensure all cards have required fields
        for i, card in enumerate(cards):
            if 'id' not in card:
                card['id'] = i + 1
            if 'question' not in card:
                card['question'] = f"Sample question {i + 1}"
            if 'answer' not in card:
                card['answer'] = f"Sample answer {i + 1}"
            if 'category' not in card:
                card['category'] = "general"
            if 'article_reference' not in card:
                card['article_reference'] = "General"
        
        flashcard_data['cards'] = cards
        flashcard_data['total_cards'] = len(cards)
        
        return flashcard_data

# Create the tool instance
flashcard_tool_instance = FlashcardGenerationTool()

@tool("flashcard_generation_tool", args_schema=FlashcardGenerationInput, return_direct=True)
def flashcard_generation_tool(topic: str, num_cards: int = 10, difficulty: str = "medium", card_type: str = "mixed") -> str:
    """
    Generate interactive flashcards about constitutional topics.
    
    Creates educational Q&A flashcards perfect for studying constitutional law,
    fundamental rights, legal principles, and landmark cases.
    
    Args:
        topic: The constitutional topic to create flashcards about
        num_cards: Number of flashcards to generate (default: 10)
        difficulty: Difficulty level - easy, medium, or hard
        card_type: Type focus - definitions, articles, cases, or mixed
    
    Returns:
        JSON string with flashcard data for interactive display
    """
    result = flashcard_tool_instance.generate_flashcards(
        topic=topic,
        num_cards=num_cards,
        difficulty=difficulty,
        card_type=card_type
    )
    
    if result['success']:
        flashcard_data = result.get('flashcard_data', {})
        cards = flashcard_data.get('cards', [])
        
        response = f"""🎴 **Flashcards Generated Successfully!**

📚 **Flashcard Set Details:**
• **Topic:** {result['topic']}
• **Total Cards:** {len(cards)}
• **Difficulty:** {flashcard_data.get('difficulty', difficulty).title()}
• **Type:** {flashcard_data.get('card_type', card_type).title()}

⏱️ **Processing Time:** {result.get('processing_time', 0):.1f} seconds

🎯 **Sample Cards:**"""
        
        # Show first 2-3 card previews
        for i, card in enumerate(cards[:3], 1):
            response += f"\n\n**Card {i}:**\n**Q:** {card.get('question', 'Sample question')}\n**A:** {card.get('answer', 'Sample answer')[:100]}..."
        
        if len(cards) > 3:
            response += f"\n\n... and {len(cards) - 3} more cards!"
        
        response += f"\n\n✅ **Your interactive flashcard set about '{result['topic']}' is ready for study!**"
        response += f"\n\n🎴 **Flashcard Data:**\n```json\n{json.dumps(flashcard_data, indent=2)}\n```"
        
    else:
        response = f"❌ **Flashcard Generation Failed**\n\n**Topic:** {result['topic']}\n**Error:** {result.get('error', 'Unknown error')}\n\nPlease try again or check the logs for more details."
    
    return response