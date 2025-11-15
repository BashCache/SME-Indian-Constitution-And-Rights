"""
QuizGenerationTool: A LangChain tool for generating quizzes using Gemini API
and optionally exporting them as documents.

Integrates with DocumentGenerationTool for export functionality.
"""

from typing import Dict, Any, List
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tools.document_exporter.enhanced_document_generator import document_generator
import json
import os
from dotenv import load_dotenv

load_dotenv()

class QuizGenerator:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            google_api_key=self.api_key,
            temperature=0.3  # Slightly higher for creative question generation
        )
        self.enhanced_doc_generator = document_generator
        
    def _generate_quiz_instructions(self, question_types: List[str], quiz_params: Dict[str, Any]) -> str:
        """Generate comprehensive quiz instructions based on question types"""
        num_questions = quiz_params.get('num_questions', 5)
        difficulty = quiz_params.get('difficulty', 'medium')
        
        # Calculate suggested time based on question types and count
        time_per_question = {
            'mcq': 2,
            'true_false': 1.5,
            'fill_blank': 3,
            'short_answer': 5,
            'descriptive': 15
        }
        
        # Estimate time based on question distribution
        if len(question_types) == 1:
            estimated_time = num_questions * time_per_question.get(question_types[0], 5)
        else:
            # For mixed types, assume equal distribution and average time
            avg_time = sum(time_per_question.get(qt, 5) for qt in question_types) / len(question_types)
            estimated_time = num_questions * avg_time
        
        instructions = f"""- Read all questions carefully before starting
- Total questions: {num_questions}
- Difficulty level: {difficulty.title()}
- Estimated time: {int(estimated_time)} minutes
- Answer all questions to the best of your ability"""
        
        # Add specific instructions for each question type
        type_instructions = []
        for qtype in question_types:
            if qtype == 'mcq':
                type_instructions.append("- For multiple choice: Select the best answer from the given options")
            elif qtype == 'fill_blank':
                type_instructions.append("- For fill in the blanks: Write the appropriate word/phrase in the blank space")
            elif qtype == 'descriptive':
                type_instructions.append("- For descriptive answers: Provide detailed explanations with examples")
            elif qtype == 'true_false':
                type_instructions.append("- For true/false: Mark each statement as either True or False")
            elif qtype == 'short_answer':
                type_instructions.append("- For short answers: Provide brief, concise responses")
        
        if type_instructions:
            instructions += "\n" + "\n".join(type_instructions)
        
        return instructions.strip()
    
    def _generate_question_type_instructions(self, question_types: List[str]) -> str:
        """Generate detailed instructions for question types being used"""
        instructions = "Instructions for question types in this quiz:\n"
        
        for qtype in question_types:
            if qtype == 'mcq':
                instructions += "- MCQ: Choose the best answer from 4 options (A, B, C, D)\n"
            elif qtype == 'fill_blank':
                instructions += "- Fill in the blank: Complete the sentence with appropriate words\n"
            elif qtype == 'descriptive':
                instructions += "- Descriptive: Provide detailed explanations and analysis\n"
            elif qtype == 'true_false':
                instructions += "- True/False: Mark statements as True or False\n"
            elif qtype == 'short_answer':
                instructions += "- Short answer: Provide brief, focused responses\n"
        
        return instructions.strip()
    
    def _generate_question_format_instructions(self, question_types: List[str], quiz_params: Dict[str, Any]) -> str:
        """Generate example formatting for mixed question types"""
        num_questions = quiz_params.get('num_questions', 5)
        
        # Templates for each question type
        templates = {
            "mcq": """Q{num}. [Question text]?
   A) Option 1
   B) Option 2
   C) Option 3
   D) Option 4""",
            
            "fill_blank": """Q{num}. The _______ guarantees equality before law.""",
            
            "descriptive": """Q{num}. [Detailed question requiring comprehensive answer]""",
            
            "true_false": """Q{num}. [Statement to be evaluated] (True/False)""",
            
            "short_answer": """Q{num}. [Brief question requiring concise answer]"""
        }
        
        instructions = ""
        question_num = 1
        
        # If single question type, show multiple examples
        if len(question_types) == 1:
            qtype = question_types[0]
            template = templates.get(qtype, templates['mcq'])
            
            for i in range(min(3, num_questions)):  # Show up to 3 examples
                instructions += template.format(num=question_num) + "\n\n"
                question_num += 1
        else:
            # For mixed types, show one example of each type
            questions_per_type = max(1, num_questions // len(question_types))
            
            for qtype in question_types:
                template = templates.get(qtype, templates['mcq'])
                instructions += f"[{qtype.upper()} Questions]\n"
                
                for i in range(min(questions_per_type, 2)):  # Show up to 2 examples per type
                    instructions += template.format(num=question_num) + "\n"
                    question_num += 1
                instructions += "\n"
        
        return instructions.strip()
    
    def _generate_answer_format_instructions(self, question_types: List[str]) -> str:
        """Generate answer key format for mixed question types"""
        
        answer_templates = {
            "mcq": """Q{num}. **Answer: A**
   **Explanation:** Brief explanation of why this is correct.""",
            
            "fill_blank": """Q{num}. **Answer: [Correct word/phrase]**
   **Explanation:** Brief explanation.""",
            
            "descriptive": """Q{num}. **Answer:** [Detailed model answer]
   **Key Points:** List main points that should be covered.
   **Grading Criteria:** [Points allocation if applicable]""",
            
            "true_false": """Q{num}. **Answer: True/False**
   **Explanation:** Brief explanation of the reasoning.""",
            
            "short_answer": """Q{num}. **Answer:** [Concise correct answer]
   **Explanation:** Brief explanation."""
        }
        
        instructions = ""
        question_num = 1
        
        if len(question_types) == 1:
            # Single question type
            qtype = question_types[0]
            template = answer_templates.get(qtype, answer_templates['mcq'])
            
            for i in range(min(2, 3)):  # Show 2-3 examples
                instructions += template.format(num=question_num) + "\n\n"
                question_num += 1
        else:
            # Mixed question types
            for qtype in question_types:
                template = answer_templates.get(qtype, answer_templates['mcq'])
                instructions += f"[{qtype.upper()} Answers]\n"
                instructions += template.format(num=question_num) + "\n\n"
                question_num += 1
        
        return instructions.strip()
        
    def create_quiz_prompt(self, quiz_params: Dict[str, Any], rag_context: str = None) -> str:
        """Create a structured prompt for quiz generation based on parameters"""
        
        prompt_template = """
You are an expert quiz generator. Generate a quiz with the following specifications:
- Number of questions: {num_questions}
- Difficulty level: {difficulty}
- Question type(s): {question_type}
- Topic: {topic}
{context_info}

IMPORTANT: Generate your response in TWO SEPARATE SECTIONS:

SECTION 1: QUESTIONS ONLY (No answers or explanations)
SECTION 2: ANSWERS AND EXPLANATIONS

{question_type_instructions}

Format your response exactly as follows:

===== QUESTIONS SECTION =====

=== QUIZ: {title} ===

**INSTRUCTIONS:**
{quiz_instructions}

**Questions:**

{question_format_instructions}

===== ANSWERS SECTION =====

=== ANSWER KEY: {title} ===

{answer_format_instructions}

Generate both sections now:
"""

        # Handle mixed question types
        question_types = quiz_params.get('question_type', 'mcq')
        if isinstance(question_types, str):
            question_types = [question_types]
        elif isinstance(question_types, list):
            question_types = question_types
        else:
            question_types = ['mcq']  # Default fallback
        
        # Generate instructions for mixed question types
        quiz_instructions = self._generate_quiz_instructions(question_types, quiz_params)
        question_type_instructions = self._generate_question_type_instructions(question_types)
        
        # Format specific instructions based on question types
        question_format_instructions = self._generate_question_format_instructions(question_types, quiz_params)
        answer_format_instructions = self._generate_answer_format_instructions(question_types)
        
        # Prepare context information
        context_info = f"\n\nContext from knowledge base:\n{rag_context}" if rag_context else ""
        title = f"{quiz_params.get('topic', 'General Knowledge').title()} Quiz - {quiz_params.get('difficulty', 'Medium').title()} Level"
        
        return prompt_template.format(
            num_questions=quiz_params.get('num_questions', 5),
            difficulty=quiz_params.get('difficulty', 'medium'),
            question_type=' + '.join(question_types),
            topic=quiz_params.get('topic', 'General Knowledge'),
            context_info=context_info,
            title=title,
            quiz_instructions=quiz_instructions,
            question_type_instructions=question_type_instructions,
            question_format_instructions=question_format_instructions,
            answer_format_instructions=answer_format_instructions
        )
    
    async def generate_quiz_content(self, quiz_params: Dict[str, Any], rag_context: str = None) -> str:
        """Generate quiz content using Gemini API"""
        try:
            prompt = self.create_quiz_prompt(quiz_params, rag_context)
            response = await self.llm.ainvoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            print(f"Error generating quiz content: {e}")
            return f"Error generating quiz: {str(e)}"
    
    def parse_quiz_sections(self, full_content: str) -> Dict[str, str]:
        """Parse the generated content into separate questions and answers sections"""
        try:
            # Split content by the section markers
            if "===== QUESTIONS SECTION =====" in full_content and "===== ANSWERS SECTION =====" in full_content:
                parts = full_content.split("===== QUESTIONS SECTION =====", 1)
                if len(parts) > 1:
                    questions_and_answers = parts[1]
                    sections = questions_and_answers.split("===== ANSWERS SECTION =====", 1)
                    
                    if len(sections) == 2:
                        questions_content = sections[0].strip()
                        answers_content = sections[1].strip()
                        
                        return {
                            "questions": questions_content,
                            "answers": answers_content,
                            "combined": full_content
                        }
            
            # Fallback: if sections not properly marked, return original content
            return {
                "questions": full_content,
                "answers": "Answer key not properly generated. Please check the quiz content.",
                "combined": full_content
            }
            
        except Exception as e:
            print(f"Error parsing quiz sections: {e}")
            return {
                "questions": full_content,
                "answers": f"Error parsing answers: {str(e)}",
                "combined": full_content
            }
    
    async def process_quiz_request(self, quiz_params: Dict[str, Any], rag_context: str = None) -> Dict[str, Any]:
        """
        Process a quiz request and return either inline content or document paths
        """
        try:
            # Generate the quiz content
            full_quiz_content = await self.generate_quiz_content(quiz_params, rag_context)
            
            # Parse into separate sections
            quiz_sections = self.parse_quiz_sections(full_quiz_content)
            
            export_info = quiz_params.get('export_info', {})
            should_export = export_info.get('should_export', False)
            
            result = {
                "quiz_content": quiz_sections["combined"],
                "questions_only": quiz_sections["questions"],
                "answers_only": quiz_sections["answers"],
                "success": True,
                "quiz_params": quiz_params
            }
            print(f"should export pdf: {should_export}")
            if should_export:
                # Generate separate documents for questions and answers
                export_format = export_info.get('export_format', 'pdf')
                topic = quiz_params.get('topic', 'Quiz')
                difficulty = quiz_params.get('difficulty', 'medium')
                
                # Generate questions document
                questions_result = self.enhanced_doc_generator.generate_quiz_document(
                    quiz_content=quiz_sections["questions"],
                    quiz_params={**quiz_params, 'export_info': {'export_format': export_format}}
                )
                
                # Generate answers document with modified filename
                answers_params = quiz_params.copy()
                answers_params['topic'] = f"{topic}_ANSWERS"
                answers_result = self.enhanced_doc_generator.generate_quiz_document(
                    quiz_content=quiz_sections["answers"],
                    quiz_params={**answers_params, 'export_info': {'export_format': export_format}}
                )
                
                if questions_result["success"] and answers_result["success"]:
                    result.update({
                        "exported": True,
                        "questions_document": {
                            "path": questions_result["file_path"],
                            "filename": questions_result["filename"],
                            "type": questions_result["document_type"]
                        },
                        "answers_document": {
                            "path": answers_result["file_path"],
                            "filename": answers_result["filename"],
                            "type": answers_result["document_type"]
                        },
                        "should_email": export_info.get('should_email', False)
                    })
            else:
                result.update({
                    "exported": False,
                    "inline_response": True
                })
            
            return result
            
        except Exception as e:
            print(f"Error processing quiz request: {e}")
            return {
                "success": False,
                "error": str(e),
                "quiz_content": "Failed to generate quiz due to an error."
            }

# Create a LangChain tool wrapper
@tool
async def quiz_generation_tool(
    quiz_params: str,
    rag_context: str = ""
) -> str:
    """
    Generate a quiz based on specified parameters for Indian Constitution and Rights.
    Supports mixed question types in a single quiz.
    
    Args:
        quiz_params: JSON string containing quiz parameters:
                    {
                        "num_questions": int,
                        "difficulty": "easy|medium|hard", 
                        "question_type": "mcq|fill_blank|descriptive|true_false|short_answer" 
                                       OR ["mcq", "fill_blank", "descriptive"] (for mixed types),
                        "topic": str,
                        "export_info": {
                            "should_export": bool,
                            "export_format": "pdf|docx|pptx",
                            "should_email": bool
                        }
                    }
        rag_context: Optional context from knowledge base
    
    Returns:
        String containing either the quiz content or success message with document path
    """
    try:
        # Parse quiz parameters
        params = json.loads(quiz_params) if isinstance(quiz_params, str) else quiz_params
        
        # Create quiz generator and process request
        generator = QuizGenerator()
        result = await generator.process_quiz_request(params, rag_context)
        
        if result["success"]:
            if result.get("exported"):
                questions_doc = result.get("questions_document", {})
                answers_doc = result.get("answers_document", {})
                
                return f"""📝 Quiz generated and exported successfully!

🎯 Quiz Details:
   • Topic: {params.get('topic', 'General')}
   • Questions: {params.get('num_questions', 5)}
   • Difficulty: {params.get('difficulty', 'medium')}
   • Type: {params.get('question_type', 'mcq')}

📄 Documents Generated:
   📋 Questions Document:
      • File: {questions_doc.get('filename', 'Unknown')}
      • Format: {questions_doc.get('type', 'Unknown').upper()}
      • Path: {questions_doc.get('path', 'Unknown')}
   
   🔑 Answer Key Document:
      • File: {answers_doc.get('filename', 'Unknown')}
      • Format: {answers_doc.get('type', 'Unknown').upper()}
      • Path: {answers_doc.get('path', 'Unknown')}

{"📧 Email will be sent separately." if result.get('should_email') else ""}

📝 Questions Preview:
{result.get('questions_only', result['quiz_content'])[:300]}...

✅ Your quiz documents are ready for distribution!
"""
            else:
                # For inline response, show questions and answers separately
                questions_content = result.get('questions_only', '')
                answers_content = result.get('answers_only', '')
                
                return f"""📝 Quiz Generated Successfully!

📋 QUESTIONS:
{questions_content[:500]}...

🔑 ANSWER KEY:
{answers_content[:500]}...

💡 Tip: To get separate documents, add 'export as PDF' to your request!
"""
        else:
            return f"Failed to generate quiz: {result.get('error', 'Unknown error')}"
            
    except Exception as e:
        return f"Error in quiz generation tool: {str(e)}"


# Export for easy import
__all__ = ['QuizGenerator', 'quiz_generation_tool']


# Additional tool for getting questions only
@tool
async def quiz_questions_tool(
    quiz_params: str,
    rag_context: str = ""
) -> str:
    """
    Generate only the quiz questions without answers (useful for quiz administration).
    
    Args:
        quiz_params: JSON string containing quiz parameters
        rag_context: Optional context from knowledge base
    
    Returns:
        String containing only the quiz questions
    """
    try:
        params = json.loads(quiz_params) if isinstance(quiz_params, str) else quiz_params
        generator = QuizGenerator()
        
        # Generate full content
        full_content = await generator.generate_quiz_content(params, rag_context)
        
        # Parse and return only questions
        sections = generator.parse_quiz_sections(full_content)
        return sections["questions"]
        
    except Exception as e:
        return f"Error generating quiz questions: {str(e)}"


@tool  
async def quiz_answers_tool(
    quiz_params: str,
    rag_context: str = ""
) -> str:
    """
    Generate only the quiz answer key (useful for educators/graders).
    
    Args:
        quiz_params: JSON string containing quiz parameters
        rag_context: Optional context from knowledge base
    
    Returns:
        String containing only the answer key with explanations
    """
    try:
        params = json.loads(quiz_params) if isinstance(quiz_params, str) else quiz_params
        generator = QuizGenerator()
        
        # Generate full content
        full_content = await generator.generate_quiz_content(params, rag_context)
        
        # Parse and return only answers
        sections = generator.parse_quiz_sections(full_content)
        return sections["answers"]
        
    except Exception as e:
        return f"Error generating quiz answers: {str(e)}"
    
    def _generate_question_type_instructions(self, question_types: List[str]) -> str:
        """Generate detailed instructions for question types being used"""
        instructions = "Instructions for question types in this quiz:\n"
        
        for qtype in question_types:
            if qtype == 'mcq':
                instructions += "- MCQ: Choose the best answer from 4 options (A, B, C, D)\n"
            elif qtype == 'fill_blank':
                instructions += "- Fill in the blank: Complete the sentence with appropriate words\n"
            elif qtype == 'descriptive':
                instructions += "- Descriptive: Provide detailed explanations and analysis\n"
            elif qtype == 'true_false':
                instructions += "- True/False: Mark statements as True or False\n"
            elif qtype == 'short_answer':
                instructions += "- Short answer: Provide brief, focused responses\n"
        
        return instructions.strip()
    
    def _generate_question_format_instructions(self, question_types: List[str], quiz_params: Dict[str, Any]) -> str:
        """Generate example formatting for mixed question types"""
        num_questions = quiz_params.get('num_questions', 5)
        
        # Templates for each question type
        templates = {
            "mcq": """Q{num}. [Question text]?
   A) Option 1
   B) Option 2
   C) Option 3
   D) Option 4""",
            
            "fill_blank": """Q{num}. The _______ guarantees equality before law.""",
            
            "descriptive": """Q{num}. [Detailed question requiring comprehensive answer]""",
            
            "true_false": """Q{num}. [Statement to be evaluated] (True/False)""",
            
            "short_answer": """Q{num}. [Brief question requiring concise answer]"""
        }
        
        instructions = ""
        question_num = 1
        
        # If single question type, show multiple examples
        if len(question_types) == 1:
            qtype = question_types[0]
            template = templates.get(qtype, templates['mcq'])
            
            for i in range(min(3, num_questions)):  # Show up to 3 examples
                instructions += template.format(num=question_num) + "\n\n"
                question_num += 1
        else:
            # For mixed types, show one example of each type
            questions_per_type = max(1, num_questions // len(question_types))
            
            for qtype in question_types:
                template = templates.get(qtype, templates['mcq'])
                instructions += f"[{qtype.upper()} Questions]\n"
                
                for i in range(min(questions_per_type, 2)):  # Show up to 2 examples per type
                    instructions += template.format(num=question_num) + "\n"
                    question_num += 1
                instructions += "\n"
        
        return instructions.strip()
    
    def _generate_answer_format_instructions(self, question_types: List[str]) -> str:
        """Generate answer key format for mixed question types"""
        
        answer_templates = {
            "mcq": """Q{num}. **Answer: A**
   **Explanation:** Brief explanation of why this is correct.""",
            
            "fill_blank": """Q{num}. **Answer: [Correct word/phrase]**
   **Explanation:** Brief explanation.""",
            
            "descriptive": """Q{num}. **Answer:** [Detailed model answer]
   **Key Points:** List main points that should be covered.
   **Grading Criteria:** [Points allocation if applicable]""",
            
            "true_false": """Q{num}. **Answer: True/False**
   **Explanation:** Brief explanation of the reasoning.""",
            
            "short_answer": """Q{num}. **Answer:** [Concise correct answer]
   **Explanation:** Brief explanation."""
        }
        
        instructions = ""
        question_num = 1
        
        if len(question_types) == 1:
            # Single question type
            qtype = question_types[0]
            template = answer_templates.get(qtype, answer_templates['mcq'])
            
            for i in range(min(2, 3)):  # Show 2-3 examples
                instructions += template.format(num=question_num) + "\n\n"
                question_num += 1
        else:
            # Mixed question types
            for qtype in question_types:
                template = answer_templates.get(qtype, answer_templates['mcq'])
                instructions += f"[{qtype.upper()} Answers]\n"
                instructions += template.format(num=question_num) + "\n\n"
                question_num += 1
        
        return instructions.strip()
