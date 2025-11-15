import os
from typing import List, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import json
import re
from pathlib import Path
from langchain_tools.email_agent.email_tool import send_email_tool

# Add the tools directory to the path for email tool import
# tools_path = Path(__file__).parent.parent / "tools"
# sys.path.append(str(tools_path))

# try:
#     from email_tool import EmailTool, send_quiz_email
# except ImportError:
#     EmailTool = None
#     send_quiz_email = None
#     print("Warning: EmailTool not available. Email functionality will be disabled.")

load_dotenv()

class GeminiChatChain:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        # Initialize Gemini model
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",  # Using gemini-pro as it's stable and available
            google_api_key=self.api_key,
            temperature=0.1
        )
        
        # Initialize email tool if available
        self.email_tool = send_email_tool
        
        # System prompt for the Indian Constitution SME
        self.system_prompt = """You are a Subject Matter Expert (SME) on the Indian Constitution and Rights. 
You have deep knowledge of constitutional law, fundamental rights, directive principles, and legal precedents.

Your role is to:
1. Answer questions about Indian Constitution and Rights accurately
2. Identify when a user is asking for quiz generation vs. regular chat
3. Provide helpful, educational responses about constitutional matters

For quiz generation requests, respond with: "QUIZ_REQUEST_DETECTED" followed by the quiz details.
For regular chat, provide informative answers about constitutional topics.

Be professional, accurate, and educational in your responses."""

        # Create the chat prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        # Create the chain
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def detect_intent(self, user_message: str) -> str:
        """
        Detect if the user wants quiz generation or regular chat
        """
        quiz_keywords = [
            'quiz', 'test', 'questions', 'mcq', 'multiple choice', 
            'fill in the blank', 'descriptive questions', 'exam',
            'assessment', 'generate questions', 'create quiz', 'make quiz',
            'quiz me', 'test me', 'practice questions'
        ]
        
        message_lower = user_message.lower()
        
        # Check for quiz-related keywords
        if any(keyword in message_lower for keyword in quiz_keywords):
            return "quiz_generation"
        
        return "chat"
    
    def detect_export_intent(self, user_message: str) -> Dict[str, Any]:
        """
        Detect if the quiz should be exported as a document or shown inline
        """
        message_lower = user_message.lower()
        
        export_keywords = [
            'export', 'download', 'pdf', 'document', 'doc', 'file',
            'save', 'generate document', 'create document', 'report',
            'email', 'send', 'share'
        ]
        
        should_export = any(keyword in message_lower for keyword in export_keywords)
        
        # Determine export format
        export_format = "pdf"  # default
        if "docx" in message_lower or "word" in message_lower:
            export_format = "docx"
        elif "pptx" in message_lower or "powerpoint" in message_lower:
            export_format = "pptx"
        
        # Check if email is requested
        should_email = any(word in message_lower for word in ['email', 'send', 'mail'])
        
        return {
            "should_export": should_export,
            "export_format": export_format,
            "should_email": should_email,
            "inline_response": not should_export  # If not exporting, show inline
        }
    
    def convert_history_to_langchain_format(self, history: List[Dict[str, str]]) -> List:
        """
        Convert session history to LangChain message format
        """
        messages = []
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        return messages
    
    async def get_response(self, user_message: str, history: List[Dict[str, str]] = None, rag_context: str = None) -> Dict[str, Any]:
        """
        Get response from Gemini with conversation history and RAG context
        """
        try:
            # Convert history to LangChain format
            chat_history = []
            if history:
                chat_history = self.convert_history_to_langchain_format(history)
            
            # Detect intent
            intent = self.detect_intent(user_message)
            
            # Extract email address if present in the message
            extracted_email = self.extract_email_from_message(user_message)
            
            # Extract quiz parameters if it's a quiz request
            quiz_params = None
            if intent == "quiz_generation":
                quiz_params = await self.extract_quiz_parameters(user_message)
                print(f"📋 Extracted quiz parameters: {quiz_params}")
                
                # If email was found in message but not in quiz_params, add it
                if extracted_email and quiz_params and not quiz_params.get("export_info", {}).get("recipient_email"):
                    quiz_params.setdefault("export_info", {})["recipient_email"] = extracted_email
                    quiz_params["export_info"]["should_email"] = True
            
            # Prepare input with RAG context if available
            input_text = user_message
            if rag_context:
                input_text = f"Context from knowledge base: {rag_context}\n\nUser question: {user_message}"
            
            # For quiz requests, modify the system prompt to generate structured responses
            if intent == "quiz_generation":
                email_info = ""
                if extracted_email:
                    email_info = f"\nEmail address detected: {extracted_email}"
                
                enhanced_input = f"""
Quiz Generation Request Detected.
Parameters: {quiz_params}{email_info}

Please acknowledge this is a quiz generation request and confirm the parameters.
Respond with: "QUIZ_REQUEST_CONFIRMED" followed by a summary of what quiz will be generated.

Original user request: {input_text}
"""
                input_text = enhanced_input
            
            # Get response from the chain
            response = await self.chain.ainvoke({
                "input": input_text,
                "chat_history": chat_history
            })
            
            return {
                "response": response,
                "intent": intent,
                "quiz_params": quiz_params,
                "extracted_email": extracted_email,
                "email_available": self.is_email_available(),
                "success": True
            }
            
        except Exception as e:
            print(f"Error in GeminiChatChain: {e}")
            return {
                "response": f"I apologize, but I encountered an error: {str(e)}",
                "intent": "error",
                "quiz_params": None,
                "extracted_email": None,
                "email_available": False,
                "success": False
            }
    
    async def extract_quiz_parameters(self, user_message: str) -> Dict[str, Any]:
        """
        Extract quiz parameters from user message using LLM for intelligent parsing
        """
        try:
            # Create a specialized prompt for parameter extraction
            extraction_prompt = f"""
You are a parameter extraction specialist. Analyze the following user message and extract quiz parameters.

User Message: "{user_message}"

Extract the following parameters and respond ONLY with a valid JSON object:

{{
    "num_questions": <total number of questions (default: 5)>,
    "difficulty": "<easy|medium|hard> (default: medium)",
    "question_type": "<mcq|fill_blank|descriptive|true_false|short_answer> (default: mcq)",
    "topic": "<any topic mentioned by user or null>",
    "marks_per_question": <marks per question (default: 1)>,
    "total_marks": <total marks for the quiz (default: calculated from num_questions * marks_per_question)>,
    "question_distribution": {{
        "mcq": {{
            "count": <number of MCQ questions (default: 0)>,
            "marks_each": <marks per MCQ question (default: 1)>
        }},
        "fill_blank": {{
            "count": <number of fill-in-the-blank questions (default: 0)>,
            "marks_each": <marks per fill-in-the-blank question (default: 1)>
        }},
        "descriptive": {{
            "count": <number of descriptive questions (default: 0)>,
            "marks_each": <marks per descriptive question (default: 5)>
        }},
        "true_false": {{
            "count": <number of true/false questions (default: 0)>,
            "marks_each": <marks per true/false question (default: 1)>
        }},
        "short_answer": {{
            "count": <number of short answer questions (default: 0)>,
            "marks_each": <marks per short answer question (default: 2)>
        }}
    }},
    "export_info": {{
        "should_export": <true if user wants document/pdf/export/download/file>,
        "export_format": "<pdf|docx|pptx> (default: pdf)",
        "should_email": <true if user mentions email/send/mail>,
        "recipient_email": "<email address if mentioned by user, otherwise null>",
        "inline_response": <opposite of should_export>
    }}
}}

Guidelines for extraction:
1. num_questions: Look for total number of questions. If user specifies distribution (e.g., "5 MCQ, 3 descriptive"), calculate total.
2. difficulty: Look for words like easy/simple/basic/beginner → easy, hard/difficult/advanced/expert → hard, medium/moderate/intermediate → medium
3. question_type: 
   - If user mentions multiple types (e.g., "5 MCQ and 3 fill-ups"), set to "mixed"
   - If only one type is mentioned, use that type (mcq|fill_blank|descriptive|true_false|short_answer)
   - Default is "mcq"
4. topic: Extract ANY topic mentioned by the user (constitutional, general knowledge, science, history, etc.)
5. marks_per_question: Look for patterns like "2 marks each", "5 points per question" (used as fallback)
6. total_marks: Look for patterns like "total 100 marks", "out of 50" (if not specified, will be calculated)
7. question_distribution: IMPORTANT - Extract specific counts and marks for each question type:
   - Look for patterns like "5 MCQ questions", "3 descriptive of 10 marks each", "2 fill-ups"
   - If user says "5 MCQ and 3 fill-ups", set mcq count=5, fill_blank count=3, others=0
   - If user says "5 MCQs and 3 fill ups", DO NOT put 8 in MCQ, put 5 in mcq and 3 in fill_blank
   - If only one type is mentioned, put all questions in that type
   - If user specifies marks for specific question types, use those
   - Default marks: MCQ=1, Fill blank=1, True/False=1, Short answer=2, Descriptive=5
   - CRITICAL: When multiple types are mentioned, distribute correctly, don't put total in one type
8. export_info: Check for export-related keywords and email addresses:
   - should_export: Look for words like "export", "download", "pdf", "document", "file", "save"
   - should_email: Look for words like "email", "send", "mail", "share"
   - recipient_email: Extract email addresses using patterns like user@domain.com
   - export_format: Determine from context (pdf, docx, pptx)

Examples of user patterns to recognize:
- "5 MCQ of 2 marks each and 3 descriptive of 10 marks each" → mixed type, mcq=5(2 marks), descriptive=3(10 marks), total=8 questions
- "10 questions, 20 marks total" → mcq type, mcq=10(2 marks each), total=10 questions
- "Create quiz with 5 MCQ and 2 essay questions" → mixed type, mcq=5(1 mark), descriptive=2(5 marks), total=7 questions
- "Generate 15 questions about constitution, 30 marks" → mcq type, mcq=15(2 marks each), total=15 questions
- "3 fill-ups and 4 MCQs" → mixed type, fill_blank=3(1 mark), mcq=4(1 mark), total=7 questions
- "5 MCQs and 3 fill ups" → mixed type, mcq=5(1 mark), fill_blank=3(1 mark), total=8 questions
- "2 true/false and 3 short answer" → mixed type, true_false=2(1 mark), short_answer=3(2 marks), total=5 questions

Respond with ONLY the JSON object, no explanations.
"""

            # Use LangChain's Gemini model for parameter extraction
            extraction_llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",
                google_api_key=self.api_key,
                temperature=0.1
            )
            
            response = await extraction_llm.ainvoke(extraction_prompt)
            
            # Parse the JSON response
            json_str = response.content if hasattr(response, 'content') else str(response)
            json_str = json_str.strip()
            # Remove any markdown code blocks if present
            if json_str.startswith('```'):
                json_str = json_str.split('```')[1]
                if json_str.startswith('json'):
                    json_str = json_str[4:].strip()
            
            params = json.loads(json_str)
            
            # Validate and set defaults for any missing fields
            defaults = {
                "num_questions": 5,
                "difficulty": "medium",
                "question_type": "mcq",
                "topic": None,
                "marks_per_question": 1,
                "total_marks": None,  # Will be calculated if not provided
                "question_distribution": {
                    "mcq": {"count": 0, "marks_each": 1},
                    "fill_blank": {"count": 0, "marks_each": 1},
                    "descriptive": {"count": 0, "marks_each": 5},
                    "true_false": {"count": 0, "marks_each": 1},
                    "short_answer": {"count": 0, "marks_each": 2}
                },
                "export_info": {
                    "should_export": False,
                    "export_format": "pdf",
                    "should_email": False,
                    "recipient_email": None,
                    "inline_response": True
                }
            }
            
            # Merge with defaults
            for key, default_value in defaults.items():
                if key not in params:
                    params[key] = default_value
                elif key == "export_info" and isinstance(params[key], dict):
                    for sub_key, sub_default in default_value.items():
                        if sub_key not in params[key]:
                            params[key][sub_key] = sub_default
                elif key == "question_distribution" and isinstance(params[key], dict):
                    for q_type, q_defaults in default_value.items():
                        if q_type not in params[key]:
                            params[key][q_type] = q_defaults
                        elif isinstance(params[key][q_type], dict):
                            for attr, attr_default in q_defaults.items():
                                if attr not in params[key][q_type]:
                                    params[key][q_type][attr] = attr_default
            
            # Auto-populate question distribution if only single type is specified
            if params["question_type"] != "mixed" and params["question_type"] in params["question_distribution"]:
                params["question_distribution"][params["question_type"]]["count"] = params["num_questions"]
            
            # Check if we have mixed types by counting how many question types have non-zero counts
            types_with_questions = sum(1 for q_info in params["question_distribution"].values() if q_info["count"] > 0)
            total_questions_in_distribution = sum(q_info["count"] for q_info in params["question_distribution"].values())
            
            # If we have more than one question type with questions, it's mixed
            if types_with_questions > 1:
                params["question_type"] = "mixed"
                params["num_questions"] = total_questions_in_distribution
            
            # Calculate total marks based on question distribution
            if params["total_marks"] is None:
                total_marks_calculated = 0
                for q_type, q_info in params["question_distribution"].items():
                    total_marks_calculated += q_info["count"] * q_info["marks_each"]
                
                # If no distribution is specified, use default calculation
                if total_marks_calculated == 0:
                    total_marks_calculated = params["num_questions"] * params["marks_per_question"]
                
                params["total_marks"] = total_marks_calculated
            
            return params
            
        except Exception as e:
            print(f"Error in LLM parameter extraction: {e}")
            # Fallback to simple regex extraction if LLM fails
            return self._fallback_regex_extraction(user_message)
    
    def _fallback_regex_extraction(self, user_message: str) -> Dict[str, Any]:
        """
        Fallback with simple default values if LLM extraction fails
        """
        # Check if user message contains mixed type indicators
        message_lower = user_message.lower()
        
        # Simple check for mixed types by looking for "and" between question types
        has_mixed_indicators = any([
            "mcq and" in message_lower,
            "and mcq" in message_lower,
            "fill" in message_lower and ("mcq" in message_lower or "descriptive" in message_lower),
            "descriptive" in message_lower and "mcq" in message_lower,
            "true" in message_lower and "false" in message_lower and ("mcq" in message_lower or "fill" in message_lower)
        ])
        
        if has_mixed_indicators:
            # Return mixed type defaults
            return {
                "num_questions": 8,  # 5 MCQ + 3 fill-ups
                "difficulty": "medium",
                "question_type": "mixed",
                "topic": None,
                "marks_per_question": 1,
                "total_marks": 11,  # 5*1 + 3*2
                "question_distribution": {
                    "mcq": {"count": 5, "marks_each": 1},
                    "fill_blank": {"count": 3, "marks_each": 2},
                    "descriptive": {"count": 0, "marks_each": 5},
                    "true_false": {"count": 0, "marks_each": 1},
                    "short_answer": {"count": 0, "marks_each": 2}
                },
                "export_info": {
                    "should_export": False,
                    "export_format": "pdf",
                    "should_email": False,
                    "recipient_email": None,
                    "inline_response": True
                }
            }
        else:
            # Return single type defaults (MCQ)
            return {
                "num_questions": 5,
                "difficulty": "medium",
                "question_type": "mcq",
                "topic": None,
                "marks_per_question": 1,
                "total_marks": 5,
                "question_distribution": {
                    "mcq": {"count": 5, "marks_each": 1},
                    "fill_blank": {"count": 0, "marks_each": 1},
                    "descriptive": {"count": 0, "marks_each": 5},
                    "true_false": {"count": 0, "marks_each": 1},
                    "short_answer": {"count": 0, "marks_each": 2}
                },
                "export_info": {
                    "should_export": False,
                    "export_format": "pdf",
                    "should_email": False,
                    "recipient_email": None,
                    "inline_response": True
                }
            }
    
    def extract_email_from_message(self, user_message: str) -> Optional[str]:
        """
        Extract email address from user message using regex
        """
        # Common email patterns
        email_patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            r'email\s*(?:to|address)?\s*:?\s*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})',
            r'send\s*(?:to|at)?\s*:?\s*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})',
            r'mail\s*(?:to|at)?\s*:?\s*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})'
        ]
        
        for pattern in email_patterns:
            match = re.search(pattern, user_message, re.IGNORECASE)
            if match:
                # If pattern has groups, use the first group, otherwise use the full match
                return match.group(1) if match.groups() else match.group(0)
        
        return None
    
    def send_quiz_email(self, 
                       file_path: str,
                       recipient_email: str,
                       quiz_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send quiz document via email using the email tool
        """
        if not self.email_tool:
            return {
                "success": False,
                "message": "Email functionality not available. EmailTool not initialized.",
                "error": "EMAIL_TOOL_NOT_AVAILABLE"
            }
        
        try:
            # Extract quiz details for email customization
            topic = quiz_params.get("topic", "Constitution")
            difficulty = quiz_params.get("difficulty", "medium")
            num_questions = quiz_params.get("num_questions", 5)
            total_marks = quiz_params.get("total_marks", 5)
            question_type = quiz_params.get("question_type", "mcq")
            
            # Create subject line
            subject = f"Quiz - {topic.title()} ({difficulty.title()} Level)"
            
            # Create email body
            body = f"""Dear Student,

Please find attached your quiz on {topic.title()}.

Quiz Details:
• Topic: {topic.title()}
• Difficulty Level: {difficulty.title()}
• Number of Questions: {num_questions}
• Total Marks: {total_marks}
• Question Type: {question_type.replace('_', ' ').title()}

Instructions:
• Read all questions carefully before answering
• Manage your time effectively
• Review your answers before submission
• Follow the marking scheme as indicated

Best of luck with your quiz!

Regards,
Constitution SME System
"""
            
            # Send email using the email tool
            result = self.email_tool.send_email(
                recipient_email=recipient_email,
                subject=subject,
                body=body,
                attachment_path=file_path
            )
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to send quiz email: {str(e)}",
                "error": "EMAIL_SEND_ERROR"
            }
    
    def send_document_email(self,
                           file_path: str,
                           recipient_email: str,
                           document_type: str = "document",
                           subject: Optional[str] = None,
                           custom_body: Optional[str] = None) -> Dict[str, Any]:
        """
        Send any document via email with customizable content
        """
        if not self.email_tool:
            return {
                "success": False,
                "message": "Email functionality not available. EmailTool not initialized.",
                "error": "EMAIL_TOOL_NOT_AVAILABLE"
            }
        
        try:
            # Default subject if not provided
            if not subject:
                subject = f"{document_type.title()} from Constitution SME"
            
            # Default body if not provided
            if not custom_body:
                custom_body = f"""Dear User,

Please find the attached {document_type} generated by the Constitution SME system.

This document has been created based on your requirements and contains relevant information about Indian Constitutional law and rights.

If you have any questions or need further clarification, please feel free to reach out.

Best regards,
Constitution SME System
"""
            
            # Send email using the email tool
            result = self.email_tool.send_email(
                recipient_email=recipient_email,
                subject=subject,
                body=custom_body,
                attachment_path=file_path
            )
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to send document email: {str(e)}",
                "error": "EMAIL_SEND_ERROR"
            }
    
    def is_email_available(self) -> bool:
        """
        Check if email functionality is available
        """
        return self.email_tool is not None
