# utils/production_orchestrator.py

import asyncio
import json
import re
from typing import Dict, List, Any
from utils.agent_tools import get_rag_answer, document_tool, email_tool
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_core.messages import HumanMessage
import json

# Production-ready orchestrator that doesn't rely on problematic LangChain versions
class ProductionLangChainAgent:
    """Production LangChain agent with tool calling for SME tasks"""
    
    def __init__(self):
        print("🔧 Initializing Production LangChain Agent...")
        
        try:
            self.get_rag_answer = get_rag_answer
            self.document_tool = document_tool
            self.email_tool = email_tool
        except ImportError:
            print(f"error with init production langchain agent")
        
        # Try to initialize real LLM, fall back to pattern matching
        self.llm = None
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",
                temperature=0.1,
                google_api_key=os.getenv("GEMINI_API_KEY"),
                verbose=False
            )
            print("✅ Real LLM initialized")
        except Exception as e:
            print(f"⚠️ LLM initialization failed, using pattern matching: {e}")
        
        print("✅ Production LangChain Agent initialized successfully!")
    
    async def _analyze_request_with_llm(self, user_message: str) -> Dict[str, Any]:
        """Analyze user request using LLM for intelligent tool calling"""
        try:
            tools_description = """
                Available tools:
                1. rag_tool - Retrieve information from knowledge base (parameters: query, source)
                2. quiz_generator_tool - Generate educational quizzes (parameters: topic, num_questions, difficulty, question_type)
                3. document_export_tool - Export any content or document to PDF/DOCX/PPTX (parameters: content, document_type, title)
                4. email_automation_tool - Send documents via email (parameters: filename, recipient, subject)

                Task: Analyze the user request and determine which tools to use and with what parameters.
                Return a JSON response with this structure:
                {
                    "tools_needed": [
                        {
                            "tool": "tool_name",
                            "parameters": {
                                "param1": "value1",
                                "param2": "value2"
                            }
                        }
                    ],
                    "reasoning": "Brief explanation of the plan",
                    "direct_response": "Direct response if no tools needed"
                }
            """

            # Create the prompt for tool selection
            analysis_prompt = f"""{tools_description}

                User Request: "{user_message}"

                Analyze this request and determine:
                1. What the user wants to accomplish
                2. Which tools are needed (if any)
                3. What parameters to use for each tool
                4. The logical sequence of operations

                If the request is for:
                - Information/explanation: use rag_tool
                - Quiz generation: use quiz_generator_tool, optionally followed by document_export_tool and/or email_automation_tool
                - Document creation: use document_export_tool
                - Email: use email_automation_tool

                For quiz generation, extract:
                - topic: from the content (default: "General Knowledge")
                - num_questions: from numbers mentioned (default: 5)
                - difficulty: easy/medium/hard (default: "medium")
                - question_type: mcq/fill_blank/true_false/short_answer/descriptive (default: "mcq")

                Respond with valid JSON only."""

            response = await asyncio.to_thread(
                self.llm.invoke,
                [HumanMessage(content=analysis_prompt)]
            )
            
            try:
                # Extract JSON from the response
                response_text = response.content.strip()
                if response_text.startswith('```json'):
                    response_text = response_text.split('```json')[1].split('```')[0].strip()
                elif response_text.startswith('```'):
                    response_text = response_text.split('```')[1].split('```')[0].strip()
                
                plan = json.loads(response_text)
                print(f"🧠 LLM Analysis: {plan.get('reasoning', 'No reasoning provided')}")
                return plan
                
            except json.JSONDecodeError as e:
                print(f"⚠️ LLM JSON parsing failed: {e}")
                print(f"Raw response: {response_text}")
                return self._analyze_request_with_patterns(user_message)
                
        except Exception as e:
            print(f"⚠️ LLM analysis failed: {e}")
            return self._analyze_request_with_patterns(user_message)

#     def _analyze_request_with_patterns(self, user_message: str) -> Dict[str, Any]:
#         message_lower = user_message.lower()
        
#         # Quiz generation patterns
#         if any(word in message_lower for word in ["quiz", "questions", "mcq", "test", "generate"]):
#             # Extract topic
#             topic = "General Knowledge"
#             if match := re.search(r"(?:on|about|regarding)\s+([^,\.\!\?]+)", user_message, re.IGNORECASE):
#                 topic = match.group(1).strip()
#             elif "article 14" in message_lower:
#                 topic = "Article 14"
#             elif "article 15" in message_lower:
#                 topic = "Article 15"
#             elif "fundamental rights" in message_lower:
#                 topic = "Fundamental Rights"
#             elif "directive principles" in message_lower:
#                 topic = "Directive Principles"
            
#             # Extract number of questions
#             num_questions = 5
#             if match := re.search(r"(\d+)\s*(?:questions?|mcqs?)", message_lower):
#                 num_questions = int(match.group(1))
            
#             # Extract question type
#             question_type = "mcq"
#             if any(word in message_lower for word in ["fill", "blank"]):
#                 question_type = "fill_blank"
#             elif any(word in message_lower for word in ["true", "false"]):
#                 question_type = "true_false"
#             elif "short answer" in message_lower:
#                 question_type = "short_answer"
#             elif "descriptive" in message_lower:
#                 question_type = "descriptive"
            
#             # Check for export
#             export_pdf = any(word in message_lower for word in ["pdf", "export", "document"])
            
#             plan = {
#                 "tools_needed": [
#                     {
#                         "tool": "quiz_generator_tool",
#                         "parameters": {
#                             "topic": topic,
#                             "num_questions": num_questions,
#                             "difficulty": "medium",
#                             "question_type": question_type
#                         }
#                     }
#                 ],
#                 "reasoning": f"Generate {num_questions} {question_type} questions on {topic}"
#             }
            
#             # Add document export if requested
#             if export_pdf:
#                 plan["tools_needed"].append({
#                     "tool": "document_export_tool",
#                     "parameters": {
#                         "content": "[[QUIZ_RESULT]]",  # Placeholder for quiz output
#                         "document_type": "pdf",
#                         "title": f"{topic.replace(' ', '_')}_Quiz"
#                     }
#                 })
#                 plan["reasoning"] += " and export as PDF"
            
#             # Check for email (after document export if needed)
#             if "email" in message_lower or "@" in user_message:
#                 email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", user_message)
#                 recipient = email_match.group(0) if email_match else "user@example.com"
                
#                 # Use appropriate filename based on whether PDF export was requested
#                 if export_pdf:
#                     filename = f"{topic.replace(' ', '_')}_Quiz.pdf"
#                 else:
#                     filename = f"{topic.replace(' ', '_')}_Quiz.txt"
                
#                 plan["tools_needed"].append({
#                     "tool": "email_automation_tool",
#                     "parameters": {
#                         "filename": filename,
#                         "recipient": recipient,
#                         "subject": f"Quiz: {topic}"
#                     }
#                 })
#                 plan["reasoning"] += f" and email to {recipient}"
            
#             return plan
        
#         # Information lookup patterns
#         elif any(word in message_lower for word in ["what is", "explain", "define", "tell me about"]):
#             return {
#                 "tools_needed": [
#                     {
#                         "tool": "rag_tool",
#                         "parameters": {
#                             "query": user_message,
#                             "source": "external_kb"
#                         }
#                     }
#                 ],
#                 "reasoning": "Information lookup required"
#             }
        
#         # Default case
#         return {
#             "tools_needed": [],
#             "direct_response": "I can help you with quiz generation, document export, and information about the Indian Constitution. Please let me know what you'd like to do!"
#         }
    
#     async def _execute_rag_tool(self, params: Dict) -> str:
#         """Execute RAG tool"""
#         try:
#             result = await asyncio.to_thread(
#                 self.get_rag_answer,
#                 query=params.get("query", ""),
#                 source=params.get("source", "external_kb"),
#                 username=params.get("username", "user"),
#                 session_id=params.get("session_id", "default"),
#                 history="",
#                 filepath=None
#             )
#             print(f"RAG result: {result}")
#             return result
#         except Exception as e:
#             return f"Error retrieving information: {str(e)}"
    
#     async def _execute_quiz_tool(self, params: Dict) -> str:
#         """Execute quiz generation tool"""
#         try:
#             from utils.quiz_generator import QuizGenerator
#             quiz_generator = QuizGenerator()
            
#             quiz_params = {
#                 'topic': params.get('topic', 'General Knowledge'),
#                 'num_questions': params.get('num_questions', 5),
#                 'difficulty': params.get('difficulty', 'medium'),
#                 'question_type': params.get('question_type', 'mcq'),
#                 'export_pdf': False  # Always False - document export handled separately
#             }
            
#             result = await quiz_generator.process_quiz_request(
#                 quiz_params=quiz_params,
#                 rag_context=""
#             )
            
#             if result["success"]:
#                 questions_content = result.get('questions_only', '')
#                 return f"✅ **Quiz Generated Successfully!**\n\n{questions_content}"
#             else:
#                 return f"❌ Failed to generate quiz: {result.get('error', 'Unknown error')}"
                
#         except Exception as e:
#             # Fallback to mock for demo
#             return f"""✅ **Quiz Generated Successfully!** (Demo Mode)

# 📋 **Quiz Details:**
# • Topic: {params.get('topic', 'General Knowledge')}
# • Questions: {params.get('num_questions', 5)}
# • Type: {params.get('question_type', 'mcq').upper()}

# **Sample Question:**
# 1. What is the main principle of {params.get('topic', 'the topic')}?
#    A) Option A
#    B) Option B ✓
#    C) Option C
#    D) Option D

# *Note: Full quiz generation requires properly configured dependencies*"""
    
#     async def _execute_document_tool(self, params: Dict) -> str:
#         """Execute document export tool"""
#         try:
#             if self.document_tool == self._mock_document:
#                 # Use async mock directly
#                 result = await self._mock_document(
#                     content=params.get("content", ""),
#                     document_type=params.get("document_type", "pdf"),
#                     title=params.get("title", "Generated Document")
#                 )
#                 return f"✅ Document exported successfully as {result}"
#             else:
#                 # Use real tool with thread
#                 print(f"In else part: {params}")
#                 result = await asyncio.to_thread(
#                     self.document_tool,
#                     content=params.get("content", ""),
#                     document_type=params.get("document_type", "pdf"),
#                     title=params.get("title", "Generated Document")
#                 )
#                 return f"✅ Document exported successfully as {result}"
#         except Exception as e:
#             return f"✅ Document would be exported as {params.get('title', 'Document')}.{params.get('document_type', 'pdf')} (Demo Mode)"
    
#     async def _execute_email_tool(self, params: Dict) -> str:
#         """Execute email automation tool"""
#         try:
#             if self.email_tool == self._mock_email:
#                 # Use async mock directly
#                 result = await self._mock_email(
#                     filename=params.get("filename", ""),
#                     recipient=params.get("recipient", ""),
#                     subject=params.get("subject", "Document from SME Assistant")
#                 )
#                 return f"✅ Email sent successfully: {result}"
#             else:
#                 # Use real tool with thread
#                 result = await asyncio.to_thread(
#                     self.email_tool,
#                     filename=params.get("filename", ""),
#                     recipient=params.get("recipient", ""),
#                     subject=params.get("subject", "Document from SME Assistant")
#                 )
#                 return f"✅ Email sent successfully to {params.get('recipient')}"
#         except Exception as e:
#             return f"✅ Email would be sent: {params.get('filename', 'file')} to {params.get('recipient', 'recipient')} (Demo Mode)"
    
    # async def process_request(self, user_message: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
    #     """Process user request using production agent"""
    #     try:
    #         print(f"🤖 Processing request: {user_message[:100]}...")
            
    #         # Analyze request (use LLM if available, otherwise pattern matching)
    #         if self.llm:
    #             plan = await self._analyze_request_with_llm(user_message)
    #         else:
    #             plan = self._analyze_request_with_patterns(user_message)
            
    #         print(f"📋 Execution plan: {json.dumps(plan, indent=2)}")

    #         # Check for direct response
    #         if plan["direct_response"]:
    #             return {
    #                 "success": True,
    #                 "response": plan["direct_response"],
    #                 "agent_used": True
    #             }
            
    #         # Execute tools
    #         final_response = ""
    #         tools_executed = []
    #         tool_outputs = {}  # Store outputs for tool chaining
            
    #         for i, tool_call in enumerate(plan.get("tools_needed", [])):
    #             tool_name = tool_call["tool"]
    #             tool_params = tool_call["parameters"]
                
    #             # Handle placeholders for tool chaining
    #             for key, value in tool_params.items():
    #                 if value == "[[QUIZ_RESULT]]" and "quiz_generator_tool" in tool_outputs:
    #                     tool_params[key] = tool_outputs["quiz_generator_tool"]
                
    #             print(f"🔧 Executing {tool_name}...")
                
    #             if tool_name == "rag_tool":
    #                 result = await self._execute_rag_tool(tool_params)
    #             elif tool_name == "quiz_generator_tool":
    #                 result = await self._execute_quiz_tool(tool_params)
    #                 tool_outputs["quiz_generator_tool"] = result  # Store for potential document export
    #             elif tool_name == "document_export_tool":
    #                 result = await self._execute_document_tool(tool_params)
    #             elif tool_name == "email_automation_tool":
    #                 result = await self._execute_email_tool(tool_params)
    #             else:
    #                 result = f"❌ Unknown tool: {tool_name}"
                
    #             final_response += f"{result}\n\n"
    #             tools_executed.append(tool_name)
            
    #         if tools_executed:
    #             final_response = final_response.strip()
    #             final_response += f"\n\n*{plan.get('reasoning', 'Task completed')}*"
    #         else:
    #             final_response = "I couldn't determine how to help with your request. Please try rephrasing it."
            
    #         return {
    #             "success": True,
    #             "response": final_response,
    #             "agent_used": True,
    #             "tools_executed": tools_executed
    #         }
            
    #     except Exception as e:
    #         print(f"❌ Processing error: {e}")
    #         return {
    #             "success": False,
    #             "response": f"I encountered an error while processing your request: {str(e)}",
    #             "agent_used": False
    #         }

    async def process_request(self, user_message: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Process user request using production agent with proper chaining + placeholder resolution"""
        try:
            print(f"🤖 Processing request: {user_message[:100]}...")

            # Step 1: Build execution plan
            if self.llm:
                plan = await self._analyze_request_with_llm(user_message)
            else:
                plan = self._analyze_request_with_patterns(user_message)

            print(f"📋 Execution plan: {json.dumps(plan, indent=2)}")

            # Step 2: Direct response
            if plan["direct_response"]:
                return {
                    "success": True,
                    "response": plan["direct_response"],
                    "agent_used": True
                }

            # Step 3: Prepare containers
            final_response = ""
            tools_executed = []
            tool_outputs = {}   # maps tool_name -> output string

            # Step 4: Execute tools sequentially
            for tool_call in plan.get("tools_needed", []):
                tool_name = tool_call["tool"]
                tool_params = tool_call.get("parameters", {})

                print(f"🔧 Preparing to execute {tool_name} with params: {tool_params}")

                # Step 4A: Resolve placeholders dynamically
                resolved_params = self._resolve_placeholders(tool_params, tool_outputs)

                print(f"🔧 Resolved params for {tool_name}: {resolved_params}")

                # Step 4B: Call the correct tool
                if tool_name == "rag_tool":
                    output = await self._execute_rag_tool(resolved_params)

                elif tool_name == "quiz_generator_tool":
                    output = await self._execute_quiz_tool(resolved_params)

                elif tool_name == "document_export_tool":
                    output = await self._execute_document_tool(resolved_params)

                elif tool_name == "email_automation_tool":
                    output = await self._execute_email_tool(resolved_params)

                else:
                    output = f"❌ Unknown tool: {tool_name}"

                # Step 4C: Store tool output for chaining
                tool_outputs[tool_name] = output
                tools_executed.append(tool_name)

                final_response += f"{output}\n\n"

            # Step 5: Final combined response
            if tools_executed:
                final_response = final_response.strip()
                final_response += f"\n\n*{plan.get('reasoning', 'Task completed')}*"
            else:
                final_response = "I couldn't determine how to help with your request."

            return {
                "success": True,
                "response": final_response,
                "agent_used": True,
                "tools_executed": tools_executed,
                "tool_outputs": tool_outputs
            }

        except Exception as e:
            print(f"❌ Processing error: {e}")
            return {
                "success": False,
                "response": f"Error while processing request: {str(e)}",
                "agent_used": False
            }


    # ---------------------------------------------------------
    # INTERNAL UTILITY FOR PLACEHOLDER RESOLUTION
    # ---------------------------------------------------------
    def _resolve_placeholders(self, parameters: Dict[str, Any], tool_outputs: Dict[str, str]):
        """
        Replaces placeholders like {{quiz_generator_tool.output}} dynamically
        using previously executed tool outputs.
        """
        PLACEHOLDER_PATTERN = re.compile(r"\{\{(.*?)\.output\}\}")
        resolved = {}

        for key, value in parameters.items():
            if isinstance(value, str):
                # Search for placeholder pattern
                match = PLACEHOLDER_PATTERN.search(value)

                if match:
                    referenced_tool = match.group(1)
                    if referenced_tool in tool_outputs:
                        resolved[key] = value.replace(
                            f"{{{{{referenced_tool}.output}}}}",
                            tool_outputs[referenced_tool]
                        )
                    else:
                        resolved[key] = value  # leave as is if tool not run yet
                else:
                    resolved[key] = value
            else:
                resolved[key] = value

        return resolved

# Global instance
_production_agent = None

async def process_with_production_langchain(user_message: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
    """Process user message using production orchestration"""
    global _production_agent
    
    if _production_agent is None:
        _production_agent = ProductionLangChainAgent()
    
    return await _production_agent.process_request(user_message, history)

# def get_production_orchestrator_info() -> Dict[str, Any]:
#     """Get production orchestrator info"""
#     return {
#         "status": "active",
#         "agent_type": "Production LangChain Agent",
#         "llm_model": "gemini-2.5-pro (with pattern matching fallback)",
#         "available_tools": [
#             {
#                 "name": "rag_tool",
#                 "description": "Retrieve relevant information from knowledge base",
#                 "parameters": ["query", "source", "username", "session_id"]
#             },
#             {
#                 "name": "quiz_generator_tool",
#                 "description": "Generate educational quizzes on any topic",
#                 "parameters": ["topic", "num_questions", "difficulty", "question_type", "export_pdf"]
#             },
#             {
#                 "name": "document_export_tool", 
#                 "description": "Export content to PDF, DOCX, or PPTX documents",
#                 "parameters": ["content", "document_type", "title", "filename"]
#             },
#             {
#                 "name": "email_automation_tool",
#                 "description": "Send documents via email",
#                 "parameters": ["filename", "recipient", "subject", "body"]
#             }
#         ],
#         "capabilities": [
#             "Quiz generation (MCQs, fill-ups, true/false, short answers, descriptive)",
#             "Document export to PDF/DOCX/PPTX formats", 
#             "Email automation for document sharing",
#             "RAG-based information retrieval",
#             "Constitutional law and rights education",
#             "Intelligent pattern-based request analysis",
#             "Robust fallback mechanisms"
#         ]
#     }
