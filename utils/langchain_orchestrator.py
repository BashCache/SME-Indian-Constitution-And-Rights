# orchestrator.py

import os
import time
import json
from datetime import datetime
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_tools.document_exporter.document_export_tool import document_export_tool
from langchain_tools.content_generator.plain_content_tool import normal_content_tool
from langchain_tools.content_generator.web_search_tool import web_search_tool
from langchain_tools.email_agent.email_tool import send_email_tool
from langchain_tools.video_generator.video_generation_tool import video_generation_tool
from langchain_tools.flashcard_generator.flashcard_generation_tool import flashcard_generation_tool
from langchain_tools.interactive_quiz.interactive_quiz_tool import interactive_quiz_tool

# your memory
from utils.memory_store import get_memory, append_to_memory
from utils.retrieval_tool import RAGTool
from utils.guardrails import GuardrailRunnable

def extract_rag_context(user_message: str, top_k: int = 3) -> str:
    try:
        rag_tool = RAGTool(model_key="legal-bert")
        search_results = rag_tool.search(user_message, top_k=top_k)
        if not search_results:
            return "No relevant context found in knowledge base."
        
        context_parts = []
        for i, result in enumerate(search_results, 1):
            context_parts.append(
                f"Document {i} (Score: {result['score']:.3f}):\n"
                f"Source: {result['source']}\n"
                f"Content: {result['text']}\n"
                f"Labels: {', '.join(result['labels'])}\n"
            )
        
        return "\n" + "="*50 + "\n".join(context_parts)
        
    except Exception as e:
        print(f"❌ Error extracting RAG context: {e}")
        return f"Error retrieving context from knowledge base: {str(e)}"


def run_guardrail_check(user_message: str, session_id: str) -> dict:
    """
    Run guardrail validation on user input and return detailed results.
    
    Args:
        user_message: The user's input query
        session_id: Session identifier
        
    Returns:
        dict: Guardrail validation results with timing and status
    """
    guardrail_start = time.time()
    guardrail_result = {
        "status": "failed",
        "validation_time": 0,
        "checks_performed": [],
        "security_verdict": "",
        "error": None
    }
    
    try:
        print(f"🔒 Running guardrail validation for session {session_id}")
        
        # Initialize guardrail
        guardrail = GuardrailRunnable()
        
        # Prepare input data in the format expected by guardrails
        input_data = {
            "input": json.dumps({
                "input": user_message,
                "history": []  # Could be extended to include chat history
            })
        }
        
        # Track checks performed
        guardrail_result["checks_performed"] = [
            "empty_input_check",
            "length_check", 
            "static_rules_check",
            "contextual_keywords_check",
            "semantic_context_check"
        ]
        
        # Run guardrail validation
        validated_input = guardrail.invoke(input_data)
        
        # If we get here, validation passed
        guardrail_result["status"] = "passed"
        guardrail_result["security_verdict"] = "SAFE"
        guardrail_result["validation_time"] = time.time() - guardrail_start
        
        print(f"✅ Guardrail validation passed in {guardrail_result['validation_time']:.3f} seconds")
        
        return guardrail_result
        
    except ValueError as ve:
        # Guardrail validation failed
        guardrail_result["status"] = "failed"
        guardrail_result["error"] = str(ve)
        guardrail_result["security_verdict"] = "UNSAFE"
        guardrail_result["validation_time"] = time.time() - guardrail_start
        
        print(f"❌ Guardrail validation failed: {ve}")
        raise ve  # Re-raise to stop processing
        
    except Exception as e:
        # Unexpected error during validation
        guardrail_result["status"] = "error"
        guardrail_result["error"] = f"Unexpected guardrail error: {str(e)}"
        guardrail_result["security_verdict"] = "UNKNOWN"
        guardrail_result["validation_time"] = time.time() - guardrail_start
        
        print(f"⚠️ Guardrail validation error: {e}")
        # Continue processing despite guardrail error (fail-open approach)
        return guardrail_result


def save_agent_session_log(
    session_id: str,
    user_message: str,
    guardrail_result: dict,
    rag_context: str,
    agent_scratchpad: list,
    tool_executions: list,
    final_response: str,
    processing_time: float,
    iterations: int,
    conversation_history: list = None
):
    """
    Save the complete agent session log to a text file.
    
    Args:
        session_id: Unique session identifier
        user_message: Original user query
        guardrail_result: Guardrail validation result and timing
        rag_context: Context retrieved from knowledge base
        agent_scratchpad: List of agent reasoning steps
        tool_executions: List of tool calls and their results
        final_response: Final response to user
        processing_time: Time taken to process the request
        iterations: Number of agent iterations
        conversation_history: Previous messages in the conversation (optional)
    """
    try:
        # Create logs directory if it doesn't exist
        logs_dir = "agent_data/session_logs"
        os.makedirs(logs_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"agent_session_{session_id}_{timestamp}.txt"
        filepath = os.path.join(logs_dir, filename)
        
        # Prepare log content
        log_content = f"""
{'='*80}
AGENT SESSION LOG
{'='*80}
Timestamp: {datetime.now().isoformat()}
Session ID: {session_id}
Processing Time: {processing_time:.2f} seconds
Agent Iterations: {iterations}

{'='*80}
USER QUERY
{'='*80}
{user_message}

{'='*80}
CONVERSATION HISTORY
{'='*80}
"""
        
        # Add conversation history
        if conversation_history and len(conversation_history) > 0:
            log_content += f"Previous messages in conversation: {len(conversation_history)}\n\n"
            # Show last few messages for context
            recent_history = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
            for i, msg in enumerate(recent_history):
                if isinstance(msg, dict):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")[:200] + ("..." if len(msg.get("content", "")) > 200 else "")
                    log_content += f"[{role.upper()}]: {content}\n\n"
                else:
                    log_content += f"[INVALID MSG FORMAT]: {str(msg)[:100]}...\n\n"
        else:
            log_content += "No previous conversation history.\n"

        log_content += f"""
{'='*80}
GUARDRAIL VALIDATION
{'='*80}
Status: {guardrail_result.get('status', 'Unknown')}
Validation Time: {guardrail_result.get('validation_time', 'N/A')} seconds
Checks Performed: {', '.join(guardrail_result.get('checks_performed', []))}
Security Verdict: {guardrail_result.get('security_verdict', 'N/A')}
{f"Error: {guardrail_result.get('error', '')}" if guardrail_result.get('error') else "✅ All security checks passed"}

{'='*80}
RAG CONTEXT RETRIEVED
{'='*80}
{rag_context}

{'='*80}
AGENT REASONING & SCRATCHPAD
{'='*80}
"""
        
        # Add scratchpad content
        if agent_scratchpad:
            for i, msg in enumerate(agent_scratchpad, 1):
                if hasattr(msg, 'content'):
                    log_content += f"\nStep {i} [{type(msg).__name__}]:\n{msg.content}\n{'-'*50}"
                else:
                    log_content += f"\nStep {i}:\n{str(msg)}\n{'-'*50}"
        else:
            log_content += "\nNo scratchpad entries recorded.\n"
        
        log_content += f"""

{'='*80}
TOOL EXECUTIONS
{'='*80}
"""
        
        # Add tool execution details
        if tool_executions:
            for i, execution in enumerate(tool_executions, 1):
                log_content += f"\nTool Execution {i}:\n{execution}\n{'-'*50}"
        else:
            log_content += "\nNo tools were executed.\n"
        
        log_content += f"""

{'='*80}
FINAL RESPONSE
{'='*80}
{final_response}

{'='*80}
END OF SESSION LOG
{'='*80}
"""
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(log_content)
        
        print(f"📝 Session log saved to: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"❌ Error saving session log: {e}")
        return None


# =============================================================
# PROMPT TEMPLATE (supports memory + scratchpad)
# =============================================================
prompt_3 = """
You are an intelligent ORCHESTRATOR that selects and sequences tools to satisfy user requests. 
You DO NOT generate content yourself — you plan which tools should produce the content.

YOUR JOB:
Given a user request + RAG context, output a JSON execution plan describing:
1. Which tools must be called
2. In what order
3. With what parameters
4. When multiple tools are needed, chain them correctly
5. Only include tools necessary for the user request

GENERAL RULES FOR TOOL SELECTION:

1. DEFAULT TOOL:
   → normal_content_tool
   Use this for ANY informational, explanatory, study-oriented, or content-generation query
   unless another tool is explicitly more appropriate.

2. WEB SEARCH:
   → web_search_tool
   Use ONLY when the user asks for:
     - “latest”, “current”, “recent”, “update”
     - “search the web”, “internet”, “online”
     - OR when RAG explicitly does NOT contain relevant information.

3. DOCUMENT CREATION:
   → document_export_tool
   If the user wants:
     - PDF
     - DOCX
     - PPTX
   ALWAYS include it as the *next* tool after content is generated.

4. EMAIL:
   → send_email_tool
   Always LAST in the chain, after document creation (if applicable).

5. VIDEO CONTENT:
   → video_generation_tool
   Only when the user explicitly asks for a video.

6. FLASHCARDS:
   → flashcard_generation_tool
   Only when the user wants “flashcards”, “study cards”, “revision cards”, “interactive flashcards”.

7. QUIZZES — IMPORTANT (NEW BEHAVIOR):
   Correctly distinguish quiz TYPE:

   a. **TEXT-BASED QUIZ (exportable)**  
      Keywords: “MCQ”, “questions”, “generate a quiz”, “10 questions”, “create quiz PDF”  
      → FIRST use normal_content_tool to generate the quiz content (uses RAG automatically)  
      → THEN document_export_tool if user wants export  
      → THEN send_email_tool if email requested  

   b. **INTERACTIVE QUIZ**  
      Keywords: “interactive quiz”, “test me”, “ask me questions”, “quiz game”  
      → use interactive_quiz_tool  
      (NEVER export these as PDF unless explicitly asked)

   c. **FLASHCARD QUIZ**  
      Keywords: “flashcards quiz”, “flashcard mode”, “study-style quiz”  
      → use flashcard_generation_tool

   ALWAYS infer the correct quiz type before choosing a tool.

8. MULTI-STEP REQUESTS:
   Always follow this order:
      1. Generate content (normal_content_tool / quiz / flashcards / interactive / video)
      2. Export content → document_export_tool (if requested)
      3. Email content → send_email_tool (if requested)

9. PARAMETER CLARIFICATION:
   If a required parameter is missing (topic, number of questions, format, email), 
   infer the most reasonable default:
      - topic: extract from request or use conversation topic
      - num_questions: default = 5 (ask if not mentioned)
      - difficulty: default = “medium”
      - format: pdf
      - email: do NOT fabricate; require explicit address. Ask if not provided.

10. IMPORTANT:
    The RAG context contains extensive knowledge on the Constitution of India.
    ALWAYS use normal_content_tool BEFORE considering web_search_tool,
    unless user explicitly requests latest/recent/web results.

FINAL REQUIREMENTS:
- Always output STRICT JSON
- No chat, no commentary, no markdown
- If no tool is needed → direct_response should contain the answer
- Ask as many questions as possible from the user if some arguments cannot be deducted for the tool calling. Do not assume
"""


new_prompt = """
You are an intelligent ORCHESTRATOR that decides when and how to call tools to satisfy user queries. Follow these instructions carefully:

TOOLS AVAILABLE:
1. normal_content_tool: Answer general questions using internal knowledge (RAG-based). This tool receives rag_context automatically and should be your DEFAULT choice.
2. web_search_tool: Answer questions using Internet search results for up-to-date information. USE ONLY when explicitly requested by user OR when RAG context is insufficient.
3. document_export_tool: Export content as PDF, DOCX, or PPTX documents.
4. send_email_tool: Send emails with content or documents.
5. video_generation_tool: Create educational videos (2–2.5 minutes) on constitutional topics.
6. flashcard_generation_tool: Create interactive flashcards for studying topics.
7. interactive_quiz_tool: Create quizzes with multiple question types, scoring, and immediate feedback.

REASONING RULES:
1. Think step-by-step using a scratchpad before deciding which tool(s) to call.
2. Use chat_history to remember prior conversation context.
3. Use ReAct-style tool calls when generating content.
4. After all tool calls, provide a clear FINAL_ANSWER in natural language.
5. Tool selection guidelines (PRIORITY ORDER):
   - DEFAULT: Use normal_content_tool for ALL questions (receives comprehensive RAG context automatically)
   - ONLY use web_search_tool when:
     * User explicitly asks for "latest", "current", "recent", "web search", "internet", or "online" information
     * User asks about events after your knowledge cutoff
     * RAG context shows "No relevant context found" or is clearly insufficient
   - Study/revision material → flashcard_generation_tool
   - Knowledge testing/quizzes → interactive_quiz_tool
   - Video content → video_generation_tool
   - Content in a file → document_export_tool
   - Content emailed → send_email_tool
6. For multi-step requests:
   - First generate content (normal content, flashcards, quizzes, or video)
   - Then optionally export to a file using document_export_tool
   - Then optionally send via email using send_email_tool
7. Always clarify missing parameters (topic, number of questions, file type, email address) before calling a tool.
IMPORTANT: The RAG context provided contains extensive knowledge about Indian Constitution and Rights. TRUST this context first and use normal_content_tool as your primary choice unless the user explicitly requests web search or the context is clearly inadequate.

IMPORTANT: The RAG context provided contains extensive knowledge about Indian Constitution and Rights. TRUST this context first and use normal_content_tool as your primary choice unless the user explicitly requests web search or the context is clearly inadequate.

EXAMPLE WORKFLOWS:
- User: "Explain Article 21" → Use normal_content_tool (RAG has this info)
- User: "Latest Supreme Court judgment on Article 21" → Use web_search_tool (explicitly asks for latest)
- User: "I want flashcards on the Constitution" → Use flashcard_generation_tool with RAG context
- User: "Search online for recent constitutional amendments" → Use web_search_tool (explicitly asks for web search)

- User: "I want flashcards on the Constitution and emailed to me as a PDF."
- Orchestrator:
   1. Recognizes request is for study material → call flashcard_generation_tool (receives rag_context automatically).
   2. Recognizes request for export → call document_export_tool with PDF format.
   3. Recognizes request for email → call send_email_tool with the exported PDF.
   4. Returns FINAL_ANSWER confirming flashcards creation, export, and email delivery.
Always prioritize RAG-based responses unless explicitly told otherwise or context is insufficient.


Always follow these rules to decide which tool(s) to call, handle multi-step requests, and produce a clear final response.

"""

old_prompt = """
You are an intelligent orchestrator that decides when to call tools.

Tools available:
- normal_content_tool (To answer the user query based out of rag)
- web_search_tool (To answer user query based out of Internet Search)
- document_export_tool (To export content as PDF, DOCX, or PPTX documents)
- send_email_tool (To send emails with content or documents)
- video_generation_tool (To create educational videos about constitutional topics)
- flashcard_generation_tool (To create interactive flashcards for studying constitutional topics)
- interactive_quiz_tool (To create interactive quizzes with immediate feedback and scoring)

Use the following reasoning rules:
1. Think step-by-step using your scratchpad.
2. Use chat_history to remember past conversation.
3. Output ReAct-style tool calls when needed.
4. After tool calls, give a FINAL_ANSWER.
5. For video requests, use video_generation_tool to create 2-2.5 minute educational videos.
6. For flashcard requests, use flashcard_generation_tool to create interactive Q&A study cards.
7. For quiz/test requests, use interactive_quiz_tool to create quizzes with multiple question types.
"""
ORCHESTRATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", new_prompt),

    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("scratchpad"),
])


# =============================================================
# Build LCEL Agent
# =============================================================

def create_orchestration_agent():
    llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",  # Using gemini-pro as it's stable and available
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.1
        )

    tools = [
        normal_content_tool,
        document_export_tool,
        send_email_tool,
        web_search_tool,
        video_generation_tool,
        flashcard_generation_tool,
        interactive_quiz_tool
    ]

    # Create tools dictionary for easy lookup
    tools_dict = {tool.name: tool for tool in tools}
    
    print(f"🔧 Registered tools: {list(tools_dict.keys())}")
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # Create the agent chain
    agent = ORCHESTRATION_PROMPT | llm_with_tools

    return agent, tools_dict


# =============================================================
# Main Orchestration Flow (LCEL + Memory)
# =============================================================

async def orchestrate_langchain_request(
    user_message: str,
    session_id: str,
    history: str,
    verbose: bool = False
) -> Dict[str, Any]:

    start = time.time()

    # ---- 1. GUARDRAIL VALIDATION (First line of defense) ----
    try:
        guardrail_result = run_guardrail_check(user_message, session_id)
        print(f"🔒 Guardrail validation completed: {guardrail_result['status']}")
    except ValueError as ve:
        # Guardrail failed - return error without further processing
        error_response = {
            "success": False,
            "response": str(ve),
            "agent_used": False,
            "processing_time": time.time() - start,
            "guardrail_failed": True,
            "iterations": 0
        }
        
        # Still save a log entry for failed requests
        try:
            failed_guardrail_result = {
                "status": "failed",
                "validation_time": time.time() - start,
                "checks_performed": ["input_validation"],
                "security_verdict": "UNSAFE",
                "error": str(ve)
            }
            
            log_filepath = save_agent_session_log(
                session_id=session_id,
                user_message=user_message,
                guardrail_result=failed_guardrail_result,
                rag_context="N/A - Request blocked by guardrails",
                agent_scratchpad=[],
                tool_executions=[],
                final_response=str(ve),
                processing_time=time.time() - start,
                iterations=0
            )
            error_response["log_filepath"] = log_filepath
        except Exception as log_error:
            print(f"❌ Error saving failed request log: {log_error}")
        
        return error_response

    # ---- 2. Load memory from history ----
    print(f"📋 Loading conversation history...")
    
    # Handle different history formats
    past_msgs = []
    if isinstance(history, str):
        # If history is a string, try to parse it as JSON or treat as empty
        try:
            import json
            past_msgs = json.loads(history) if history.strip() else []
        except (json.JSONDecodeError, AttributeError):
            print(f"⚠️ Could not parse history string, treating as empty: {history[:100]}...")
            past_msgs = []
    elif isinstance(history, list):
        past_msgs = history
    else:
        print(f"⚠️ Unexpected history type: {type(history)}, treating as empty")
        past_msgs = []

    print(f"📋 Loaded {len(past_msgs)} previous messages from history")

    # Convert to LangChain message format
    chat_history_lcel = []
    for i, msg in enumerate(past_msgs):
        try:
            if isinstance(msg, dict):
                if msg.get("role") == "user":
                    chat_history_lcel.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") in ["assistant", "ai"]:
                    chat_history_lcel.append(AIMessage(content=msg.get("content", "")))
                else:
                    print(f"⚠️ Unknown message role at index {i}: {msg.get('role')}")
            else:
                print(f"⚠️ Invalid message format at index {i}: {type(msg)}")
        except Exception as e:
            print(f"⚠️ Error processing message at index {i}: {e}")
            continue

    print(f"✅ Converted {len(chat_history_lcel)} messages to LangChain format")

    # ---- 3. Extract RAG Context ----
    print(f"🔍 Extracting RAG context for: {user_message}")
    rag_context = extract_rag_context(user_message, top_k=5)
    print(f"✅ RAG context extracted (length: {len(rag_context)} chars)")
    if verbose:
        print(f"📄 RAG Context Preview: {rag_context[:200]}...")

    # ---- Build agent ----
    agent, tools_dict = create_orchestration_agent()

    # ---- Agent Input (now includes RAG context + conversation history) ----
    # Enhance user message with RAG context and conversation summary for better tool decision making
    history_context = ""
    if chat_history_lcel:
        # Create a brief summary of recent conversation for context
        recent_messages = chat_history_lcel[-4:]  # Last 4 messages for context
        history_summary = []
        for msg in recent_messages:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            history_summary.append(f"{role}: {content_preview}")
        
        history_context = f"""

Recent Conversation Context:
{chr(10).join(history_summary)}
"""

    enhanced_input = f"""User Query: {user_message}

Available Context from Knowledge Base:
{rag_context}{history_context}

Based on this context, conversation history, and the user's query, determine the appropriate tool(s) to use."""

    agent_input = {
        "input": enhanced_input,
        "chat_history": chat_history_lcel,
        "scratchpad": []
    }

    print(f"Agent input prepared with RAG context integration")
    if chat_history_lcel:
        print(f"📋 Conversation context: {len(chat_history_lcel)} previous messages included")
    else:
        print(f"📋 No previous conversation history available")
    
    # ---- Run agent with tool execution loop ----
    max_iterations = 3
    iteration = 0
    tool_results = []
    tool_executions = []  # Store detailed tool execution info
    tool_result = ""
    
    while iteration < max_iterations:
        # Get response from agent
        result = agent.invoke(agent_input)
        
        print(f"Agent response: {result}")
        
        # Check if there are tool calls
        if hasattr(result, 'tool_calls') and result.tool_calls:
            print(f"Tool calls found: {result.tool_calls}")
            
            # Execute each tool call
            for tool_call in result.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                
                print(f"Executing tool: {tool_name} with args: {tool_args}")
                
                # Record tool execution details
                execution_start = time.time()
                execution_record = {
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "timestamp": datetime.now().isoformat(),
                    "execution_time": 0,
                    "success": False,
                    "result": None,
                    "error": None
                }
                
                if tool_name in tools_dict:
                    try:
                        # Inject RAG context for tools that need it
                        if tool_name in ["normal_content_tool", "flashcard_generation_tool", "interactive_quiz_tool", "video_generation_tool"]:
                            # These tools benefit from RAG context
                            if isinstance(tool_args, dict):
                                tool_args["rag_context"] = rag_context
                                print(f"🔍 Injected RAG context into {tool_name} (length: {len(rag_context)} chars)")
                            else:
                                tool_args = {
                                    "user_query": str(tool_args) if tool_args else user_message,
                                    "rag_context": rag_context
                                }
                                print(f"🔍 Created structured args with RAG context for {tool_name}")
                        
                        # Special handling for normal_content_tool which expects user_query + rag_context
                        if tool_name == "normal_content_tool":
                            if not isinstance(tool_args, dict) or "user_query" not in tool_args:
                                tool_args = {
                                    "user_query": user_message,
                                    "rag_context": rag_context
                                }
                                print(f"🔍 Structured normal_content_tool with proper parameters")
                        
                        # Special handling for video_generation_tool which expects topic + rag_context
                        elif tool_name == "video_generation_tool":
                            if not isinstance(tool_args, dict):
                                tool_args = {
                                    "topic": str(tool_args) if tool_args else user_message,
                                    "rag_context": rag_context
                                }
                            elif "rag_context" not in tool_args:
                                tool_args["rag_context"] = rag_context
                            print(f"🔍 Structured video_generation_tool with proper parameters: {tool_args.keys()}")
                        
                        # Update execution record with final args
                        execution_record["tool_args"] = tool_args
                        
                        # Execute the tool
                        tool_result = tools_dict[tool_name].invoke(tool_args)
                        execution_record["result"] = tool_result
                        execution_record["success"] = True
                        execution_record["execution_time"] = time.time() - execution_start
                        
                        print(f"Tool result: {tool_result}")
                        tool_results.append(f"Tool {tool_name} executed successfully: {tool_result}")
                        print(f"✅ Tool {tool_name} result: {tool_result}")
                        
                    except Exception as e:
                        execution_record["error"] = str(e)
                        execution_record["execution_time"] = time.time() - execution_start
                        error_msg = f"❌ Error executing {tool_name}: {str(e)}"
                        tool_results.append(error_msg)
                        print(error_msg)
                        # Print more debugging info
                        print(f"Tool args type: {type(tool_args)}")
                        print(f"Tool args content: {tool_args}")
                        import traceback
                        traceback.print_exc()
                else:
                    execution_record["error"] = f"Unknown tool: {tool_name}. Available tools: {list(tools_dict.keys())}"
                    error_msg = f"❌ Unknown tool: {tool_name}. Available tools: {list(tools_dict.keys())}"
                    tool_results.append(error_msg)
                    print(error_msg)
                
                # Store execution record
                tool_executions.append(execution_record)
            
            # Add tool results to scratchpad for next iteration
            agent_input["scratchpad"].extend([
                AIMessage(content=result.content),
                HumanMessage(content="Tool results: " + "\n".join(tool_results))
            ])
            
            iteration += 1
        else:
            # No more tool calls, we're done
            print("No tool calls found, finishing...")
            break
    
    # Get final response content

    """
    if isinstance(result, list):
        final_response = " ".join([msg.get("text", "") for msg in result if msg.get("type") == "text"])
    elif hasattr(result, "content"):
        final_response = result.content
    else:
        final_response = str(result)


    print(f"Final response: {final_response}")
    print(f"Type : {type(final_response)}")
    final_answer = final_response
    if type(final_response) == str:
        final_answer = final_response
    elif type(final_response) == list:
        final_answer = final_response[0]['text']
    
    # If final_answer is empty, try to get response from scratchpad
    if not final_answer or final_answer.strip() == "":
        print("⚠️ Final answer is empty, checking scratchpad for content...")
        if agent_input.get("scratchpad"):
            for msg in reversed(agent_input["scratchpad"]):
                if hasattr(msg, "content") and msg.content and msg.content.strip():
                    # Skip tool result messages, look for actual responses
                    if not msg.content.startswith("Tool results:"):
                        final_answer = msg.content.strip()
                        print(f"📋 Using content from scratchpad: {final_answer[:100]}...")
                        break
        
        # Last resort fallback
        if not final_answer or final_answer.strip() == "":
            final_answer = "I apologize, but I was unable to generate a proper response. Please try rephrasing your question."
            print("🔄 Using fallback response")
    """
    if isinstance(result, list):
        final_response = " ".join([msg.get("text", "") for msg in result if msg.get("type") == "text"])
    elif hasattr(result, "content"):
        final_response = result.content
    else:
        final_response = str(result)

    if type(final_response) == str:
        final_answer = final_response
    elif type(final_response) == list:
        final_answer = final_response[0]['text']
    
    if final_answer == "":
        final_answer = tool_result

    print(f"final answer: {final_answer}")
    
    # Save complete session log including guardrail results
    try:
        log_filepath = save_agent_session_log(
            session_id=session_id,
            user_message=user_message,
            guardrail_result=guardrail_result,
            rag_context=rag_context,
            agent_scratchpad=agent_input.get("scratchpad", []),
            tool_executions=tool_executions,
            final_response=final_answer,
            processing_time=time.time() - start,
            iterations=iteration,
            conversation_history=past_msgs
        )
        print(f"📝 Complete session log with guardrail data saved to: {log_filepath}")
    except Exception as e:
        print(f"❌ Error saving session log: {e}")
    
    append_to_memory(session_id, "user", user_message)
    append_to_memory(session_id, "assistant", final_answer)

    return {
        "success": True,
        "response": final_answer,
        "agent_used": True,
        "processing_time": time.time() - start,
        "iterations": iteration,
        "guardrail_passed": True,
        "log_filepath": log_filepath if 'log_filepath' in locals() else None
    }
