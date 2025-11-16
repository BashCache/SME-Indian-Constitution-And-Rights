# orchestrator.py

import os
import time
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
from mock_rag import RAGTool

def extract_rag_context(user_message: str, top_k: int = 3) -> str:
    try:
        rag_tool = RAGTool(model_key="mpnet")
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


# =============================================================
# PROMPT TEMPLATE (supports memory + scratchpad)
# =============================================================

new_prompt = """
You are an intelligent ORCHESTRATOR that decides when and how to call tools to satisfy user queries. Follow these instructions carefully:

TOOLS AVAILABLE:
1. normal_content_tool: Answer general questions using internal knowledge (RAG-based). This tool receives rag_context automatically. If not, then use web search tool
2. web_search_tool: Answer questions using Internet search results for up-to-date information.
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
5. Tool selection guidelines:
   - General answers/explanations with available context → normal_content_tool
   - Up-to-date/external info beyond knowledge base or information from rag context is not available→ web_search_tool
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

EXAMPLE WORKFLOW:
- User: "I want flashcards on the Constitution and emailed to me as a PDF."
- Orchestrator:
   1. Recognizes request is for study material → call flashcard_generation_tool (receives rag_context automatically).
   2. Recognizes request for export → call document_export_tool with PDF format.
   3. Recognizes request for email → call send_email_tool with the exported PDF.
   4. Returns FINAL_ANSWER confirming flashcards creation, export, and email delivery.

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

    # ---- Load memory ----
    past_msgs = history

    chat_history_lcel = []
    for msg in past_msgs:
        if msg["role"] == "user":
            chat_history_lcel.append(HumanMessage(content=msg["content"]))
        else:
            chat_history_lcel.append(AIMessage(content=msg["content"]))

    # ---- Extract RAG Context ----
    print(f"🔍 Extracting RAG context for: {user_message}")
    rag_context = extract_rag_context(user_message, top_k=3)
    print(f"✅ RAG context extracted (length: {len(rag_context)} chars)")
    if verbose:
        print(f"📄 RAG Context Preview: {rag_context[:200]}...")

    # ---- Build agent ----
    agent, tools_dict = create_orchestration_agent()

    # ---- Agent Input (now includes RAG context) ----
    # Enhance user message with RAG context for better tool decision making
    enhanced_input = f"""User Query: {user_message}

Available Context from Knowledge Base:
{rag_context}

Based on this context and the user's query, determine the appropriate tool(s) to use."""

    agent_input = {
        "input": enhanced_input,
        "chat_history": chat_history_lcel,
        "scratchpad": []
    }

    print(f"Agent input prepared with RAG context integration")
    
    # ---- Run agent with tool execution loop ----
    max_iterations = 3
    iteration = 0
    
    while iteration < max_iterations:
        # Get response from agent
        result = agent.invoke(agent_input)
        
        print(f"Agent response: {result}")
        
        # Check if there are tool calls
        if hasattr(result, 'tool_calls') and result.tool_calls:
            print(f"Tool calls found: {result.tool_calls}")
            
            # Execute each tool call
            tool_results = []
            for tool_call in result.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                
                print(f"Executing tool: {tool_name} with args: {tool_args}")
                
                if tool_name in tools_dict:
                    try:
                        # Inject RAG context for tools that need it
                        if tool_name in ["normal_content_tool", "flashcard_generation_tool", "interactive_quiz_tool"]:
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
                        
                        # Execute the tool
                        tool_result = tools_dict[tool_name].invoke(tool_args)
                        
                        tool_results.append(f"Tool {tool_name} executed successfully: {tool_result}")
                        print(f"✅ Tool {tool_name} result: {tool_result}")
                    except Exception as e:
                        error_msg = f"❌ Error executing {tool_name}: {str(e)}"
                        tool_results.append(error_msg)
                        print(error_msg)
                        # Print more debugging info
                        print(f"Tool args type: {type(tool_args)}")
                        print(f"Tool args content: {tool_args}")
                        import traceback
                        traceback.print_exc()
                else:
                    error_msg = f"❌ Unknown tool: {tool_name}. Available tools: {list(tools_dict.keys())}"
                    tool_results.append(error_msg)
                    print(error_msg)
            
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

    # LCEL can return string, dict, or AIMessage
    # Handle LCEL result which may be a list of dicts
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
    
    print(f"final answer: {final_answer}")
    # ---- Save memory ----
    append_to_memory(session_id, "user", user_message)
    append_to_memory(session_id, "assistant", final_response)

    return {
        "success": True,
        "response": final_answer,
        "agent_used": True,
        "processing_time": time.time() - start,
        "iterations": iteration
    }
