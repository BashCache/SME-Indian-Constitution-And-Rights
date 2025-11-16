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


# =============================================================
# PROMPT TEMPLATE (supports memory + scratchpad)
# =============================================================

ORCHESTRATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
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

"""),

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

    # ---- Build agent ----
    agent, tools_dict = create_orchestration_agent()

    # ---- Agent Input ----
    agent_input = {
        "input": user_message,
        "chat_history": chat_history_lcel,
        "scratchpad": []
    }

    print(f"Agent input: {agent_input}")
    
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
                        # Execute the tool - handle both dict args and direct args
                        if isinstance(tool_args, dict):
                            tool_result = tools_dict[tool_name].invoke(tool_args)
                        else:
                            tool_result = tools_dict[tool_name].invoke({"args": tool_args})
                        
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
