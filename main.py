from pathlib import Path
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Depends
from sqlalchemy.orm import Session
from utils.route_helper import auth_router
from utils.route_helper import router as sessions_router
from utils.route_helper import get_session_details_and_messages, upload_file_to_db, add_conversation_details_to_db
from db_models.crud_operations import get_db
from utils.guardrails import GuardrailRunnable
from utils.models import ChatRequest
from utils.gemini_chain import GeminiChatChain
from utils.langchain_orchestrator import orchestrate_langchain_request
import os
import shutil
import json

app = FastAPI(title="Subject Matter Expert - Indian Constituion and Rights")

app.include_router(auth_router)
app.include_router(sessions_router)

guardrail = GuardrailRunnable()
gemini_chain = GeminiChatChain()
def get_session(session_id: str) -> dict:
    SESSION_DIR = 'agent_data/sessions'
    path = Path(SESSION_DIR) / f"{session_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Session '{session_id}' not found.")
    with open(path, "r") as f:
        return json.load(f)
    
def save_session(session_data: dict):
    SESSION_DIR = 'agent_data/sessions'
    path = Path(SESSION_DIR) / f"{session_data['session_id']}.json"
    with open(path, "w") as f:
        json.dump(session_data, f, indent=2)


@app.on_event("startup")
async def init_data_dirs():
    os.makedirs("agent_data/sessions", exist_ok=True)

# @app.post("/chat")
# async def chat(request: ChatRequest, db: Session = Depends(get_db)): # <-- This stays async
#     """
#     Main chat endpoint - processes message with LangChain agent or Gemini chain
#     """
#     session_id = request.session_id
#     user_message = request.message
#     filepath=None
#     if request.filepath:
#         filepath = request.filepath

#     session_details = get_session_details_and_messages(session_id, db)

#     print(f"Session ID: {session_id}, Session: {session_details}")
#     if session_id not in session_details["session_id"]:
#         raise HTTPException(status_code=404, detail="Session not found")
    
#     username = session_details['username']
#     history = session_details["messages"]
    
#     # Check if user wants to use LangChain orchestration
#     use_langchain = "use langchain" in user_message.lower() or "orchestrate" in user_message.lower()
        
#     try:
#         print(f"\n{'='*70}")
#         print(f"📨 Incoming chat request (LangChain: {use_langchain})")
#         print(f"{'='*70}\n")

#         try:
#             guardrail_payload = json.dumps({
#                 "input": user_message
#             })

#             # guardrail.invoke() is synchronous → run safely in thread
#             await asyncio.to_thread(
#                 guardrail.invoke,
#                 {"input": guardrail_payload}
#             )

#         except Exception as e:
#             # Reject unsafe input before planner, RAG, tools
#             raise HTTPException(status_code=400, detail=str(e))
    
#         user_query_details = {"role": "user", "content": user_message}
#         await add_conversation_details_to_db(session_id, user_query_details, db)
        
#         # Step 1: Get RAG context if needed
#         rag_context = None
        
#         # Step 2: Choose processing method
#         if use_langchain:
#             # Use LangChain orchestration with tool calling
#             try:
#                 print("🔗 Using LangChain orchestration with tool calling...")
#                 result = await process_with_production_langchain(user_message, history)
                
#                 if result["success"]:
#                     final_response = result["response"]
#                     if result["agent_used"]:
#                         tools_used = result.get("tools_executed", [])
#                         if tools_used:
#                             final_response += f"\n\n*Processed using LangChain agent with tools: {', '.join(tools_used)}*"
#                         else:
#                             final_response += "\n\n*Processed using LangChain agent orchestration*"
#                     else:
#                         final_response += "\n\n*Processed using LangChain fallback mode*"
#                 else:
#                     final_response = result["response"]
                
#             except Exception as e:
#                 print(f"❌ LangChain orchestration failed: {e}")
#                 final_response = "I apologize, but I encountered an error with the orchestration system. Please try again."
        
#         else:
#             # Use existing Gemini chain (default behavior)
#             try:
#                 result = await gemini_chain.get_response(
#                     user_message=user_message,
#                     history=history,
#                     rag_context=rag_context
#                 )
                
#                 if not result["success"]:
#                     raise Exception(result["response"])
                
#                 final_response = result["response"]
#                 intent = result["intent"]
#                 quiz_params = result.get("quiz_params")
                
#                 # If quiz is detected, process the quiz generation
#                 if intent == "quiz_generation" and quiz_params:
#                     print(f"🎯 Quiz generation detected! Parameters: {quiz_params}")
                    
#                     try:
#                         # Use the quiz generation tool directly (not through ainvoke which might have issues)
#                         from utils.quiz_generator import QuizGenerator
#                         quiz_generator = QuizGenerator()
                        
#                         # Process the quiz request
#                         quiz_result = await quiz_generator.process_quiz_request(
#                             quiz_params=quiz_params,
#                             rag_context=rag_context or ""
#                         )
                        
#                         if quiz_result["success"]:
#                             if quiz_result.get("exported"):
#                                 # If documents were generated, provide a summary response
#                                 questions_doc = quiz_result.get("questions_document", {})
#                                 answers_doc = quiz_result.get("answers_document", {})
                                
#                                 final_response = f"""📝 **Quiz Generated Successfully!**

# 🎯 **Quiz Details:**
# • **Topic:** {quiz_params.get('topic', 'General')}
# • **Questions:** {quiz_params.get('num_questions', 5)}
# • **Difficulty:** {quiz_params.get('difficulty', 'medium').title()}
# • **Type:** {quiz_params.get('question_type', 'mcq')}

# 📄 **Documents Created:**
# • **Questions Document:** {questions_doc.get('filename', 'Quiz questions file')}
# • **Answer Key:** {answers_doc.get('filename', 'Quiz answers file')}

# 💡 **What's included:**
# - Complete quiz with instructions at the top
# - Separate answer key with explanations
# - Ready for classroom use or self-assessment

# ✅ **Your quiz files are ready for use!**

# {quiz_result.get('questions_only', '')[:800]}...

# *Note: Full content has been exported to PDF documents.*"""
#                             else:
#                                 # For inline quiz, provide a more structured response
#                                 questions_content = quiz_result.get('questions_only', '')
                                
#                                 # Truncate if too long for UI but keep structure
#                                 if len(questions_content) > 2000:
#                                     lines = questions_content.split('\n')
#                                     truncated_lines = lines[:30]  # Keep first 30 lines
#                                     truncated_content = '\n'.join(truncated_lines)
                                    
#                                     final_response = f"""📝 **Quiz Generated Successfully!**

# {truncated_content}

# *[Quiz continues... Total content is longer than displayed here]*

# 💡 **Tip:** To get the complete quiz as separate PDF documents, ask me to "export this quiz as PDF" or include "export" in your request!"""
#                                 else:
#                                     final_response = f"""📝 **Quiz Generated Successfully!**

# {questions_content}

# 💡 **Tip:** To get this quiz as separate PDF documents, ask me to "export this quiz as PDF"!"""
                            
#                             print(f"📝 Quiz generated successfully - Response length: {len(final_response)}")
#                         else:
#                             final_response = f"❌ Failed to generate quiz: {quiz_result.get('error', 'Unknown error')}"
                        
#                     except Exception as quiz_error:
#                         print(f"❌ Quiz generation failed: {quiz_error}")
#                         import traceback
#                         traceback.print_exc()
#                         final_response = f"❌ I encountered an error while generating the quiz: {str(quiz_error)}. Please try again with simpler parameters."
                
#             except Exception as e:
#                 print(f"Error with Gemini chain: {e}")
#                 final_response = "I apologize, but I'm having trouble processing your request right now. Please try again."
        
#         print(f"Final response length: {len(final_response)} characters")
#         print(f"Final response preview: {final_response[:200]}...")
                
#         # 6. Store the user message + AI response
#         ai_answer = {"role": "assistant", "content": final_response}
#         session_details["messages"].append(user_query_details)
#         session_details["messages"].append(ai_answer)

#         # Save updated session
#         save_session(session_details)
#         await add_conversation_details_to_db(session_id, ai_answer, db)
        
#         return {
#             "response": final_response,
#             "session_id": session_id,
#             "message_count": len(session_details["messages"])
#         }
        
#     except Exception as e:
#         print(f"\n❌ Error processing chat: {e}")
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.post("/chat/langchain")
async def chat_langchain(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Dedicated endpoint for LangChain agent orchestration
    """
    session_id = request.session_id
    user_message = request.message
    
    session_details = get_session_details_and_messages(session_id, db)
    
    if session_id not in session_details["session_id"]:
        raise HTTPException(status_code=404, detail="Session not found")
    
    history = session_details["messages"]
    
    try:
        print(f"\n🔗 LangChain Orchestration Request")
        print(f"Message: {user_message}")
        
        # Store user message
        user_query_details = {"role": "user", "content": user_message}
        await add_conversation_details_to_db(session_id, user_query_details, db)
        
        # Process with LangChain orchestration
        # result = await process_with_production_langchain(user_message, history)
        result = await orchestrate_langchain_request(user_message, session_id, history)
        
        if result["success"]:
            final_response = result["response"]
            
            # Add metadata about processing
            processing_info = "\n\n---\n**Processing Details:**\n"
            if result["agent_used"]:
                tools_used = result.get("tools_executed", [])
                if tools_used:
                    processing_info += f"✅ **LangChain Agent:** Successfully executed tools: {', '.join(tools_used)}\n"
                else:
                    processing_info += "✅ **LangChain Agent:** Successfully processed request\n"
            else:
                processing_info += "⚠️ **Fallback Mode:** Used fallback processing\n"
            
            final_response += processing_info
        else:
            final_response = f"❌ {result['response']}"
        
        # Store AI response
        ai_answer = {"role": "assistant", "content": final_response}
        session_details["messages"].append(user_query_details)
        session_details["messages"].append(ai_answer)
        
        save_session(session_details)
        await add_conversation_details_to_db(session_id, ai_answer, db)
        
        return {
            "response": final_response,
            "session_id": session_id,
            "orchestration_used": True,
            "agent_used": result.get("agent_used", False),
            "message_count": len(session_details["messages"])
        }
        
    except Exception as e:
        print(f"❌ Error in LangChain orchestration: {e}")
        raise HTTPException(status_code=500, detail=f"Orchestration error: {str(e)}")

@app.post("/generate-video")
async def generate_video(
    topic: str = Form(...),
    duration: float = Form(default=150.0),
    style: str = Form(default="educational"),
    include_examples: bool = Form(default=True),
    session_id: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    Generate educational video about constitutional topics
    """
    try:
        print(f"🎬 Video generation request: {topic}")
        
        # Import video tool
        from langchain_tools.video_generator.video_generation_tool import video_tool_instance
        
        # Generate video
        result = video_tool_instance.generate_video(
            topic=topic,
            duration=duration,
            style=style,
            include_examples=include_examples
        )
        
        # Store generation record in session
        if result['success']:
            generation_record = {
                "role": "system",
                "content": f"Video generated: {topic} ({result.get('video_info', {}).get('duration_seconds', duration)}s)"
            }
            await add_conversation_details_to_db(session_id, generation_record, db)
        
        return result
        
    except Exception as e:
        print(f"❌ Error in video generation endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Video generation error: {str(e)}")

UPLOAD_DIR = "agent_data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_file(username: str = Form(...), session_id: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Upload and store a file for a user's session.
    """
    user_dir = os.path.join(UPLOAD_DIR, username)
    os.makedirs(user_dir, exist_ok=True)
    print(f"User dir: {user_dir}")
    save_path = os.path.join(user_dir, f"{session_id}_{file.filename}")

    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file.file.seek(0)
        await upload_file_to_db(username, session_id, file, db)
        return {"message": "File uploaded successfully.", "file_path": save_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")