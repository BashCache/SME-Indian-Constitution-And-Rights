from pathlib import Path
import re
from typing import Any, Dict
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from utils.auth import router as auth_router
from utils.sessions import router as sessions_router
from utils.sessions import get_session_details_and_messages, upload_file_to_db, add_conversation_details_to_db
from utils.guardrails import GuardrailRunnable
from utils.orchestrator import get_orchestrator, get_unified_plan, ToolPlan
from utils.models import ChatRequest
from utils.agent_tools import get_rag_answer, document_tool, email_tool
from utils.file_store import get_session
import os
import asyncio
import shutil
import json

app = FastAPI(title="Subject Matter Expert - Indian Constituion and Rights")

app.include_router(auth_router)
app.include_router(sessions_router)

guardrail = GuardrailRunnable()

TOOL_EXECUTOR_MAP = {
    "document_tool": document_tool,
    "email_tool": email_tool,
}

# ============================================
# HELPER: PLACEHOLDER SUBSTITUTION
# ============================================
# (This function is synchronous and fast, no async needed)
def substitute_placeholders(args: Dict[str, Any], rag_result: str, last_answer: str, step_outputs: Dict[int, Any]) -> Dict[str, Any]:
    # ... (function content is correct, no changes needed)
    substituted_args = {}
    for key, value in args.items():
        if isinstance(value, str):
            if value == "[[RAG_RESULT]]":
                substituted_args[key] = rag_result
            elif value == "[[LAST_ANSWER]]":
                substituted_args[key] = last_answer
            else:
                match = re.match(r"\[\[STEP_(\d+)_RESULT\]\]", value)
                if match:
                    step_num = int(match.group(1))
                    substituted_args[key] = step_outputs.get(step_num, None)
                else:
                    substituted_args[key] = value
        elif isinstance(value, dict):
            substituted_args[key] = substitute_placeholders(value, rag_result, last_answer, step_outputs)
        else:
            substituted_args[key] = value
    return substituted_args

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

@app.post("/chat")
async def chat(request: ChatRequest): # <-- This stays async
    """
    Main chat endpoint - processes message with LangChain agent
    """
    session_id = request.session_id
    user_message = request.message
    filepath=None
    if request.filepath:
        filepath = request.filepath
    
    # session = get_session(session_id)
    session_details = get_session_details_and_messages(session_id)

    print(f"Session ID: {session_id}, Session: {session_details}")
    if session_id not in session_details["session_id"]:
        raise HTTPException(status_code=404, detail="Session not found")
    
    username = session_details['username']
    history = session_details["messages"]
    history = session_details["messages"]
    
    # Get orchestrator (this is still fast, just inits the class)
    orchestrator = get_orchestrator()
    
    try:
        print(f"\n{'='*70}")
        print(f"📨 Incoming chat request")
        print(f"{'='*70}\n")

        try:
            guardrail_payload = json.dumps({
                "input": user_message
            })

            # guardrail.invoke() is synchronous → run safely in thread
            await asyncio.to_thread(
                guardrail.invoke,
                {"input": guardrail_payload}
            )

        except Exception as e:
            # Reject unsafe input before planner, RAG, tools
            raise HTTPException(status_code=400, detail=str(e))
    
        user_query_details = {"role": "user", "content": user_message}
        await add_conversation_details_to_db(session_id, user_query_details)
        
        # 1. Call your synchronous Tool Planner
        print("Step 1: Calling Tool Planner...")
        plan: ToolPlan = await asyncio.to_thread(
            get_unified_plan,
            query=user_message,
            history=history
        )

        # 2. Check for simple chat
        rag_answer = ""
        action_log = []
        if plan.chat_response:
            print(f"Step 2: Simple chat response: {plan.chat_response}")
            final_response = plan.chat_response
            rag_answer = final_response
        else:            
            # 3. Execute RAG
            print(f"Step 2: Executing RAG (Source: {plan.rag_source})")
            if plan.rag_source and plan.rag_source != "none":
                rag_answer = await asyncio.to_thread(
                    get_rag_answer,
                    query=user_message,
                    source=plan.rag_source,
                    username=username,
                    session_id=session_id,
                    history=history,
                    filepath=filepath
                )
                if rag_answer:
                    action_log.append(f"Retrieved information using {plan.rag_source}.")

        # 4. Execute Action Plan
        print(f"Step 3: Executing Action Plan ({len(plan.execution_plan or [])} steps)")
        step_outputs: Dict[int, Any] = {}
        
        if plan.execution_plan:
            last_answer = history[-1]['content'] if history else ""
            
            for i, step in enumerate(plan.execution_plan, 1):
                tool_name = step.name
                tool_args = step.args
                
                print(f"  -> Executing Step {i}: {tool_name}")
                
                if tool_name not in TOOL_EXECUTOR_MAP:
                    print(f"  ❌ Error: Unknown tool '{tool_name}'")
                    action_log.append(f"Error: Unknown tool '{tool_name}'")
                    continue
                
                print(f"Tool args: {tool_args}")
                # (This substitution is fast and synchronous)
                # substituted_args = substitute_placeholders(tool_args, rag_answer, last_answer, step_outputs)
                
                executor_func = TOOL_EXECUTOR_MAP[tool_name]
                # print(substituted_args)
                # # <--- ASYNC REQUIRED HERE
                # # We run the blocking tool (e.g., file I/O) in a thread
                # result = await asyncio.to_thread(
                #     executor_func.invoke,
                #     substituted_args
                # )

                if tool_name == "document_tool":
                    input_args = {
                        'content': rag_answer,
                        'document_type': tool_args['document_type'],
                        'title': tool_args['title']
                    }
                    result = await asyncio.to_thread(
                        executor_func.invoke,
                        input_args
                    )
                # TODO: To fix the filename properly and not based on step_outputs
                elif tool_name == "email_tool":
                    input_args = {
                        'filename': step_outputs[i-1],
                        'recipient': tool_args['recipient']
                    }
                    result = await asyncio.to_thread(
                        executor_func.invoke,
                        input_args
                    )
                print(f"  ✅ Step {i} result: {result}")
                action_log.append(str(result))
                step_outputs[i] = result
        
        # 5. Formulate final response
        if rag_answer and not action_log:
            final_response = rag_answer # RAG only
        else:
            final_response = f"{rag_answer}\n\n---\n*Actions taken: {', '.join(action_log)}*"

        print(f"Fnal response: {final_response}")
        response = "abc"
        
        print(f"\n💬 Response generated: {response[:200]}...")
        
        # 6. Store the user message + AI response
        ai_answer = {"role": "assistant", "content": final_response}
        session_details["messages"].append(user_query_details)
        session_details["messages"].append(ai_answer)

        # Save updated session
        save_session(session_details)
        await add_conversation_details_to_db(session_id, ai_answer)
        
        return {
            "response": response,
            "session_id": session_id,
            "message_count": len(session_details["messages"])
        }
        
    except Exception as e:
        print(f"\n❌ Error processing chat: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")
    
# @app.delete("/sessions/{session_id}")
# def delete_session(session_id: str):
#     try:
#         directory_path = "agent_data/sessions/"
#         filename = f"{session_id}.json"
#         filepath = os.path.join(directory_path, filename)
#         print(filepath)

#         if not os.path.exists(filepath):
#             raise HTTPException(status_code=404, detail="Session file not found.")

#         os.remove(filepath)

#         return {"message": f"Session '{session_id}' deleted successfully for user."}

    # except HTTPException:
    #     raise
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))
    
UPLOAD_DIR = "agent_data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_file(username: str = Form(...), session_id: str = Form(...), file: UploadFile = File(...)):
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
        await upload_file_to_db(username, session_id, file)
        return {"message": "File uploaded successfully.", "file_path": save_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")