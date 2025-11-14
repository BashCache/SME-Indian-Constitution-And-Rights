# sessions.py
from fastapi import APIRouter, HTTPException, Depends, File, UploadFile
from utils.models import SessionCreateRequest, SessionResponse
from utils.file_store import list_sessions, create_session, get_session
from db_models import crud_operations as crud
from requests import Session
from db_models.crud_operations import get_db
from db_models.models import UploadedDocument
import datetime

crud.ensure_db()
router = APIRouter(prefix="/sessions", tags=["sessions"])

# @router.get("/{username}")
# async def get_user_sessions(username: str):
#     return list_sessions(username)

@router.get("/{username}")
def get_user_sessions(username: str, db: Session = Depends(get_db)):
    return crud.list_sessions_by_username(db, username)

# @router.post("/create", response_model=SessionResponse)
# async def create_new_session(req: SessionCreateRequest):
#     session = create_session(req.username, req.title)
#     return SessionResponse(**session)

@router.post("/create")
def create_session_endpoint(req: SessionCreateRequest, db: Session = Depends(get_db)):
    s = crud.create_chat_session(db, req.username, req.title)
    return {
        "session_id": s.session_id,
        "title": s.title,
        "started_at": s.started_at.isoformat()
    }

@router.get("/get/{session_id}")
async def fetch_session(session_id: str):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

@router.delete("/{session_id}")
async def delete_session(session_id: str, db: Session = Depends(get_db)):
    crud.delete_session(db, session_id)
    return {"message": f"Session '{session_id}' deleted successfully for user."}

def get_session_details_and_messages(session_id: str, db: Session = Depends(get_db)):
    session_metadata_details = crud.fetch_session_by_sessionid(db, session_id)
    if session_metadata_details is None:
        raise HTTPException(status_code=404, detail="Session not found")
    session_conversation_deets = crud.fetch_conversation_messages(db, session_id)
    return {
        "session_id": session_metadata_details["session_id"],
        "title": session_metadata_details["title"],
        "username": session_id.split('_')[0],
        "messages": [
            {"role": m["role"], "content": m["content"]}
            for m in session_conversation_deets
        ]
    }

async def upload_file_to_db(
    username: str,
    session_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    user = crud.get_user_by_username(db, username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    extracted_text = None

    db_doc = crud.create_uploded_doc(db, file, session_id, file_bytes, user.id, extracted_text)
    print(f"DB doc: {db_doc.id}")
    return {
        "message": "File uploaded and stored successfully.",
        "document_id": db_doc.id,
        "filename": db_doc.filename,
        "size_bytes": len(file_bytes),
    }

async def add_conversation_details_to_db(session_id, conv_messages, db: Session = Depends(get_db)):
    try:
        # Save user message
        crud.save_conversation_message(
            db=db,
            session_id=session_id,
            role=conv_messages["role"],
            content=conv_messages["content"],
            metadata={}
        )
        print("✅ Conversation messages saved.")
    except Exception as e:
        print(f"❌ Error saving messages: {e}")
        raise
