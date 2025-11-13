# sessions.py
from fastapi import APIRouter, HTTPException
from utils.models import SessionCreateRequest, SessionResponse
from utils.file_store import list_sessions, create_session, get_session

router = APIRouter(prefix="/sessions", tags=["sessions"])

@router.get("/{username}")
async def get_user_sessions(username: str):
    return list_sessions(username)

@router.post("/create", response_model=SessionResponse)
async def create_new_session(req: SessionCreateRequest):
    session = create_session(req.username, req.title)
    return SessionResponse(**session)

@router.get("/get/{session_id}")
async def fetch_session(session_id: str):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session
