from pydantic import BaseModel
from typing import Optional

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    success: bool
    message: str

class SessionCreateRequest(BaseModel):
    username: str
    title: str

class SessionResponse(BaseModel):
    session_id: str
    title: str
    created_at: str

class ChatRequest(BaseModel):
    session_id: str
    message: str
    filepath: Optional[str] = None
