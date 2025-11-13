# auth.py
from fastapi import APIRouter, HTTPException
from utils.models import LoginRequest, LoginResponse
from utils.file_store import validate_user

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/login", response_model=LoginResponse)
async def login(req: LoginRequest):
    if validate_user(req.username, req.password):
        return LoginResponse(success=True, message=f"Welcome {req.username}!")
    raise HTTPException(status_code=401, detail="Invalid credentials")
