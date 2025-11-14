# auth.py
from fastapi import APIRouter, Depends, HTTPException
from requests import Session
from utils.models import LoginRequest, LoginResponse
from utils.file_store import validate_user
from db_models.crud_operations import get_db
from db_models.helper import authenticate_user

router = APIRouter(prefix="/auth", tags=["auth"])

# @router.post("/login", response_model=LoginResponse)
# async def login(req: LoginRequest):
#     if validate_user(req.username, req.password):
#         return LoginResponse(success=True, message=f"Welcome {req.username}!")
#     raise HTTPException(status_code=401, detail="Invalid credentials")

@router.post("/login")
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """
    Handles user login by authenticating against the database.
    """
    print(f"Attempting login for user: {request.username}")
    
    # Use the correct function to check the username and hashed password
    user = authenticate_user(db, request.username, request.password)
    
    if not user:
        print("❌ Authentication failed: Invalid credentials.")
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    print(f"✅ Login successful for: {user.username}")
    return LoginResponse(success=True, message=f"Welcome {request.username}!")