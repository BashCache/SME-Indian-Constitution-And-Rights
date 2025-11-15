# sessions.py
from fastapi import APIRouter, HTTPException, Depends, UploadFile
from sqlalchemy.orm import Session
from typing import Optional
from utils.models import LoginRequest, LoginResponse, SessionCreateRequest
from db_models.helper import authenticate_user
from db_models import crud_operations as crud
from db_models.crud_operations import get_db
from utils.extractor.file_extractor import FileExtractor
import os
import tempfile

crud.ensure_db()
router = APIRouter(prefix="/sessions", tags=["sessions"])
auth_router = APIRouter(prefix="/auth", tags=["auth"])

@auth_router.post("/login")
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

def extract_text_from_file_bytes(file_bytes: bytes, filename: str) -> Optional[str]:
    print(f"🔄 Starting text extraction for {filename}")
    extracted_text = None
    temp_file_path = None
    
    try:
        # Create a temporary file to work with the extractor
        file_extension = os.path.splitext(filename)[1]
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(file_bytes)
            temp_file.flush()
            temp_file_path = temp_file.name
            
        # Use FileExtractor to extract text
        extractor = FileExtractor()
        print(f"🚀 Starting extraction from: {temp_file_path}")
        extraction_result = extractor.extract_text(temp_file_path)
        
        if extraction_result and extraction_result.content:
            extracted_text = extraction_result.content
            print(f"✅ Extracted {len(extracted_text)} characters from {filename}")
        else:
            error_msg = extraction_result.metadata.get("error", "Unknown extraction error") if extraction_result and extraction_result.metadata else "Failed to extract text"
            print(f"⚠️ Text extraction failed for {filename}: {error_msg}")
            
    except Exception as e:
        print(f"❌ Error during text extraction for {filename}: {e}")
        import traceback
        print(f"🔍 Full traceback: {traceback.format_exc()}")
        
    finally:
        if temp_file_path:
            try:
                os.unlink(temp_file_path)
                print(f"🗑️ Cleaned up temporary file: {temp_file_path}")
            except Exception as cleanup_error:
                print(f"⚠️ Failed to cleanup temp file {temp_file_path}: {cleanup_error}")
                
    return extracted_text

@router.get("/{username}")
def get_user_sessions(username: str, db: Session = Depends(get_db)):
    return crud.list_sessions_by_username(db, username)

@router.post("/create")
def create_session_endpoint(req: SessionCreateRequest, db: Session = Depends(get_db)):
    s = crud.create_chat_session(db, req.username, req.title)
    return {
        "session_id": s.session_id,
        "title": s.title,
        "started_at": s.started_at.isoformat()
    }

# @router.get("/get/{session_id}")
# async def fetch_session(session_id: str):
#     session = get_session(session_id)
#     if not session:
#         raise HTTPException(status_code=404, detail="Session not found")
#     return session

@router.delete("/{session_id}")
async def delete_session(session_id: str, db: Session = Depends(get_db)):
    crud.delete_session(db, session_id)
    return {"message": f"Session '{session_id}' deleted successfully for user."}

@router.patch("/{session_id}/rename")
async def rename_session(session_id: str, new_title: dict, db: Session = Depends(get_db)):
    title = new_title.get("title", "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title cannot be empty")
    
    success = crud.rename_session(db, session_id, title)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"message": f"Session '{session_id}' renamed to '{title}' successfully."}

def get_session_details_and_messages(session_id: str, db: Session):
    session_metadata_details = crud.fetch_session_by_sessionid(db, session_id)
    if session_metadata_details is None:
        raise HTTPException(status_code=404, detail="Session not found")
    session_conversation_deets = crud.fetch_conversation_messages(session_id)
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
    file: UploadFile,
    db: Session,
):
    print(f"🚀 Starting upload_file_to_db for user: {username}, session: {session_id}, file: {file.filename}")
    
    try:
        user = crud.get_user_by_username(db, username)
        if not user:
            print(f"❌ User not found: {username}")
            raise HTTPException(status_code=404, detail="User not found")

        file_bytes = await file.read()
        if not file_bytes:
            print(f"❌ Empty file uploaded: {file.filename}")
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        print(f"📦 File read successfully: {len(file_bytes)} bytes")

        try:
            print(f"📤 Starting upload process for {file.filename}")
            extracted_text = None  
            # extract_text_from_file_bytes(file_bytes, file.filename)
            print(f"📄 Skipping text extraction for debugging - {file.filename}")
        except Exception as e:
            print(f"❌ Critical error in extract_text_from_file_bytes: {e}")
            extracted_text = None

        print(f"💾 About to call crud.create_uploded_doc...")
        db_doc = crud.create_uploded_doc(db, file, session_id, file_bytes, user.id, extracted_text)
        print(f"✅ DB doc created successfully: {db_doc.id}")
        
        result = {
            "message": "File uploaded and stored successfully.",
            "document_id": db_doc.id,
            "filename": db_doc.filename,
            "size_bytes": len(file_bytes),
            "extracted_text_length": len(extracted_text) if extracted_text else 0
        }
        print(f"🎉 Upload completed successfully: {result}")
        return result
        
    except HTTPException as he:
        print(f"❌ HTTP Exception in upload_file_to_db: {he.detail}")
        raise
    except Exception as e:
        print(f"❌ Unexpected error in upload_file_to_db: {e}")
        import traceback
        print(f"🔍 Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

async def add_conversation_details_to_db(session_id, conv_messages, db: Session):
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
