from typing import Optional, List
from fastapi import HTTPException
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from uuid import uuid4
from datetime import datetime
from .models import SessionLocal, User, init_db, SessionInfo, ConversationMessage, UploadedDocument

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_db_session() -> Session:
    """Get a new DB session (must be closed manually)."""
    return SessionLocal()

def get_db() -> Session: # type: ignore
    """Get a new DB session (for dependency injection - yields and auto-closes)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_user(db: Session, username: str, password: str) -> User:
    hashed = hash_password(password)
    user = User(username=username, hashed_password=hashed)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def get_user_by_username(db: Session, username: str) -> Optional[User]:
    return db.query(User).filter(User.username == username).first()

def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    user = get_user_by_username(db, username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

def list_users(db: Session, limit: int = 100) -> List[User]:
    return db.query(User).limit(limit).all()

def create_chat_session(
    db: Session,
    username: str,
    title: str,
    metadata: Optional[dict] = None
) -> SessionInfo:
    """
    Create a new chat session for a user and return it.
    """
    try:
        user = get_user_by_username(db, username)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        session_id = f"{username}_{uuid4().hex[:12]}"  # consistent with your CLI
        new_session = SessionInfo(
            session_id=session_id,
            user_id=user.id,
            title=title,
            started_at=datetime.utcnow(),
            metadata_custom=metadata or {}
        )

        db.add(new_session)
        db.commit()
        db.refresh(new_session)
        return new_session
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

def list_sessions_by_username(db: Session, username: str) -> List[dict]:
    try:
        user = get_user_by_username(db, username)
        if not user:
            return []
        
        rows = db.query(SessionInfo).filter(SessionInfo.user_id == user.id).order_by(SessionInfo.started_at.desc()).all()
        return [
            {
                "session_id": r.session_id,
                "title": r.title,
                "started_at": r.started_at.isoformat(),
                "metadata_custom": r.metadata_custom or {}
            } for r in rows
        ]
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

def delete_session(db: Session, session_id: str) -> bool:
    try:
        row = db.query(SessionInfo).filter(SessionInfo.session_id == session_id).first()
        if not row:
            return False
        db.delete(row)
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

def rename_session(db: Session, session_id: str, new_title: str) -> bool:
    """Rename a session with the given session_id"""
    try:
        row = db.query(SessionInfo).filter(SessionInfo.session_id == session_id).first()
        if not row:
            return False
        
        row.title = new_title
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

def fetch_session_by_sessionid(db: Session, session_id: str) -> Optional[dict]:
    try:
        row = db.query(SessionInfo).filter(SessionInfo.session_id == session_id).first()
        if not row:
            return None

        return {
            "session_id": row.session_id,
            "title": row.title,
            "started_at": row.started_at.isoformat(),
            "metadata_custom": row.metadata_custom or {}
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

def fetch_conversation_messages(session_id: str) -> List[dict]:
    db = get_db_session()
    try:
        rows = db.query(ConversationMessage).filter(ConversationMessage.session_id == session_id).order_by(ConversationMessage.timestamp.asc()).all()
        return [
            {
                "session_id": r.session_id,
                "role": r.role,
                "timestamp": r.timestamp.isoformat(),
                "content": r.content
            } for r in rows
        ]
    except Exception as e:
        db.rollback()
        print(f" Exception: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        db.close()

def create_uploded_doc(db: Session, file, session_id, file_bytes, user_id, extracted_text) -> UploadedDocument:
    try:
        db_doc = UploadedDocument(
            filename=file.filename,
            content_text=extracted_text,
            raw_blob=file_bytes,
            uploader_id=user_id,
            metadata_custom={
                "session_id": session_id,
                "content_type": file.content_type,
                "size_bytes": len(file_bytes),
            },
            created_at=datetime.utcnow(),
        )
        db.add(db_doc)
        db.commit()
        db.refresh(db_doc)
        return db_doc
    except Exception as e:
        db.rollback()
        print(f" Exception: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

def save_conversation_message(db: Session, session_id: str, role: str, content: str, metadata: dict = None):
    try:
        msg = ConversationMessage(session_id=session_id, role=role, content=content, metadata_custom=metadata or {})
        db.add(msg)
        db.commit()
        db.refresh(msg)
        return msg
    except Exception as e:
        db.rollback()
        print(f" Exception: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        
def ensure_db():
    init_db()