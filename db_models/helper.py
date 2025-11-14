# helper.py (Corrected)

from .models import SessionLocal, UploadedDocument, GeneratedDocument, ConversationMessage, User, init_db
from sqlalchemy.orm import Session
from typing import List, Optional
from passlib.context import CryptContext
import os

init_db()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_user(username: str, hashed_password: str) -> User:
    db: Session = SessionLocal()
    try:
        user = User(username=username, hashed_password=hashed_password)
        db.add(user); db.commit(); db.refresh(user)
        return user
    finally:
        db.close()

def get_user_by_username(username: str) -> Optional[User]:
    db = SessionLocal()
    try:
        # ---
        # FIX: Changed User.email to User.username to match the model
        # ---
        return db.query(User).filter(User.username == username).first()
    finally:
        db.close()

def save_uploaded_document(filename: str, text: str, raw_bytes: bytes = None, uploader_id: int = None, metadata: dict = None):
    db = SessionLocal()
    try:
        doc = UploadedDocument(filename=filename, content_text=text, raw_blob=raw_bytes, uploader_id=uploader_id, metadata=metadata or {})
        db.add(doc); db.commit(); db.refresh(doc)
        return doc
    finally:
        db.close()

def save_generated_document(title: str, document_type: str, content_blob: bytes, metadata: dict = None):
    db = SessionLocal()
    try:
        doc = GeneratedDocument(title=title, document_type=document_type, content_blob=content_blob, metadata=metadata or {})
        db.add(doc); db.commit(); db.refresh(doc)
        return doc
    finally:
        db.close()

def save_conversation_message(session_id: str, role: str, content: str, metadata: dict = None):
    db = SessionLocal()
    try:
        msg = ConversationMessage(session_id=session_id, role=role, content=content, metadata=metadata or {})
        db.add(msg); db.commit(); db.refresh(msg)
        return msg
    finally:
        db.close()

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    user = get_user_by_username(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

def list_users(db: Session, limit: int = 100) -> List[User]:
    return db.query(User).limit(limit).all()
