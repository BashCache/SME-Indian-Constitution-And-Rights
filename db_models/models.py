from sqlalchemy import (
    Column, Integer, String, Text, DateTime, LargeBinary, Boolean, JSON, create_engine
)
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sme_constitution_2.db")

Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class UploadedDocument(Base):
    __tablename__ = "uploaded_documents"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=True)
    content_text = Column(Text, nullable=True)      # extracted text
    raw_blob = Column(LargeBinary, nullable=True)   # optional raw file bytes
    uploader_id = Column(Integer, nullable=True)
    metadata_custom = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class GeneratedDocument(Base):
    __tablename__ = "generated_documents"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=True)
    document_type = Column(String, nullable=False)
    content_blob = Column(LargeBinary, nullable=True)  # binary output (pdf/docx/pptx)
    metadata_custom = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class SessionInfo(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True, nullable=False)
    title=Column(String, index=True, nullable=False)
    user_id = Column(Integer, nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    metadata_custom = Column(JSON, nullable=True)

class ConversationMessage(Base):
    __tablename__ = "conversation_messages"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True, nullable=True)
    role = Column(String, nullable=True)
    content = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metadata_custom = Column(JSON, nullable=True)

def init_db():
    Base.metadata.create_all(bind=engine)