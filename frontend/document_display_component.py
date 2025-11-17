"""
Document Display Component for Streamlit UI

This module provides components for displaying and interacting with generated documents
in the Constitutional AI Assistant interface.
"""

import streamlit as st
import os
import base64
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional
import PyPDF2
import io


def get_file_icon(file_type: str) -> str:
    """Get appropriate emoji icon for file type"""
    icon_map = {
        '.pdf': '📄',
        '.docx': '📝',
        '.doc': '📝',
        '.pptx': '📊',
        '.ppt': '📊',
        '.txt': '📃',
        '.html': '🌐',
        '.json': '⚙️',
        '.csv': '📊',
        '.xlsx': '📊',
        '.xls': '📊'
    }
    return icon_map.get(file_type.lower(), '📁')


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


def get_file_download_link(file_path: str, filename: str = None) -> str:
    """Generate a download link for a file"""
    if not os.path.exists(file_path):
        return None
    
    if filename is None:
        filename = os.path.basename(file_path)
    
    try:
        with open(file_path, "rb") as f:
            file_data = f.read()
        
        b64 = base64.b64encode(file_data).decode()
        
        # Get MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            mime_type = "application/octet-stream"
        
        return f"data:{mime_type};base64,{b64}"
    except Exception as e:
        st.error(f"Error generating download link: {str(e)}")
        return None


def display_pdf_preview(file_path: str, max_pages: int = 3) -> bool:
    """Display PDF preview with text extraction"""
    if not os.path.exists(file_path) or not file_path.lower().endswith('.pdf'):
        return False
    
    try:
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            st.markdown("### 📖 PDF Preview")
            st.markdown(f"**Total Pages:** {len(pdf_reader.pages)}")
            
            # Show first few pages as text
            preview_text = []
            pages_to_show = min(max_pages, len(pdf_reader.pages))
            
            for i in range(pages_to_show):
                try:
                    page = pdf_reader.pages[i]
                    text = page.extract_text()
                    if text.strip():
                        preview_text.append(f"**Page {i+1}:**\n{text[:500]}{'...' if len(text) > 500 else ''}")
                except Exception:
                    preview_text.append(f"**Page {i+1}:** [Could not extract text]")
            
            if preview_text:
                with st.expander(f"📄 Preview (First {pages_to_show} pages)", expanded=False):
                    for text in preview_text:
                        st.markdown(text)
                        st.markdown("---")
            
            return True
    except Exception as e:
        st.error(f"Could not preview PDF: {str(e)}")
        return False


def display_text_file_preview(file_path: str, max_chars: int = 1000) -> bool:
    """Display text file preview"""
    if not os.path.exists(file_path):
        return False
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        st.markdown("### 📖 Text Preview")
        
        preview_content = content[:max_chars]
        if len(content) > max_chars:
            preview_content += "\n\n... [File truncated for preview]"
        
        with st.expander("📄 File Content", expanded=False):
            st.text_area("Content", preview_content, height=300, disabled=True)
        
        return True
    except Exception as e:
        st.error(f"Could not preview text file: {str(e)}")
        return False


def display_html_preview(file_path: str) -> bool:
    """Display HTML file preview"""
    if not os.path.exists(file_path) or not file_path.lower().endswith('.html'):
        return False
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        
        st.markdown("### 🌐 HTML Preview")
        
        # Show HTML source in expander
        with st.expander("📄 HTML Source", expanded=False):
            st.code(html_content[:2000] + ("..." if len(html_content) > 2000 else ""), language="html")
        
        # Render HTML content (be careful with security)
        with st.expander("🌐 Rendered Preview", expanded=True):
            # For security, we should sanitize HTML, but for generated documents it should be safe
            st.components.v1.html(html_content, height=400, scrolling=True)
        
        return True
    except Exception as e:
        st.error(f"Could not preview HTML file: {str(e)}")
        return False


def display_document_card(doc_info: Dict[str, Any]) -> None:
    """Display a document card with preview and download options"""
    file_path = doc_info.get("file_path", "")
    filename = doc_info.get("filename", "Unknown File")
    file_type = doc_info.get("file_type", "")
    file_size = doc_info.get("file_size", 0)
    created_at = doc_info.get("created_at", "")
    
    icon = get_file_icon(file_type)
    
    # Document card
    with st.container():
        st.markdown(f"""
        <div style="
            border: 1px solid #ddd; 
            border-radius: 10px; 
            padding: 15px; 
            margin: 10px 0; 
            background-color: #f9f9f9;
        ">
        """, unsafe_allow_html=True)
        
        # Header with file info
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown(f"## {icon} {filename}")
            if created_at:
                from datetime import datetime
                try:
                    dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    st.caption(f"📅 Created: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                except:
                    st.caption(f"📅 Created: {created_at}")
        
        with col2:
            st.markdown(f"**Type:** {file_type.upper()}")
            st.markdown(f"**Size:** {format_file_size(file_size)}")
        
        with col3:
            # Download button
            if os.path.exists(file_path):
                download_link = get_file_download_link(file_path, filename)
                if download_link:
                    st.markdown(f"""
                    <a href="{download_link}" download="{filename}">
                        <button style="
                            background-color: #4CAF50; 
                            color: white; 
                            padding: 8px 16px; 
                            border: none; 
                            border-radius: 5px; 
                            cursor: pointer;
                            width: 100%;
                        ">⬇️ Download</button>
                    </a>
                    """, unsafe_allow_html=True)
            else:
                st.error("❌ File not found")
        
        # Preview based on file type
        if os.path.exists(file_path):
            if file_type.lower() == '.pdf':
                display_pdf_preview(file_path)
            elif file_type.lower() in ['.txt', '.md']:
                display_text_file_preview(file_path)
            elif file_type.lower() == '.html':
                display_html_preview(file_path)
            else:
                st.info(f"📁 {file_type.upper()} file preview not available. Use download button to view.")
        
        st.markdown("</div>", unsafe_allow_html=True)


def display_generated_documents_section(documents: List[Dict[str, Any]]) -> None:
    """Display a section with all generated documents"""
    if not documents:
        return
    
    st.markdown("---")
    st.markdown("## 📁 Generated Documents")
    st.markdown(f"Found {len(documents)} document(s) from your request:")
    
    for i, doc_info in enumerate(documents):
        with st.expander(f"{get_file_icon(doc_info.get('file_type', ''))} {doc_info.get('filename', f'Document {i+1}')}", expanded=(i == 0)):
            display_document_card(doc_info)


def scan_generated_documents_folder(folder_path: str = "generated_documents", 
                                   limit: int = 10,
                                   sort_by_date: bool = True) -> List[Dict[str, Any]]:
    """Scan the generated documents folder and return recent files"""
    if not os.path.exists(folder_path):
        return []
    
    documents = []
    
    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            if os.path.isfile(file_path):
                stat = os.stat(file_path)
                
                doc_info = {
                    "file_path": file_path,
                    "filename": filename,
                    "file_type": os.path.splitext(filename)[1].lower(),
                    "file_size": stat.st_size,
                    "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
                }
                documents.append(doc_info)
        
        # Sort by creation date (newest first)
        if sort_by_date:
            documents.sort(key=lambda x: x["created_at"], reverse=True)
        
        return documents[:limit]
    
    except Exception as e:
        st.error(f"Error scanning documents folder: {str(e)}")
        return []


def display_recent_documents_sidebar():
    """Display recent documents in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 Recent Documents")
    
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs_folder = os.path.join(project_root, "generated_documents")
    
    recent_docs = scan_generated_documents_folder(docs_folder, limit=5)
    
    if recent_docs:
        for doc in recent_docs:
            with st.sidebar.expander(f"{get_file_icon(doc['file_type'])} {doc['filename'][:20]}{'...' if len(doc['filename']) > 20 else ''}", expanded=False):
                st.write(f"**Type:** {doc['file_type'].upper()}")
                st.write(f"**Size:** {format_file_size(doc['file_size'])}")
                
                if os.path.exists(doc['file_path']):
                    download_link = get_file_download_link(doc['file_path'], doc['filename'])
                    if download_link:
                        st.markdown(f"[⬇️ Download]({download_link})")
    else:
        st.sidebar.info("No recent documents found")


# Import datetime for the scan function
from datetime import datetime
