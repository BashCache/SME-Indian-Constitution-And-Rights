# from langchain.tools import tool
import json
import google.generativeai as genai
from typing import Any
from tools.document_generation_tool import DocumentGenerationTool
from langchain_core.tools import tool
from utils.doc_writer import DocumentWriter
from utils.email_impl import send_email
from utils.extractor.file_extractor import FileExtractor

_doc_tool = DocumentGenerationTool(output_directory="generated_documents")

def get_rag_answer(query: str, source: str, username: str, session_id: str, history: str, filepath) -> str:
    """
    This is your separate, synchronous (blocking) RAG implementation.
    It will be called asynchronously by FastAPI.
    """
    print(f"\n📚 Calling internal RAG system for: '{query}' with source: {source}")
    # --- This is a blocking call (e.g., DB query, network) ---

    uploaded_content_text = None
    uploaded_content_data = None
    if filepath:
        print(f"filepath: {filepath}")
        f = FileExtractor()
        uploaded_content_data = f.extract_text(filepath)
        uploaded_content_text = uploaded_content_data.content if uploaded_content_data else None

    print(f"Query: {query}")
    print(f"History: {history}")
    print(f"Uploaded content data: {uploaded_content_data}")
    if source == "external_kb":
        mock_answer = f"From **External KB**: Article 14 of the Indian Constitution guarantees the right to equality."
    elif source == "session_docs":
        # Write the extractor here
        mock_answer = f"From **Session Docs**: The uploaded document discusses corporate policy."
    elif source == "both":
        mock_answer = f"From **Both Sources**: Article 14 guarantees equality. The uploaded document outlines a corporate policy."
    else:
        mock_answer = ""
    print(f"✅ RAG Answer (first 100): {mock_answer[:100]}...")
    return mock_answer

@tool
def document_tool(content: str, document_type: str = "pdf", title: str = "Generated Document") -> str:
    """
from utils.doc_writer import DocumentWriter
    Generate a document (PDF, DOCX, or PPTX) using the provided content.
    'content' is the full text to put in the document.
    Returns the filename (e.g., 'Generated Document.pdf').
    """
    print(f"\n📄 [EXECUTE] DOCUMENT_TOOL (type={document_type}, title={title})")
    try:
        print(f"   ...with content (first 100 chars): {content[:100]}...")
        result = f"Document '{title}.{document_type}' generated successfully."
        print(f"✅ {result}")
        DocumentWriter.write(content, 'pdf', 'agent_data/generated_docs/', title)
        return f"{title}.{document_type}" # Return the filename
    except Exception as e:
        error_msg = f"Error generating document: {e}"
        print(f"❌ {error_msg}")
        return error_msg

@tool
def email_tool(filename: str, recipient: str = "user@example.com") -> str:
    """
    Send an email with a document attachment.
    'filename' is the path/name of the file to attach.
    Returns a success message.
    """
    print(f"\n📧 [EXECUTE] EMAIL_TOOL (to={recipient}, file={filename})")
    try:
        send_email(filename, recipient)
        result = f"Email sent successfully to {recipient} with attachment: {filename}"
        print(f"✅ {result}")
        return result
    except Exception as e:
        error_msg = f"Error sending email: {e}"
        print(f"❌ {error_msg}")
        return error_msg
    

# @tool("RAG Tool", return_direct=False)
# def rag_tool(input_json: str) -> str:
#     """
#     Uses the conversation history and query as input, and get answer
#     using the LLM.
#     """
#     try:
#         model = genai.GenerativeModel("gemini-2.5-pro")
#         prompt = f"Given the input JSON {input_json}, answer the query"
#         response = model.generate_content(prompt)
#         return response.text.strip()
#     except Exception as e:
#         return f"[RAG tool] Error: {e}"
    
# @tool("Document Generation Tool", return_direct=False)
# def doc_tool(payload: str) -> str:
#     """
#     Given the payload, convert it into a document based on its type.
#     """
#     try:
#         return "Generated document successfully"
#     except Exception as e:
#         return f"[Document Generation Tool] Error: {e}"
    
# @tool("Email Automation Tool", return_direct=False)
# def email_tool(filename: str) -> str:
#     """
#     Sends the file to the corresponding mail.
#     """
#     try:
#         return "Email automated"
#     except Exception as e:
#         return f"[Email Tool] Error: {e}"

# @tool("RAG_Tool", return_direct=False)
# def rag_tool(input_json: str) -> str:
#     """
#     Retrieve contextual information related to the query using conversation history.
#     """
#     try:
#         print(f"input json: {input_json}")
#         data = json.loads(input_json)
#         print(f"Data: {data}")
#         query = data.get("input", "")
#         history = data.get("history", [])

#         # Build a readable full conversation string (user + assistant)
#         full_context = "\n".join([
#             f"{m['role'].capitalize()}: {m['content']}" for m in history
#         ])

#         return (
#             f"[RAG_Tool] Retrieved info based on:\n"
#             f"Input: '{query}'\n\n"
#             f"Conversation Context:\n{full_context[-2000:]}"  # truncate for safety
#         )

#     except Exception as e:
#         return f"[RAG_Tool] Error parsing input: {str(e)}"

# @tool("DocumentGenerationTool", return_direct=False)
# def document_tool(payload: str) -> str:
#     """Generate a document using the project's document generator factory.

#     The payload may be either a raw content string (defaults to PDF) or a JSON
#     string with keys:
#       - type: one of 'pdf', 'docx', 'pptx'
#       - content: the textual content to render
#       - filename: optional output filename

#     Returns a message with the generated filename.
#     """
#     # try JSON payload, else treat as raw content
#     try:
#         data = json.loads(payload)
#     except Exception:
#         data = {"content": payload, "document_type": "pdf"}

#     # ensure required keys exist
#     content = data.get("content", "")
#     doc_type = data.get("document_type", "pdf")
#     output_path = data.get("output_path")
#     print(f"Doc type: {doc_type}, output path: {output_path}")

#     if not content:
#         return "[DocumentGenerationTool] No content supplied."

#     result = _doc_tool.run({
#         "content": content,
#         "document_type": doc_type,
#         "output_path": output_path,
#         "title": data.get("title"),
#         "author": data.get("author"),
#         "subject": data.get("subject")
#     })
#     return result

# @tool("EmailAutomationTool", return_direct=False)
# def email_tool(filename: str) -> str:
#     """Send an email with the generated document."""
#     return f"[EmailAutomationTool] Sent file '{filename}' via email."
