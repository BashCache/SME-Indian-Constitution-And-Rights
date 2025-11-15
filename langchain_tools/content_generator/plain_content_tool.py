"""
Simple RAG Content Generator Tool using Gemini API
"""

import os
import google.generativeai as genai
from langchain_core.tools import tool

def setup_gemini():
    """Initialize Gemini API"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not found")
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-pro")


@tool
def normal_content_tool(user_query: str, rag_context: str) -> str:
    """
    Generate answer using user query and RAG context via Gemini API.
    
    Args:
        user_query: The user's question
        rag_context: Retrieved context/documents related to the query
        
    Returns:
        Generated answer based on query and context
    """
    try:
        # Initialize Gemini
        model = setup_gemini()
        
        # Create prompt combining query and context
        prompt = f"""Based on the following context, answer the user's question:

CONTEXT:
{rag_context}

QUESTION:
{user_query}

Please provide a clear and accurate answer based on the given context."""
        
        # Generate response
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"Error generating response: {str(e)}"


# Alternative simple function (non-tool version)
def generate_rag_answer(user_query: str, rag_context: str) -> str:
    """
    Simple function to generate answer using Gemini API
    
    Args:
        user_query: The user's question
        rag_context: Retrieved context/documents
        
    Returns:
        Generated answer
    """
    try:
        model = setup_gemini()
        
        prompt = f"""Context: {rag_context}

Question: {user_query}

Answer based on the context above:"""
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"Error: {str(e)}"
