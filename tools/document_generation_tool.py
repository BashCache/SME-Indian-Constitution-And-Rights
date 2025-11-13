"""
DocumentGenerationTool: A tool for generating documents
in DOCX, PPTX, and PDF formats from text content.

Uses Strategy Pattern with an interface for document generators.
Can be used standalone or integrated with LangChain agents.
"""

import datetime
from typing import Optional, Any
from pathlib import Path

from .document_generator_factory import DocumentGeneratorFactory
from pydantic import BaseModel, Field

class DocumentGenerationInput(BaseModel):
    """Input schema for DocumentGenerationTool"""
    content: str = Field(..., description="The text content to be converted into a document")
    document_type: str = Field(..., description="Type of document to generate: 'docx', 'pptx', or 'pdf'")
    output_path: Optional[str] = Field(None, description="Output file path. If not provided, a default name will be generated")
    title: Optional[str] = Field(None, description="Title for the document")
    author: Optional[str] = Field(None, description="Author name for document metadata")
    subject: Optional[str] = Field(None, description="Subject/topic for document metadata")

class DocumentGenerationTool:
    """
    A LangChain tool for generating documents in various formats (DOCX, PPTX, PDF)
    from text content.
    
    Uses Strategy Pattern to delegate document generation to appropriate generators.
    
    This tool can:
    - Generate DOCX documents with proper formatting
    - Generate PPTX presentations with slides
    - Generate PDF documents with styled content
    """
    
    def __init__(self, output_directory: str = "generated_documents"):
        """
        Initialize the DocumentGenerationTool.
        
        Args:
            output_directory: Directory where generated documents will be saved
        """
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.factory = DocumentGeneratorFactory()
        self.name = "document_generation_tool"
        self.description = (
            "Generates documents in DOCX, PPTX, or PDF format from text content. "
            "Input should be a dictionary with 'content' (text to convert), "
            "'document_type' ('docx', 'pptx', or 'pdf'), and optional 'output_path', "
            "'title', 'author', and 'subject' fields."
        )
    
    def _generate_output_path(self, document_type: str, output_path: Optional[str] = None) -> str:
        """
        Generate or validate output path for the document.
        
        Args:
            document_type: Type of document
            output_path: Optional custom output path
            
        Returns:
            Validated output path string
        """
        if not output_path:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"document_{timestamp}.{document_type.lower()}"
            output_path = str(self.output_directory / filename)
        else:
            # Ensure output directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path = str(output_path)
        
        return output_path
    
    def _run(self, content: str, document_type: str, output_path: Optional[str] = None,
            title: Optional[str] = None, author: Optional[str] = None,
            subject: Optional[str] = None, **kwargs) -> str:
        """
        Generate a document from text content using the Strategy pattern.
        
        Args:
            content: Text content to convert
            document_type: Type of document ('docx', 'pptx', or 'pdf')
            output_path: Optional output file path
            title: Optional document title
            author: Optional author name
            subject: Optional subject/topic
        
        Returns:
            Path to the generated document
        """
        # Get the appropriate generator using factory
        generator = self.factory.create_generator(document_type)
        
        # Generate output path
        final_output_path = self._generate_output_path(document_type, output_path)
        
        # Delegate generation to the appropriate generator
        result_path = generator.generate(
            content=content,
            output_path=final_output_path,
            title=title,
            author=author,
            subject=subject
        )
        
        return f"Document generated successfully at: {result_path}"
    
    def _arun(self, content: str, document_type: str, output_path: Optional[str] = None,
             title: Optional[str] = None, author: Optional[str] = None,
             subject: Optional[str] = None) -> str:
        """Async version of _run (not implemented, as document generation is synchronous)"""
        raise NotImplementedError("Async document generation is not supported")
    
    def run(self, tool_input: Any) -> str:
        """
        Run the tool with input.
        
        This method handles multiple input types:
        1. Direct dictionary input (for programmatic use)
        2. DocumentGenerationInput Pydantic model instance
        3. String (will be treated as content with default docx format)
        
        Args:
            tool_input: Can be:
                - Dictionary with keys: content, document_type, output_path, title, author, subject
                - DocumentGenerationInput Pydantic model instance
                - String (will be treated as content with default docx format)
        
        Returns:
            Path to the generated document
        """
        # Handle Pydantic model input (from LangChain)
        if isinstance(tool_input, DocumentGenerationInput):
            return self._run(
                content=tool_input.content,
                document_type=tool_input.document_type,
                output_path=tool_input.output_path,
                title=tool_input.title,
                author=tool_input.author,
                subject=tool_input.subject
            )
        # Handle dictionary input
        elif isinstance(tool_input, dict):
            return self._run(
                content=tool_input.get('content', ''),
                document_type=tool_input.get('document_type', 'docx'),
                output_path=tool_input.get('output_path'),
                title=tool_input.get('title'),
                author=tool_input.get('author'),
                subject=tool_input.get('subject')
            )
        else:
            # If it's a string, try to parse it or use as content
            return self._run(content=str(tool_input), document_type='docx')


# Helper function to create the tool instance
def create_document_generation_tool(output_directory: str = "generated_documents") -> DocumentGenerationTool:
    """
    Factory function to create a DocumentGenerationTool instance.
    
    Args:
        output_directory: Directory where generated documents will be saved
    
    Returns:
        DocumentGenerationTool instance
    """
    return DocumentGenerationTool(output_directory=output_directory)
