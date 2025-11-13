# Import base interface and generators
from .generators.document_generator import DocumentGenerator
from .generators.docx_generator import DocxGenerator
from .generators.pptx_generator import PptxGenerator
from .generators.pdf_generator import PdfGenerator

from enum import Enum
from typing import Dict

class DocumentType(str, Enum):
    """Supported document types"""
    DOCX = "docx"
    PPTX = "pptx"
    PDF = "pdf"

class DocumentGeneratorFactory:
    """
    Factory class to create appropriate document generator based on document type.
    Implements the Strategy pattern by selecting the right generator.
    """
    
    _generators: Dict[DocumentType, type[DocumentGenerator]] = {
        DocumentType.DOCX: DocxGenerator,
        DocumentType.PPTX: PptxGenerator,
        DocumentType.PDF: PdfGenerator,
    }
    
    @classmethod
    def create_generator(cls, document_type: str) -> DocumentGenerator:
        """
        Create a document generator for the specified document type.
        
        Args:
            document_type: Type of document ('docx', 'pptx', or 'pdf')
            
        Returns:
            DocumentGenerator instance
            
        Raises:
            ValueError: If document type is not supported
        """
        try:
            doc_type = DocumentType(document_type.lower())
        except ValueError:
            raise ValueError(f"Unsupported document type: {document_type}. Must be one of: docx, pptx, pdf")
        
        generator_class = cls._generators.get(doc_type)
        if not generator_class:
            raise ValueError(f"No generator available for document type: {document_type}")
        
        return generator_class()
    
    @classmethod
    def register_generator(cls, document_type: DocumentType, generator_class: type[DocumentGenerator]):
        """
        Register a new document generator type.
        Allows extending the system with new document types.
        
        Args:
            document_type: DocumentType enum value
            generator_class: Class that implements DocumentGenerator interface
        """
        if not issubclass(generator_class, DocumentGenerator):
            raise TypeError(f"Generator class must inherit from DocumentGenerator")
        cls._generators[document_type] = generator_class
