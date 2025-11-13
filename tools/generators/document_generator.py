"""
Base interface for document generators.

Defines the abstract base class (interface) that all document generators must implement.
Contains common utility methods used by all generators.
"""

import re
from abc import ABC, abstractmethod
from typing import Optional, List


class DocumentGenerator(ABC):
    """
    Abstract base class (interface) for document generators.
    Defines common methods and the contract for all document generators.
    """
    
    @staticmethod
    def split_into_paragraphs(content: str) -> List[str]:
        """
        Split content into paragraphs, handling various separators.
        Common utility method used by all generators.
        
        Args:
            content: Text content to split
            
        Returns:
            List of paragraph strings
        """
        # Split by double newlines, then by single newlines
        paragraphs = re.split(r'\n\s*\n', content.strip())
        # Further split long paragraphs if needed
        result = []
        for para in paragraphs:
            if len(para) > 500:  # Split very long paragraphs
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current = ""
                for sentence in sentences:
                    if len(current + sentence) > 500:
                        if current:
                            result.append(current.strip())
                        current = sentence
                    else:
                        current += " " + sentence if current else sentence
                if current:
                    result.append(current.strip())
            else:
                result.append(para.strip())
        return [p for p in result if p]
    
    @staticmethod
    def split_into_slides(content: str, max_slide_length: int = 300) -> List[str]:
        """
        Split content into slides for presentations.
        Common utility method for presentation generators.
        
        Args:
            content: Text content to split
            max_slide_length: Maximum characters per slide
            
        Returns:
            List of slide content strings
        """
        paragraphs = DocumentGenerator.split_into_paragraphs(content)
        slides = []
        current_slide = []
        current_length = 0
        
        for para in paragraphs:
            para_length = len(para)
            if current_length + para_length > max_slide_length and current_slide:
                slides.append("\n".join(current_slide))
                current_slide = [para]
                current_length = para_length
            else:
                current_slide.append(para)
                current_length += para_length
        
        if current_slide:
            slides.append("\n".join(current_slide))
        
        return slides if slides else [content]
    
    @staticmethod
    def is_heading(text: str) -> bool:
        """
        Check if a text string is likely a heading.
        Common utility method for all generators.
        
        Args:
            text: Text to check
            
        Returns:
            True if text appears to be a heading
        """
        return text.startswith('#') or (text.isupper() and len(text) < 100)
    
    @staticmethod
    def extract_heading_level(text: str) -> tuple[int, str]:
        """
        Extract heading level and text from markdown-style heading.
        
        Args:
            text: Text that may contain markdown heading
            
        Returns:
            Tuple of (level, heading_text)
        """
        if text.startswith('#'):
            level = len(text) - len(text.lstrip('#'))
            heading_text = text.lstrip('#').strip()
            return (min(level, 3), heading_text)
        return (2, text)  # Default level 2 for all-caps headings
    
    @abstractmethod
    def generate(self, content: str, output_path: str, title: Optional[str] = None,
                author: Optional[str] = None, subject: Optional[str] = None) -> str:
        """
        Generate a document from text content.
        
        Args:
            content: Text content to convert
            output_path: Output file path
            title: Optional document title
            author: Optional author name
            subject: Optional subject/topic
        
        Returns:
            Path to the generated document
        """
        pass

