"""
PDF Generator: Generates PDF documents from text content.
"""

from typing import Optional
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.colors import black

from .document_generator import DocumentGenerator


class PdfGenerator(DocumentGenerator):
    """Generator for PDF documents"""
    
    def generate(self, content: str, output_path: str, title: Optional[str] = None,
                author: Optional[str] = None, subject: Optional[str] = None) -> str:
        """Generate a PDF document from text content"""
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Container for the 'Flowable' objects
        story = []
        
        # Define styles
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=black,
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=black,
            spaceAfter=12,
            spaceBefore=12
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=12,
            textColor=black,
            spaceAfter=12,
            alignment=TA_JUSTIFY,
            leftIndent=36
        )
        
        # Add title if provided
        if title:
            story.append(Paragraph(title, title_style))
            story.append(Spacer(1, 0.2*inch))
        
        # Split content into paragraphs
        paragraphs = self.split_into_paragraphs(content)
        
        for para_text in paragraphs:
            if not para_text.strip():
                continue
            
            # Clean text for PDF (escape special characters)
            para_text = para_text.replace('&', '&amp;')
            para_text = para_text.replace('<', '&lt;')
            para_text = para_text.replace('>', '&gt;')
            
            # Check if it's a heading
            if self.is_heading(para_text):
                level, heading_text = self.extract_heading_level(para_text)
                if level == 1:
                    story.append(Paragraph(heading_text, title_style))
                else:
                    story.append(Paragraph(heading_text, heading_style))
                story.append(Spacer(1, 0.1*inch))
            else:
                story.append(Paragraph(para_text, body_style))
        
        # Build PDF
        doc.build(story)
        return output_path

