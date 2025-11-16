"""
Enhanced PDF formatter with markdown processing and content-aware styling.
Provides professional formatting for educational and legal documents.
"""

import re
from typing import List, Tuple, Dict, Any
from datetime import datetime
import os

# Try to import ReportLab components, with fallback
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.colors import black, blue, gray, darkblue
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    # Create dummy classes for when ReportLab is not available
    class SimpleDocTemplate:
        def __init__(self, *args, **kwargs): pass
        def build(self, story): pass
    
    class Paragraph:
        def __init__(self, text, style): 
            self.text = text
            self.style = style
    
    class Spacer:
        def __init__(self, width, height): pass
    
    letter = (612, 792)  # Standard letter size
    TA_CENTER = 1
    TA_LEFT = 0
    TA_RIGHT = 2
    TA_JUSTIFY = 4


class MarkdownProcessor:
    """Processes markdown-like syntax for PDF formatting"""
    
    @staticmethod
    def process_text(text: str) -> str:
        """Convert markdown syntax to ReportLab markup"""
        # Handle multiple asterisks patterns first
        # Bold text with double asterisks
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        
        # Italic text with single asterisks (but avoid interfering with bold)
        text = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'<i>\1</i>', text)
        
        # Code/monospace
        text = re.sub(r'`(.*?)`', r'<font name="Courier">\1</font>', text)
        
        # Links (basic)
        text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<link href="\2">\1</link>', text)
        
        # Clean up any remaining asterisks that might be formatting artifacts
        text = re.sub(r'\*{3,}', '', text)  # Remove triple or more asterisks
        
        return text
    
    @staticmethod
    def extract_headings(content: str) -> List[Tuple[int, str, str]]:
        """Extract headings and return (level, title, content)"""
        lines = content.split('\n')
        sections = []
        current_content = []
        
        for line in lines:
            heading_match = re.match(r'^(#{1,6})\s+(.+)', line)
            if heading_match:
                if current_content:
                    sections.append((0, '', '\n'.join(current_content)))
                    current_content = []
                level = len(heading_match.group(1))
                title = heading_match.group(2)
                sections.append((level, title, ''))
            else:
                current_content.append(line)
        
        if current_content:
            sections.append((0, '', '\n'.join(current_content)))
        
        return sections


class ContentAnalyzer:
    """Analyzes content to determine document type and structure"""
    
    @staticmethod
    def detect_document_type(content: str) -> str:
        """Detect if content is quiz, legal, report, or general"""
        content_lower = content.lower()
        
        quiz_indicators = [
            'q1.', 'q2.', 'question', 'a)', 'b)', 'c)', 'd)', 'mcq', 'quiz', 
            'instructions:', 'answer:', 'marks:', 'total marks:', 'time allowed:',
            'question 1', 'question 2', 'read each question', 'choose the best',
            'attempt all questions', 'descriptive questions'
        ]
        legal_indicators = [
            'article', 'section', 'constitution', 'fundamental rights', 
            'directive principles', 'supreme court', 'high court', 'judgment',
            'petitioner', 'respondent', 'constitutional law'
        ]
        report_indicators = [
            'executive summary', 'conclusion', 'recommendations', 'analysis',
            'introduction', 'methodology', 'findings', 'abstract'
        ]
        
        quiz_score = sum(1 for indicator in quiz_indicators if indicator in content_lower)
        legal_score = sum(1 for indicator in legal_indicators if indicator in content_lower)
        report_score = sum(1 for indicator in report_indicators if indicator in content_lower)
        
        # Lower threshold for quiz detection since quiz content can be varied
        if quiz_score >= 2:
            return 'quiz'
        elif legal_score >= 2:
            return 'legal'
        elif report_score >= 2:
            return 'report'
        else:
            return 'general'
    
    @staticmethod
    def extract_quiz_structure(content: str) -> Dict[str, List[str]]:
        """Extract quiz questions, options, and other elements"""
        lines = content.split('\n')
        structure = {
            'instructions': [],
            'questions': [],
            'current_question': None,
            'options': [],
            'answers': [],
            'metadata': []  # For titles, dates, marks info
        }
        
        current_section = 'metadata'
        in_instructions = False
        current_question_text = []
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Check for instruction section markers
            if re.match(r'^\s*INSTRUCTIONS?\s*:?\s*$', line_stripped, re.IGNORECASE):
                in_instructions = True
                current_section = 'instructions'
                structure['instructions'].append(line_stripped)
                continue
            
            # Check if we're in instructions and line starts with a number or bullet
            if in_instructions and (re.match(r'^\d+\.', line_stripped) or 
                                  re.match(r'^-', line_stripped) or 
                                  any(word in line_stripped.lower() for word in ['read', 'choose', 'write', 'attempt', 'time', 'marks'])):
                structure['instructions'].append(line_stripped)
                continue
            
            # End of instructions section (when we hit a heading or question)
            if in_instructions and (re.match(r'^#+\s', line_stripped) or 
                                   re.match(r'^\*\*.*\*\*\s*$', line_stripped) or
                                   re.match(r'^Question\s+\d+', line_stripped, re.IGNORECASE)):
                in_instructions = False
                current_section = 'questions'
            
            # Question detection - multiple patterns
            question_patterns = [
                r'^Q\d+[\.\:]',  # Q1., Q2:
                r'^\d+[\.\:]',   # 1., 2:
                r'^Question\s+\d+\s*[\:\.]',  # Question 1:, Question 2.
                r'^\*\*Question\s+\d+\s*[\:\.]?\*\*'  # **Question 1:**
            ]
            
            if any(re.match(pattern, line_stripped, re.IGNORECASE) for pattern in question_patterns):
                # Save previous question if exists
                if structure['current_question'] or current_question_text:
                    full_question = structure['current_question'] or ''
                    if current_question_text:
                        full_question += '\n' + '\n'.join(current_question_text)
                    
                    structure['questions'].append({
                        'question': full_question.strip(),
                        'options': structure['options'].copy()
                    })
                    structure['options'].clear()
                    current_question_text = []
                
                structure['current_question'] = line_stripped
                current_section = 'question'
                continue
            
            # Continue building current question text
            if current_section == 'question' and structure['current_question']:
                # Check for MCQ options
                if re.match(r'^[A-Da-d][\.\)]\s', line_stripped, re.IGNORECASE):
                    structure['options'].append(line_stripped)
                # Check for answer markers
                elif 'answer:' in line_stripped.lower() or 'correct:' in line_stripped.lower():
                    structure['answers'].append(line_stripped)
                # Otherwise, it's part of the question text
                else:
                    current_question_text.append(line_stripped)
                continue
            
            # Handle metadata (titles, dates, marks info)
            if current_section == 'metadata' or (not in_instructions and current_section != 'question'):
                # Look for titles, dates, marks
                if (re.match(r'^\*\*.*\*\*\s*$', line_stripped) or  # **Title**
                    'date:' in line_stripped.lower() or
                    'marks:' in line_stripped.lower() or
                    'total marks:' in line_stripped.lower() or
                    re.match(r'^#+\s', line_stripped)):  # # Heading
                    structure['metadata'].append(line_stripped)
        
        # Add last question
        if structure['current_question'] or current_question_text:
            full_question = structure['current_question'] or ''
            if current_question_text:
                full_question += '\n' + '\n'.join(current_question_text)
            
            structure['questions'].append({
                'question': full_question.strip(),
                'options': structure['options'].copy()
            })
        
        return structure


class StyleManager:
    """Manages different style sets for various document types"""
    
    def __init__(self):
        if REPORTLAB_AVAILABLE:
            self.base_styles = getSampleStyleSheet()
            self.custom_styles = {}
            self._create_custom_styles()
        else:
            self.custom_styles = self._create_fallback_styles()
    
    def _create_custom_styles(self):
        """Create custom styles for different content types"""
        if not REPORTLAB_AVAILABLE:
            return self._create_fallback_styles()
            
        # Title styles
        self.custom_styles['title'] = ParagraphStyle(
            'CustomTitle',
            parent=self.base_styles['Title'],
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=darkblue,
            fontName='Helvetica-Bold'
        )
        
        self.custom_styles['subtitle'] = ParagraphStyle(
            'CustomSubtitle',
            parent=self.base_styles['Heading1'],
            fontSize=14,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=blue,
            fontName='Helvetica-Bold'
        )
        
        # Quiz-specific styles
        self.custom_styles['quiz_instructions'] = ParagraphStyle(
            'QuizInstructions',
            parent=self.base_styles['Normal'],
            fontSize=10,
            leftIndent=20,
            rightIndent=20,
            spaceAfter=20,
            backgroundColor=gray,
            borderPadding=10
        )
        
        self.custom_styles['quiz_question'] = ParagraphStyle(
            'QuizQuestion',
            parent=self.base_styles['Normal'],
            fontSize=12,
            fontName='Helvetica-Bold',
            spaceAfter=10,
            spaceBefore=15
        )
        
        self.custom_styles['quiz_option'] = ParagraphStyle(
            'QuizOption',
            parent=self.base_styles['Normal'],
            fontSize=11,
            leftIndent=30,
            spaceAfter=5
        )
        
        # Legal document styles
        self.custom_styles['legal_article'] = ParagraphStyle(
            'LegalArticle',
            parent=self.base_styles['Normal'],
            fontSize=12,
            fontName='Helvetica-Bold',
            spaceAfter=10,
            textColor=darkblue
        )
        
        self.custom_styles['legal_text'] = ParagraphStyle(
            'LegalText',
            parent=self.base_styles['Normal'],
            fontSize=11,
            alignment=TA_JUSTIFY,
            leftIndent=20,
            spaceAfter=12
        )
        
        # Heading styles
        for i in range(1, 7):
            self.custom_styles[f'heading{i}'] = ParagraphStyle(
                f'CustomHeading{i}',
                parent=self.base_styles[f'Heading{min(i, 6)}'],
                fontSize=16 - i,
                fontName='Helvetica-Bold',
                textColor=darkblue,
                spaceAfter=12,
                spaceBefore=15
            )
        
        # Body text styles
        self.custom_styles['body'] = ParagraphStyle(
            'CustomBody',
            parent=self.base_styles['Normal'],
            fontSize=11,
            alignment=TA_JUSTIFY,
            spaceAfter=10
        )
        
        self.custom_styles['emphasis'] = ParagraphStyle(
            'Emphasis',
            parent=self.base_styles['Normal'],
            fontSize=11,
            fontName='Helvetica-Bold',
            textColor=darkblue
        )
    
    def _create_fallback_styles(self):
        """Create simple fallback styles when ReportLab is not available"""
        return {
            'title': 'title',
            'subtitle': 'subtitle', 
            'quiz_instructions': 'instructions',
            'quiz_question': 'question',
            'quiz_option': 'option',
            'legal_article': 'article',
            'legal_text': 'text',
            'body': 'normal',
            'emphasis': 'bold'
        }
    
    def get_style(self, style_name: str):
        """Get a style by name"""
        if REPORTLAB_AVAILABLE:
            return self.custom_styles.get(style_name, self.base_styles['Normal'])
        else:
            return self.custom_styles.get(style_name, 'normal')


class HeaderFooterCanvas:
    """Custom canvas for adding headers and footers"""
    
    def __init__(self, *args, **kwargs):
        self.title = kwargs.pop('title', 'Document')
        if REPORTLAB_AVAILABLE:
            canvas.Canvas.__init__(self, *args, **kwargs)
            self.pages = []
        else:
            self.filename = args[0] if args else 'output.pdf'
    
    def showPage(self):
        self.pages.append(dict(self.__dict__))
        self._startPage()
    
    def save(self):
        page_count = len(self.pages)
        for page_num, page in enumerate(self.pages, 1):
            self.__dict__.update(page)
            self.draw_header_footer(page_num, page_count)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)
    
    def draw_header_footer(self, page_num: int, page_count: int):
        """Draw header and footer on each page"""
        # Header
        self.setFont('Helvetica', 10)
        self.drawString(50, letter[1] - 50, self.title)
        self.drawRightString(letter[0] - 50, letter[1] - 50, datetime.now().strftime("%B %d, %Y"))
        
        # Footer
        self.drawCentredText(letter[0] / 2, 30, f"Page {page_num} of {page_count}")
        
        # Draw line under header
        self.setStrokeColor(gray)
        self.line(50, letter[1] - 60, letter[0] - 50, letter[1] - 60)
    
    def drawCentredText(self, x, y, text):
        """Helper method to draw centered text"""
        self.drawString(x - len(text) * 3, y, text)


class EnhancedPDFFormatter:
    """Main PDF formatter with enhanced capabilities"""
    
    def __init__(self):
        self.markdown_processor = MarkdownProcessor()
        self.content_analyzer = ContentAnalyzer()
        self.style_manager = StyleManager()
    
    def format_content(self, content: str, title: str = "Document", output_path: str = "output.pdf") -> str:
        """Format content and generate PDF"""
        if not REPORTLAB_AVAILABLE:
            return self._fallback_pdf_generation(content, title, output_path)
            
        # Analyze content type
        doc_type = self.content_analyzer.detect_document_type(content)
        
        # Create document with custom canvas
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=100,
            bottomMargin=72,
            canvasmaker=lambda *args, **kwargs: HeaderFooterCanvas(*args, title=title, **kwargs)
        )
        
        # Build story based on document type
        if doc_type == 'quiz':
            story = self._build_quiz_story(content, title)
        elif doc_type == 'legal':
            story = self._build_legal_story(content, title)
        elif doc_type == 'report':
            story = self._build_report_story(content, title)
        else:
            story = self._build_general_story(content, title)
        
        # Build PDF
        doc.build(story)
        return output_path
    
    def _fallback_pdf_generation(self, content: str, title: str, output_path: str) -> str:
        """Fallback PDF generation when ReportLab is not available"""
        # Create a simple text file with enhanced formatting
        formatted_content = self._format_content_as_text(content, title)
        
        # Write to text file for now (could be enhanced to use other PDF libraries)
        text_path = output_path.replace('.pdf', '_formatted.txt')
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(formatted_content)
        
        print(f"Warning: ReportLab not available. Generated formatted text file: {text_path}")
        return text_path
    
    def _format_content_as_text(self, content: str, title: str) -> str:
        """Format content as enhanced text when PDF generation is not available"""
        lines = []
        
        # Title
        lines.append("=" * 80)
        lines.append(title.center(80))
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
        lines.append("")
        lines.append("-" * 80)
        lines.append("")
        
        # Process content
        doc_type = self.content_analyzer.detect_document_type(content)
        
        if doc_type == 'quiz':
            lines.extend(self._format_quiz_as_text(content))
        elif doc_type == 'legal':
            lines.extend(self._format_legal_as_text(content))
        else:
            lines.extend(self._format_general_as_text(content))
        
        # Footer
        lines.append("")
        lines.append("-" * 80)
        lines.append("Generated by Enhanced PDF Formatter")
        lines.append("For best results, install ReportLab: pip install reportlab")
        
        return "\n".join(lines)
    
    def _format_quiz_as_text(self, content: str) -> List[str]:
        """Format quiz content as enhanced text"""
        lines = []
        quiz_structure = self.content_analyzer.extract_quiz_structure(content)
        
        # Metadata (title, date, marks)
        if quiz_structure['metadata']:
            for metadata in quiz_structure['metadata']:
                clean_text = re.sub(r'\*\*(.*?)\*\*', r'\1', metadata)
                clean_text = re.sub(r'^#+\s*', '', clean_text)
                
                if 'date:' in clean_text.lower() or 'marks:' in clean_text.lower():
                    lines.append(clean_text)
                else:
                    lines.append(clean_text.upper())
                    lines.append("=" * len(clean_text))
                lines.append("")
        
        # Instructions
        if quiz_structure['instructions']:
            lines.append("INSTRUCTIONS:")
            lines.append("=" * 50)
            for instruction in quiz_structure['instructions']:
                if instruction.upper() not in ["INSTRUCTIONS:", "INSTRUCTIONS"]:
                    lines.append(f"• {instruction}")
            lines.append("")
        
        # Questions
        if quiz_structure['questions']:
            lines.append("QUESTIONS:")
            lines.append("=" * 50)
            lines.append("")
            
            for i, q_data in enumerate(quiz_structure['questions'], 1):
                lines.append(f"{q_data['question']}")
                lines.append("")
                
                # Options (if any)
                if q_data['options']:
                    for option in q_data['options']:
                        lines.append(f"   {option}")
                    lines.append("")
                else:
                    # For descriptive questions, add answer space
                    lines.append("   [Answer space]")
                    lines.append("")
                
                lines.append("-" * 40)
                lines.append("")
        
        # Answers section (if any)
        if quiz_structure['answers']:
            lines.append("")
            lines.append("ANSWERS:")
            lines.append("=" * 50)
            for answer in quiz_structure['answers']:
                lines.append(f"• {answer}")
            lines.append("")
        
        return lines
    
    def _format_legal_as_text(self, content: str) -> List[str]:
        """Format legal content as enhanced text"""
        lines = []
        sections = self.markdown_processor.extract_headings(content)
        
        for level, heading_title, section_content in sections:
            if level > 0:  # It's a heading
                lines.append("")
                if level == 1:
                    lines.append(heading_title.upper())
                    lines.append("=" * len(heading_title))
                elif level == 2:
                    lines.append(heading_title)
                    lines.append("-" * len(heading_title))
                else:
                    lines.append(f"{'  ' * (level-3)}{heading_title}")
                lines.append("")
            else:  # It's content
                paragraphs = section_content.split('\n\n')
                for para in paragraphs:
                    if para.strip():
                        # Check for legal articles
                        if re.match(r'Article \d+', para) or re.match(r'Section \d+', para):
                            lines.append(f"*** {para} ***")
                        else:
                            lines.append(para)
                        lines.append("")
        
        return lines
    
    def _format_general_as_text(self, content: str) -> List[str]:
        """Format general content as enhanced text"""
        lines = []
        
        # Process markdown
        processed_content = self.markdown_processor.process_text(content)
        # Remove HTML-like tags for text output
        processed_content = re.sub(r'<[^>]+>', '', processed_content)
        
        paragraphs = processed_content.split('\n\n')
        for para in paragraphs:
            if para.strip():
                lines.append(para)
                lines.append("")
        
        return lines
    
    def _build_quiz_story(self, content: str, title: str) -> List:
        """Build story elements for quiz documents"""
        story = []
        
        # Extract quiz structure
        quiz_structure = self.content_analyzer.extract_quiz_structure(content)
        
        # Add metadata (title, date, marks) first
        if quiz_structure['metadata']:
            for metadata_line in quiz_structure['metadata']:
                processed_text = self.markdown_processor.process_text(metadata_line)
                # Remove markdown formatting for cleaner display
                clean_text = re.sub(r'\*\*(.*?)\*\*', r'\1', processed_text)
                clean_text = re.sub(r'^#+\s*', '', clean_text)
                
                # Style based on content
                if 'date:' in clean_text.lower():
                    story.append(Paragraph(clean_text, self.style_manager.get_style('body')))
                elif 'marks:' in clean_text.lower() or 'total marks:' in clean_text.lower():
                    story.append(Paragraph(clean_text, self.style_manager.get_style('emphasis')))
                else:
                    # Likely a title
                    story.append(Paragraph(clean_text, self.style_manager.get_style('title')))
                
                story.append(Spacer(1, 8))
        else:
            # Fallback title if no metadata
            story.append(Paragraph(title, self.style_manager.get_style('title')))
            story.append(Spacer(1, 20))
        
        # Instructions
        if quiz_structure['instructions']:
            story.append(Paragraph("INSTRUCTIONS", self.style_manager.get_style('subtitle')))
            story.append(Spacer(1, 10))
            
            for instruction in quiz_structure['instructions']:
                if instruction.upper() != "INSTRUCTIONS:" and instruction.upper() != "INSTRUCTIONS":
                    processed_text = self.markdown_processor.process_text(instruction)
                    story.append(Paragraph(processed_text, self.style_manager.get_style('quiz_instructions')))
            story.append(Spacer(1, 20))
        
        # Questions
        if quiz_structure['questions']:
            story.append(Paragraph("QUESTIONS", self.style_manager.get_style('subtitle')))
            story.append(Spacer(1, 15))
            
            for i, q_data in enumerate(quiz_structure['questions'], 1):
                # Question text
                question_text = self.markdown_processor.process_text(q_data['question'])
                story.append(Paragraph(question_text, self.style_manager.get_style('quiz_question')))
                story.append(Spacer(1, 8))
                
                # Options (if any)
                if q_data['options']:
                    for option in q_data['options']:
                        option_text = self.markdown_processor.process_text(option)
                        story.append(Paragraph(option_text, self.style_manager.get_style('quiz_option')))
                    story.append(Spacer(1, 10))
                else:
                    # For descriptive questions, add some space for answers
                    story.append(Spacer(1, 30))  # Space for written answers
                
                # Add separator between questions (except for last question)
                if i < len(quiz_structure['questions']):
                    story.append(Spacer(1, 10))
        
        # Answers section (if any)
        if quiz_structure['answers']:
            story.append(Spacer(1, 30))
            story.append(Paragraph("ANSWERS", self.style_manager.get_style('subtitle')))
            story.append(Spacer(1, 10))
            
            for answer in quiz_structure['answers']:
                answer_text = self.markdown_processor.process_text(answer)
                story.append(Paragraph(answer_text, self.style_manager.get_style('body')))
                story.append(Spacer(1, 5))
        
        return story
    
    def _build_legal_story(self, content: str, title: str) -> List:
        """Build story elements for legal documents"""
        story = []
        
        # Title
        story.append(Paragraph(title, self.style_manager.get_style('title')))
        story.append(Spacer(1, 20))
        
        # Process content with legal formatting
        sections = self.markdown_processor.extract_headings(content)
        
        for level, heading_title, section_content in sections:
            if level > 0:  # It's a heading
                style_name = f'heading{min(level, 6)}'
                story.append(Paragraph(heading_title, self.style_manager.get_style(style_name)))
                story.append(Spacer(1, 10))
            else:  # It's content
                paragraphs = section_content.split('\n\n')
                for para in paragraphs:
                    if para.strip():
                        # Check for legal articles
                        if re.match(r'Article \d+', para) or re.match(r'Section \d+', para):
                            processed_text = self.markdown_processor.process_text(para)
                            story.append(Paragraph(processed_text, self.style_manager.get_style('legal_article')))
                        else:
                            processed_text = self.markdown_processor.process_text(para)
                            story.append(Paragraph(processed_text, self.style_manager.get_style('legal_text')))
                        story.append(Spacer(1, 8))
        
        return story
    
    def _build_report_story(self, content: str, title: str) -> List:
        """Build story elements for report documents"""
        story = []
        
        # Title
        story.append(Paragraph(title, self.style_manager.get_style('title')))
        story.append(Spacer(1, 20))
        
        # Process with heading extraction
        sections = self.markdown_processor.extract_headings(content)
        
        for level, heading_title, section_content in sections:
            if level > 0:  # It's a heading
                style_name = f'heading{min(level, 6)}'
                story.append(Paragraph(heading_title, self.style_manager.get_style(style_name)))
                story.append(Spacer(1, 12))
            else:  # It's content
                paragraphs = section_content.split('\n\n')
                for para in paragraphs:
                    if para.strip():
                        processed_text = self.markdown_processor.process_text(para)
                        story.append(Paragraph(processed_text, self.style_manager.get_style('body')))
                        story.append(Spacer(1, 10))
        
        return story
    
    def _build_general_story(self, content: str, title: str) -> List:
        """Build story elements for general documents"""
        story = []
        
        # Title
        story.append(Paragraph(title, self.style_manager.get_style('title')))
        story.append(Spacer(1, 20))
        
        # Process content
        sections = self.markdown_processor.extract_headings(content)
        
        for level, heading_title, section_content in sections:
            if level > 0:  # It's a heading
                style_name = f'heading{min(level, 6)}'
                story.append(Paragraph(heading_title, self.style_manager.get_style(style_name)))
                story.append(Spacer(1, 10))
            else:  # It's content
                paragraphs = section_content.split('\n\n')
                for para in paragraphs:
                    if para.strip():
                        processed_text = self.markdown_processor.process_text(para)
                        story.append(Paragraph(processed_text, self.style_manager.get_style('body')))
                        story.append(Spacer(1, 8))
        
        return story