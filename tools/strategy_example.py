"""
Example demonstrating the Strategy Pattern implementation in DocumentGenerationTool

This example shows how the tool uses the Strategy pattern with an interface
to route document generation to the appropriate generator class.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.document_generation_tool import (
    DocumentGeneratorFactory,
    create_document_generation_tool
)
from tools.generators.document_generator import DocumentGenerator
from tools.document_generator_factory import DocumentGeneratorFactory

def example_using_interface_directly():
    """Example: Using generators directly through the interface"""
    print("=" * 60)
    print("Example 1: Using DocumentGenerator Interface Directly")
    print("=" * 60)
    
    content = """
    # Fundamental Rights
    
    The Indian Constitution guarantees several fundamental rights.
    
    ## Right to Equality
    
    Article 14 guarantees equality before law.
    """
    
    # Use common interface methods
    paragraphs = DocumentGenerator.split_into_paragraphs(content)
    print(f"\nSplit content into {len(paragraphs)} paragraphs:")
    for i, para in enumerate(paragraphs, 1):
        print(f"  {i}. {para[:50]}...")
    
    # Check if text is a heading
    is_heading = DocumentGenerator.is_heading("# Main Title")
    print(f"\n'# Main Title' is a heading: {is_heading}")
    
    # Extract heading level
    level, text = DocumentGenerator.extract_heading_level("## Section Title")
    print(f"Heading level: {level}, Text: {text}")


def example_using_factory():
    """Example: Using the Factory to get appropriate generator"""
    print("\n" + "=" * 60)
    print("Example 2: Using DocumentGeneratorFactory (Strategy Pattern)")
    print("=" * 60)
    
    content = "This is test content for document generation."
    output_dir = "generated_documents"
    
    # Factory automatically selects the right generator
    docx_gen = DocumentGeneratorFactory.create_generator('docx')
    pptx_gen = DocumentGeneratorFactory.create_generator('pptx')
    pdf_gen = DocumentGeneratorFactory.create_generator('pdf')
    
    print("\n✓ Created generators via factory:")
    print(f"  - DOCX Generator: {type(docx_gen).__name__}")
    print(f"  - PPTX Generator: {type(pptx_gen).__name__}")
    print(f"  - PDF Generator: {type(pdf_gen).__name__}")
    
    # All generators implement the same interface
    print("\n✓ All generators share common methods:")
    print(f"  - split_into_paragraphs: {hasattr(docx_gen, 'split_into_paragraphs')}")
    print(f"  - generate: {hasattr(docx_gen, 'generate')}")
    print(f"  - is_heading: {hasattr(docx_gen, 'is_heading')}")


def example_using_tool_with_strategy():
    """Example: Using the tool which internally uses Strategy pattern"""
    print("\n" + "=" * 60)
    print("Example 3: Using DocumentGenerationTool (Routes to Strategy)")
    print("=" * 60)
    
    content = """
    # Indian Constitution
    
    The Constitution of India is the supreme law of India.
    
    ## Preamble
    
    We, the people of India, having solemnly resolved...
    """
    
    tool = create_document_generation_tool()
    
    # The tool internally uses the factory to route to the right generator
    print("\nGenerating documents (tool routes to appropriate generator):")
    
    # DOCX - routes to DocxGenerator
    result = tool.run({
        'content': content,
        'document_type': 'docx',
        'title': 'Constitution Overview',
        'output_path': 'generated_documents/strategy_test.docx'
    })
    print(f"  DOCX: {result}")
    
    # PPTX - routes to PptxGenerator
    result = tool.run({
        'content': content,
        'document_type': 'pptx',
        'title': 'Constitution Overview',
        'output_path': 'generated_documents/strategy_test.pptx'
    })
    print(f"  PPTX: {result}")
    
    # PDF - routes to PdfGenerator
    result = tool.run({
        'content': content,
        'document_type': 'pdf',
        'title': 'Constitution Overview',
        'output_path': 'generated_documents/strategy_test.pdf'
    })
    print(f"  PDF: {result}")


def example_extending_with_new_generator():
    """Example: How to extend the system with a new generator type"""
    print("\n" + "=" * 60)
    print("Example 4: Extending with New Generator (Future Extension)")
    print("=" * 60)
    
    print("""
    To add a new document type (e.g., HTML):
    
    1. Create a new generator class:
    
    class HtmlGenerator(DocumentGenerator):
        def generate(self, content, output_path, title=None, author=None, subject=None):
            # Implementation for HTML generation
            pass
    
    2. Register it with the factory:
    
    DocumentGeneratorFactory.register_generator(DocumentType.HTML, HtmlGenerator)
    
    3. Use it:
    
    generator = DocumentGeneratorFactory.create_generator('html')
    generator.generate(content, 'output.html')
    """)


def example_design_benefits():
    """Example: Demonstrating design pattern benefits"""
    print("\n" + "=" * 60)
    print("Example 5: Design Pattern Benefits")
    print("=" * 60)
    
    print("""
    Benefits of Strategy Pattern Implementation:
    
    1. Separation of Concerns:
       - Each generator handles its own format
       - Common logic (paragraph splitting) in interface
       
    2. Open/Closed Principle:
       - Open for extension (new generators)
       - Closed for modification (existing code unchanged)
       
    3. Single Responsibility:
       - Each generator has one job
       - Factory handles routing
       - Tool handles LangChain integration
       
    4. Easy Testing:
       - Test each generator independently
       - Test factory separately
       - Mock generators in tool tests
       
    5. Maintainability:
       - Changes to one format don't affect others
       - Common methods in one place
       - Clear interface contract
    """)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DocumentGenerationTool - Strategy Pattern Examples")
    print("=" * 60)
    
    try:
        example_using_interface_directly()
        example_using_factory()
        example_using_tool_with_strategy()
        example_extending_with_new_generator()
        example_design_benefits()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

