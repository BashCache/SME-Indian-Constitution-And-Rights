# DocumentGenerationTool

A LangChain-compatible tool for generating documents in DOCX, PPTX, and PDF formats from text content.

## Architecture

The tool uses the **Strategy Pattern** with an interface-based design:

- **`DocumentGenerator`** (Interface/Abstract Base Class): Defines the contract and common methods
  - `split_into_paragraphs()`: Common utility for splitting content
  - `split_into_slides()`: Common utility for presentation slide splitting
  - `is_heading()`: Common utility for detecting headings
  - `generate()`: Abstract method implemented by each generator

- **Concrete Generators**: 
  - `DocxGenerator`: Generates DOCX documents
  - `PptxGenerator`: Generates PPTX presentations
  - `PdfGenerator`: Generates PDF documents

- **`DocumentGeneratorFactory`**: Routes to the appropriate generator based on document type

- **`DocumentGenerationTool`**: LangChain-compatible tool that uses the factory to delegate generation

## Features

- **Strategy Pattern**: Clean separation of concerns with interface-based design
- **DOCX Generation**: Create Microsoft Word documents with proper formatting, headings, and paragraphs
- **PPTX Generation**: Generate PowerPoint presentations with automatically split slides
- **PDF Generation**: Create PDF documents with styled content and proper formatting
- **LangChain Integration**: Fully compatible with LangChain agents and tools
- **Flexible Input**: Supports markdown-style headings and automatic content structuring
- **Extensible**: Easy to add new document types by implementing the `DocumentGenerator` interface

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install python-docx python-pptx reportlab langchain pydantic
```

## Usage

### Basic Usage

```python
from tools.document_generation_tool import DocumentGenerationTool

# Create tool instance
tool = DocumentGenerationTool(output_directory="generated_documents")

# Generate a DOCX document
result = tool.run({
    'content': 'Your text content here...',
    'document_type': 'docx',
    'title': 'My Document',
    'author': 'Author Name',
    'subject': 'Document Subject'
})

print(result)  # Prints: "Document generated successfully at: path/to/file.docx"
```

### Supported Document Types

- `'docx'`: Microsoft Word document
- `'pptx'`: Microsoft PowerPoint presentation
- `'pdf'`: PDF document

### Input Parameters

- `content` (required): Text content to convert into a document
- `document_type` (required): Type of document ('docx', 'pptx', or 'pdf')
- `output_path` (optional): Custom output file path. If not provided, a timestamped filename will be generated
- `title` (optional): Document title
- `author` (optional): Author name for document metadata
- `subject` (optional): Subject/topic for document metadata

### LangChain Agent Integration

```python
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from tools.document_generation_tool import create_document_generation_tool

# Create the tool
doc_tool = create_document_generation_tool()

# Add to agent's tool list
tools = [doc_tool]

# Initialize agent (example)
# agent = initialize_agent(
#     tools=tools,
#     llm=OpenAI(),
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True
# )
```

### Content Formatting

The tool automatically handles:

- **Markdown-style headings**: Lines starting with `#` are converted to headings
- **Paragraph splitting**: Content is intelligently split into paragraphs
- **Slide generation** (for PPTX): Content is automatically split into appropriate slide lengths
- **Text formatting**: Proper spacing, indentation, and alignment

### Example: Generating All Formats

```python
from tools.document_generation_tool import create_document_generation_tool

content = """
# Main Title

This is the introduction paragraph.

## Section 1

Content for section 1 goes here.

## Section 2

Content for section 2 goes here.
"""

tool = create_document_generation_tool()

# Generate DOCX
tool.run({
    'content': content,
    'document_type': 'docx',
    'title': 'My Report',
    'author': 'SME Agent'
})

# Generate PPTX
tool.run({
    'content': content,
    'document_type': 'pptx',
    'title': 'My Presentation'
})

# Generate PDF
tool.run({
    'content': content,
    'document_type': 'pdf',
    'title': 'My PDF Document'
})
```

## File Structure

```
tools/
├── __init__.py
├── document_generation_tool.py      # Main tool and factory
├── document_generator_base.py       # Base interface/ABC
├── generators/                      # Generator implementations
│   ├── __init__.py
│   ├── docx_generator.py           # DOCX generator
│   ├── pptx_generator.py           # PPTX generator
│   └── pdf_generator.py            # PDF generator
├── example_usage.py                 # Basic usage examples
├── strategy_example.py              # Strategy pattern examples
├── test_tool.py                     # Test script
└── README.md                        # This file
```

## Output Directory

By default, generated documents are saved in the `generated_documents` directory. You can specify a custom directory when creating the tool:

```python
tool = DocumentGenerationTool(output_directory="my_custom_directory")
```

## Error Handling

The tool includes error handling for:

- Invalid document types
- File system errors
- Missing dependencies
- Invalid content formats

## Dependencies

- `python-docx`: For DOCX generation
- `python-pptx`: For PPTX generation
- `reportlab`: For PDF generation
- `langchain`: For LangChain integration
- `pydantic`: For input validation

## Notes

- PPTX presentations automatically split content into slides based on length
- PDF generation uses A4 page size with standard margins
- DOCX documents include proper paragraph formatting and indentation
- All documents support metadata (title, author, subject)

## Design Pattern Benefits

1. **Separation of Concerns**: Each generator handles its own format independently
2. **Open/Closed Principle**: Easy to extend with new document types without modifying existing code
3. **Single Responsibility**: Each class has one clear purpose
4. **Maintainability**: Changes to one format don't affect others
5. **Testability**: Each generator can be tested independently

## Using the Strategy Pattern Directly

You can also use the generators directly through the interface:

```python
from tools.document_generation_tool import DocumentGeneratorFactory

# Get the appropriate generator
generator = DocumentGeneratorFactory.create_generator('docx')

# Use common interface methods
paragraphs = DocumentGenerator.split_into_paragraphs(content)

# Generate document
generator.generate(content, 'output.docx', title='My Document')
```

## Extending with New Document Types

To add a new document type:

1. Create a new generator class implementing `DocumentGenerator`:

```python
from tools.document_generation_tool import DocumentGenerator, DocumentType

class HtmlGenerator(DocumentGenerator):
    def generate(self, content, output_path, title=None, author=None, subject=None):
        # Your HTML generation logic
        pass
```

2. Register it with the factory:

```python
from tools.document_generation_tool import DocumentGeneratorFactory, DocumentType

DocumentGeneratorFactory.register_generator(DocumentType.HTML, HtmlGenerator)
```

3. Use it:

```python
generator = DocumentGeneratorFactory.create_generator('html')
generator.generate(content, 'output.html')
```

## Example Scripts

- **`example_usage.py`**: Basic usage examples
- **`strategy_example.py`**: Examples demonstrating the Strategy pattern implementation

Run the examples:

```bash
python tools/example_usage.py
python tools/strategy_example.py
```

