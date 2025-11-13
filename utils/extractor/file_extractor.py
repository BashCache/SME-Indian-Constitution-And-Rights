import os
from utils.extractor.pdf_extractor import PDFExtractor
from utils.extractor.docx_extractor import DOCXExtractor
from utils.extractor.pptx_extractor import PPTXExtractor
from utils.extractor.text_extractor import TextExtractor
from utils.extractor.extraction_result import ExtractionResult

class FileExtractor:
    """Main entrypoint that delegates extraction to format-specific extractors."""

    def __init__(self):
        self.extractors = {
            ".pdf": PDFExtractor(),
            ".docx": DOCXExtractor(),
            ".doc": DOCXExtractor(),
            ".pptx": PPTXExtractor(),
            ".ppt": PPTXExtractor(),
            ".txt": TextExtractor(),
        }

    def extract_text(self, file_path: str) -> ExtractionResult:
        ext = os.path.splitext(file_path)[1].lower()
        extractor = self.extractors.get(ext)
        if not extractor:
            return ExtractionResult(content="", metadata={"error": f"Unsupported file type: {ext}"})
        return extractor.extract(file_path)
