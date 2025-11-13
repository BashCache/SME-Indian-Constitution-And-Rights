from docx import Document
from utils.extractor.base_extractor import BaseExtractor
from utils.extractor.extraction_result import ExtractionResult

class DOCXExtractor(BaseExtractor):
    """Extractor for DOCX/DOC files."""

    def extract(self, file_path: str) -> ExtractionResult:
        text = []
        try:
            doc = Document(file_path)
            for para in doc.paragraphs:
                if para.text.strip():
                    text.append(para.text.strip())

            content = "\n".join(text)
            metadata = {"file_type": "docx", "paragraphs": len(text)}
            return ExtractionResult(content=content, metadata=metadata)

        except Exception as e:
            return ExtractionResult(content="", metadata={"error": str(e), "file_type": "docx"})
