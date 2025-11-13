from utils.extractor.base_extractor import BaseExtractor
from utils.extractor.extraction_result import ExtractionResult

class TextExtractor(BaseExtractor):
    """Extractor for plain text files (.txt)."""

    def extract(self, file_path: str) -> ExtractionResult:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            metadata = {"file_type": "txt", "length": len(content)}
            return ExtractionResult(content=content, metadata=metadata)
        except Exception as e:
            return ExtractionResult(content="", metadata={"error": str(e), "file_type": "txt"})
