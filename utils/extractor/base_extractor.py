from abc import ABC, abstractmethod
from utils.extractor.extraction_result import ExtractionResult

class BaseExtractor(ABC):
    """Abstract base class for all file extractors."""

    @abstractmethod
    def extract(self, file_path: str) -> ExtractionResult:
        """Extract text or meaningful content from the given file."""
        pass
