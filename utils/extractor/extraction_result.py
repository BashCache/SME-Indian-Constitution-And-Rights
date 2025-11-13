from dataclasses import dataclass
from typing import Optional, List, Dict

@dataclass
class ExtractionResult:
    """Standard structured output from all extractors."""
    content: str                     # full extracted text
    metadata: Optional[Dict] = None  # e.g., filename, pages, slide count
    chunks: Optional[List[str]] = None  # optional pre-split chunks for RAG
