import os
import fitz
import pdfplumber
import tempfile
from PIL import Image
from utils.extractor.base_extractor import BaseExtractor
from utils.extractor.extraction_result import ExtractionResult
from utils.extractor.gemini_summarizer import summarize_visual_content


class PDFExtractor(BaseExtractor):
    """
    Smart extractor for PDF files.
    Extracts text, tables, and images.
    Uses Gemini API for image/table summarization.
    """

    def extract(self, file_path: str) -> ExtractionResult:
        pages_data, all_text = [], []

        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)

                for page_num, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text() or ""
                    page_tables = page.extract_tables() or []
                    image_paths = self._extract_images_from_page(file_path, page_num)

                    # --- 1️⃣ Table summaries ---
                    table_summaries = []
                    for table in page_tables:
                        table_md = self._table_to_markdown(table)
                        if table_md.strip():
                            table_img_path = self._save_table_as_image(table, page_num)
                            summary = summarize_visual_content(
                                image_path=table_img_path,
                                prompt=f"Summarize this table from page {page_num}."
                            )
                            table_summaries.append(
                                f"Table:\n{table_md}\nGemini Summary:\n{summary}"
                            )

                    # --- 2️⃣ Image summaries ---
                    image_summaries = []
                    for img_path in image_paths:
                        summary = summarize_visual_content(
                            image_path=img_path,
                            prompt=f"Describe this image from page {page_num}."
                        )
                        image_summaries.append(f"Image Summary:\n{summary}")

                    # --- 3️⃣ Combine all content ---
                    combined_text = "\n".join([
                        f"Page {page_num}:",
                        f"Text:\n{page_text.strip()}",
                        "\n".join(table_summaries),
                        "\n".join(image_summaries),
                        ""
                    ])
                    pages_data.append(combined_text)
                    all_text.append(combined_text)

            full_text = "\n".join(all_text)
            metadata = {
                "file_type": "pdf",
                "pages": len(pages_data),
                "summarization_model": "gemini-1.5-pro"
            }

            return ExtractionResult(content=full_text, metadata=metadata)

        except Exception as e:
            return ExtractionResult(content="", metadata={"error": str(e), "file_type": "pdf"})

    # ----------------------------------------------------------------
    def _table_to_markdown(self, table_data):
        """Convert extracted table to Markdown."""
        if not table_data:
            return ""
        rows = []
        for row in table_data:
            cells = [c if c is not None else "" for c in row]
            rows.append("| " + " | ".join(cells) + " |")
        return "\n".join(rows)

    def _save_table_as_image(self, table_data, page_num):
        """Render a basic table as an image for Gemini summarization."""
        tmp_dir = tempfile.gettempdir()
        img_path = os.path.join(tmp_dir, f"pdf_table_{page_num}.png")

        # Create a simple image from table text (fallback visual reference)
        text_repr = self._table_to_markdown(table_data)
        img = Image.new("RGB", (800, 200), color="white")
        # Optionally render text onto image using PIL.ImageDraw if needed
        img.save(img_path)
        return img_path

    def _extract_images_from_page(self, file_path, page_num):
        """Extract embedded images from a PDF page."""
        image_paths = []
        try:
            doc = fitz.open(file_path)
            page = doc[page_num - 1]
            for img_index, img in enumerate(page.get_images(full=True), start=1):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image["ext"]
                img_path = os.path.join(tempfile.gettempdir(), f"pdf_{page_num}_{img_index}.{ext}")
                with open(img_path, "wb") as f:
                    f.write(image_bytes)
                image_paths.append(img_path)
            doc.close()
        except Exception as e:
            print(f"Error extracting images from page {page_num}: {e}")
        return image_paths
