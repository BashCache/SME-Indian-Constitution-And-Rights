import fitz  # PyMuPDF
import re
import os

# --- Configuration ---

# Define page margins to ignore headers/footers (as a percentage of page height)
MARGIN_TOP = 0.08  # Ignore top 8% of the page
MARGIN_BOTTOM = 0.08  # Ignore bottom 8% of the page

# List of exact junk strings to filter out
JUNK_STRINGS = [
    "republished",
    "NCERT",
    "Reprint 2025-26",
    "11102CH01",
    "READ A CARTOON", # Catches plain-text "READ A CARTOON" headers
    "Shankar. Copyright: Children's Book Trust.",
    "Copyright Cagle Cartoons.",
    "Indian Constitution at Work" # NEW: Removes the repeating header
]

# Regex pattern for any text that looks like a page number or other junk
# We keep the Chapter 1 title regex to avoid repeating it, but it will appear once
JUNK_PATTERN = re.compile(r"^\d+$|Chapter 1: Constitution: Why and How\?|\"Castle of Cards\".*Cagle Cartoons")

# --- Helper Functions ---

def format_table_as_markdown(table_data):
    """Converts a list-of-lists (table) into a Markdown string."""
    markdown = ""
    if not table_data:
        return markdown

    # Create header
    header = table_data[0]
    markdown += "table starts| " + " | ".join(str(h).strip() if h else "" for h in header) + " |\n"
    markdown += "| " + " | ".join(["---"] * len(header)) + " |\n"

    # Create rows
    for row in table_data[1:]:
        # Replace newlines within a cell to keep the table structure
        cleaned_row = [str(cell).strip().replace("\n", " ") if cell else "" for cell in row]
        markdown += "| " + " | ".join(cleaned_row) + " |\n"
    
    return markdown + "\n" + "table ends\n"

def is_block_in_rect(block_bbox, rect):
    """Checks if a text block is contained within a given rectangle (like a table)."""
    b_x0, b_y0, b_x1, b_y1 = block_bbox
    r_x0, r_y0, r_x1, r_y1 = rect
    # Check for intersection
    return (b_x0 < r_x1 and b_x1 > r_x0 and b_y0 < r_y1 and b_y1 > r_y0)

def process_ncert_pdf(pdf_path, output_txt_path):
    """
    Extracts clean text and tables from an NCERT PDF, filtering out junk.
    """
    print(f"Opening PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    final_content = []
    stop_processing = False # Flag to stop processing when "Exercises" is found

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_height = page.rect.height
        
        # Define the content area to ignore headers and footers
        content_y_start = page_height * MARGIN_TOP
        content_y_end = page_height * (1.0 - MARGIN_BOTTOM)
        
        page_content = []
        
        # 1. Find and process tables first
        table_bboxes = []
        tables = page.find_tables()
        if tables:
            for table in tables:
                table_bboxes.append(table.bbox)
                table_data = table.extract()

                # --- NEW: JUNK TABLE FILTER ---
                # Check if this is an unwanted "cartoon" table by looking at its content
                is_junk_table = False
                if table_data and table_data[0] and table_data[0][0]:
                    first_cell_text = str(table_data[0][0]).strip()
                    if "READ A CARTOON" in first_cell_text or \
                       "Countries of the European Union" in first_cell_text or \
                       "The writing of the new Iraqi constitution" in first_cell_text:
                        is_junk_table = True

                if is_junk_table:
                    print(f"  Skipping (cartoon table) on page {page_num + 1}")
                    continue  # Skip this table and move to the next one
                # --- END NEW FILTER ---

                print(f"  Found and processed table on page {page_num + 1}")
                md_table = format_table_as_markdown(table_data)
                page_content.append(md_table)

        # 2. Process text blocks
        blocks = page.get_text("blocks")
        for block in blocks:
            block_bbox = block[:4]
            block_text = block[4].strip()
            
            # --- NEW: Check for "Exercises" ---
            # If we find the "Exercises" header, set the flag and stop processing blocks on this page
            if block_text.startswith("Exercises"):
                print(f"\nFound 'Exercises' on page {page_num + 1}. Stopping extraction.")
                stop_processing = True
                break # Stop processing blocks on this page
            # --- END NEW CHECK ---

            # --- Filtering Logic ---

            # Filter 1: Skip if empty
            if not block_text:
                continue

            # Filter 2: Skip if block is in the header/footer margin
            block_y_middle = (block_bbox[1] + block_bbox[3]) / 2
            if block_y_middle < content_y_start or block_y_middle > content_y_end:
                continue
            
            # Filter 3: Skip if it's one of the junk strings
            if any(junk_str in block_text for junk_str in JUNK_STRINGS):
                continue

            # Filter 4: Skip if it matches the junk regex pattern
            # We check the *first* block only for the chapter title, to allow it once.
            if page_num > 0 and JUNK_PATTERN.match(block_text):
                continue

            # Filter 5: Skip if this text is inside a table we already processed
            in_table = False
            for table_rect in table_bboxes:
                if is_block_in_rect(block_bbox, table_rect):
                    in_table = True
                    break
            if in_table:
                continue
            
            # --- End of Filtering ---
            
            # If all checks pass, add the clean text
            page_content.append(block_text)

        # Add processed content for this page to the final list
        if page_content:
            # We use a single newline join here for tighter text,
            # and double newline join for separating pages.
            final_content.append("\n".join(page_content))
            
        # --- NEW: Check flag to stop page loop ---
        if stop_processing:
            break # Stop processing subsequent pages
        # --- END NEW CHECK ---

    doc.close()
    
    # 3. Save the final clean text
    final_text = "\n\n".join(final_content)
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(final_text)
        
    print(f"\nSuccessfully extracted and cleaned text.")
    print(f"Output saved to: {output_txt_path}")
    if os.path.exists(pdf_path):
        print(f"Original size: {os.path.getsize(pdf_path) / 1024:.2f} KB")
    print(f"Extracted text size: {len(final_text.encode('utf-8')) / 1024:.2f} KB")

# --- Main execution ---
if __name__ == "__main__":
    BOOKS_FOLDER = os.path.join("data", "pdf")
    OUTPUT_FOLDER = os.path.join("extracted_data", "pdf_extracted")
    
    # Create output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Process all PDF files in the books folder
    pdf_files = [f for f in os.listdir(BOOKS_FOLDER) 
                if f.lower().endswith('.pdf') and 
                os.path.isfile(os.path.join(BOOKS_FOLDER, f))]
    
    if not pdf_files:
        print(f"No PDF files found in {BOOKS_FOLDER}")
    else:
        print(f"Found {len(pdf_files)} PDF files to process")
        for pdf_file in pdf_files:
            input_path = os.path.join(BOOKS_FOLDER, pdf_file)
            base_name = os.path.splitext(pdf_file)[0]
            output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_extracted.txt")
            
            print(f"\nProcessing: {pdf_file}")
            process_ncert_pdf(input_path, output_path)

