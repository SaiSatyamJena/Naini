# /workspace/visualize_coords.py
import fitz  # PyMuPDF
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Hardcoded Values for Page 107 ---
PDF_STEM = "Copy of report for FS 2020-21"
PAGE_NUMBER = 107
# Format: Matches the file naming convention used in pdf_processor.py
SINGLE_PAGE_PDF_FILENAME = f"page_{PAGE_NUMBER:03d}.pdf"
# These are the coordinates calculated by test_table_extraction.py for page 107
COORDINATES_STR = "35.03,125.25,536.59,308.75"

# --- Define Paths using pathlib ---
WORKSPACE_DIR = Path("/workspace")
SINGLE_PAGE_PDF_DIR = WORKSPACE_DIR / "data" / "persistent_single_pdfs" / PDF_STEM

INPUT_PDF_PATH = SINGLE_PAGE_PDF_DIR/SINGLE_PAGE_PDF_FILENAME
OUTPUT_DIR = Path("/workspace/data/viz dump")
OUTPUT_PDF_PATH = OUTPUT_DIR / f"page_{PAGE_NUMBER:04d}_lattice_box_viz.pdf"
# --- End Hardcoded Values ---

def draw_rectangle_on_page(pdf_path: Path, page_num_in_pdf: int, coords_str: str, output_path: Path):
    """Draws a rectangle on a specific PDF page based on coordinates."""
    if not pdf_path.is_file():
        log.error(f"Input PDF not found: {pdf_path}")
        return

    doc = None
    try:
        doc = fitz.open(str(pdf_path))
        # For single-page PDFs, the page index is always 0
        if page_num_in_pdf < 1 or page_num_in_pdf > len(doc):
             log.error(f"Page number {page_num_in_pdf} is out of range (1-{len(doc)}).")
             return

        # Use index 0 for single-page PDF
        page_index = page_num_in_pdf - 1 # Usually 0
        page = doc[page_index]

        # Parse coordinates
        try:
            x1, y1, x2, y2 = map(float, coords_str.split(','))
            rect = fitz.Rect(x1, y1, x2, y2)
            log.info(f"Drawing rectangle: {rect} on page index {page_index} of {pdf_path.name}")
        except Exception as e:
            log.error(f"Could not parse coordinates '{coords_str}': {e}")
            return

        # Draw the rectangle (red outline)
        page.draw_rect(rect, color=(1, 0, 0), width=1.5) # Red color, 1.5pt width

        # Save the modified PDF
        output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure output dir exists
        doc.save(str(output_path), garbage=4, deflate=True)
        log.info(f"Saved PDF with rectangle to: {output_path}")

    except Exception as e:
        log.error(f"Error processing PDF {pdf_path}: {e}", exc_info=True)
    finally:
        if doc:
            doc.close()

if __name__ == "__main__":
    log.info(f"Visualizing coordinates '{COORDINATES_STR}' for page {PAGE_NUMBER}...")
    log.info(f"Input PDF: {INPUT_PDF_PATH}")
    log.info(f"Output PDF: {OUTPUT_PDF_PATH}")

    # Call the function with the hardcoded values
    # The page number passed is 1 because Camelot/Fitz expects a 1-based page index within the file,
    # even though our single-page PDF only contains one page (at index 0 internally).
    draw_rectangle_on_page(INPUT_PDF_PATH, 1, COORDINATES_STR, OUTPUT_PDF_PATH)