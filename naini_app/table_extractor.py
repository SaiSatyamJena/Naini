# naini_app/table_extractor.py
# REFINED VERSION based on table_extractor_v2.py provided by user
# Addresses coordinate mapping bug, ensures robustness, retains necessary functions.

from __future__ import annotations # Ensure compatibility with older Python versions for type hints

import logging
import pandas as pd
from PIL import Image
import io
import fitz # PyMuPDF
from pathlib import Path
from typing import Tuple, Optional, List # Adjusted imports for clarity
import re
import traceback # Explicitly imported, though exc_info=True often suffices

# --- Configuration ---
# Camelot tuning parameters
CAMELOT_LATTICE_LINE_SCALE = 40
CAMELOT_PROCESS_BACKGROUND = True

# img2table tuning parameters
IMG2TABLE_IMPLICIT_ROWS = False
IMG2TABLE_BORDERLESS = True # Set based on typical PDF table appearance (often without full borders)
IMG2TABLE_MIN_CONFIDENCE = 50 # OCR confidence threshold

# PyMuPDF ToC parsing parameters
PYMUPDF_TOC_PAGE_NUM_COL_START_RATIO = 0.80 # Assume page numbers in right 20%
PYMUPDF_TOC_LINE_GAP_THRESHOLD = 5 # Points threshold to separate multi-line entries

# --- Standard Placeholder for Failures ---
TABLE_FAILURE_PLACEHOLDER = "[TABLE EXTRACTION FAILED OR EMPTY]"

# Configure logger for this module
log = logging.getLogger(__name__) # Use __name__ for standard logger hierarchy

# --- Library Import Handling & Setup ---

# Camelot (Optional)
try:
    import camelot
    CAMELOT_AVAILABLE = True
    log.info("Camelot-py library found and imported.")
except ImportError:
    camelot = None # Ensure camelot is None if import fails
    CAMELOT_AVAILABLE = False
    log.warning("Camelot-py library not found. Camelot table extraction will not be available.")

# img2table & Tesseract (Optional but needed for fallback)
try:
    from img2table.ocr import TesseractOCR
    from img2table.document import Image as Img2TableImage
    from pytesseract import TesseractNotFoundError

    # Initialize Tesseract OCR Instance for img2table only once
    ocr_instance = None
    try:
        # Attempt to initialize TesseractOCR, assuming Tesseract is in PATH
        ocr_instance = TesseractOCR(lang='eng', n_threads=1)
        log.info("TesseractOCR instance initialized successfully for img2table.")
        IMG2TABLE_AVAILABLE = True
    except TesseractNotFoundError:
        ocr_instance = None # Explicitly set to None on failure
        IMG2TABLE_AVAILABLE = False
        log.error(
            "Tesseract executable not found by pytesseract (is it installed and in system PATH?). "
            "img2table extraction fallback will not be available.", exc_info=False
        )
    except Exception as e:
        ocr_instance = None # Explicitly set to None on other init errors
        IMG2TABLE_AVAILABLE = False
        log.error(f"An unexpected error occurred initializing TesseractOCR for img2table: {e}", exc_info=True)

except ImportError:
    # Define placeholders if img2table or pytesseract are missing
    TesseractOCR = None
    Img2TableImage = None
    TesseractNotFoundError = type('TesseractNotFoundError', (Exception,), {}) # Dummy exception class
    ocr_instance = None
    IMG2TABLE_AVAILABLE = False
    log.warning("img2table or pytesseract library not found. img2table extraction fallback will not be available.")

# Import utilities, handling potential ImportError if run standalone
try:
    from .utils import map_image_box_to_pdf_coords
    UTILS_AVAILABLE = True
except ImportError:
    log.warning("Could not import 'map_image_box_to_pdf_coords' from .utils. Coordinate mapping (and thus Camelot) will likely fail.")
    UTILS_AVAILABLE = False
    # Define a dummy function so the rest of the code doesn't crash immediately if run standalone
    # This dummy won't actually work correctly, just prevents NameError
    def map_image_box_to_pdf_coords(*args, **kwargs) -> Optional[fitz.Rect]:
        log.error("Using dummy map_image_box_to_pdf_coords because import failed!")
        return None

# --- Helper Function: DataFrame to Markdown ---
def _df_to_markdown(df: pd.DataFrame) -> Optional[str]:
    """Converts a non-empty Pandas DataFrame to a Markdown table string."""
    if df is None or df.empty:
        log.debug("DataFrame is None or empty, skipping markdown conversion.")
        return None
    # Simple check for tables that are just one cell and effectively empty/whitespace
    if df.shape == (1, 1):
        cell_content = str(df.iloc[0, 0]).strip()
        if not cell_content:
             log.debug("DataFrame is 1x1 with empty content, skipping markdown conversion.")
             return None
    try:
        # Fill NA values with empty strings for cleaner Markdown output
        df_filled = df.fillna('')
        markdown_string = df_filled.to_markdown(index=False)
        # Check if the markdown string itself is empty or just headers/separators
        if not markdown_string or len(markdown_string.splitlines()) < 2: # Need header + separator at minimum
            log.debug("DataFrame resulted in empty or header-only markdown, skipping.")
            return None
        return markdown_string
    except Exception as e:
        log.error(f"Failed to convert DataFrame to Markdown: {e}", exc_info=True)
        return None

# --- ToC/LoF/LoT Parsing (PyMuPDF Text Blocks) ---
# NOTE: This function is intended to be called specifically for ToC-like pages (e.g., 2-7) by main.py
def extract_toc_like_table_pymupdf(pdf_page: fitz.Page, table_box_pdf_coords: fitz.Rect) -> tuple[Optional[str], str]:
    """
    Extracts table-like structure (e.g., ToC, LoF, LoT) using PyMuPDF text blocks.
    Assumes a layout with text descriptions and right-aligned page numbers.

    Args:
        pdf_page: The fitz.Page object.
        table_box_pdf_coords: The fitz.Rect defining the table area in PDF points.

    Returns:
        Tuple: (markdown_string, method_used) or (None, 'failure')
    """
    method_used = "pymupdf-parsing"
    log.info(f"Attempting ToC-like table extraction using PyMuPDF text parsing for region: {table_box_pdf_coords}")

    # Input validation
    if not isinstance(pdf_page, fitz.Page):
        log.error(f"Invalid input: 'pdf_page' must be a fitz.Page object, got {type(pdf_page)}.")
        return None, "failure"
    if not isinstance(table_box_pdf_coords, fitz.Rect) or table_box_pdf_coords.is_empty:
        log.error(f"Invalid input: 'table_box_pdf_coords' must be a valid fitz.Rect, got {table_box_pdf_coords}.")
        return None, "failure"

    try:
        # Extract text blocks, sorted vertically then horizontally
        # Flags can be tuned if needed, but defaults are often reasonable
        flags = fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES # Ignore image blocks
        blocks = pdf_page.get_text("dict", clip=table_box_pdf_coords, sort=True, flags=flags)["blocks"]

        if not blocks:
            log.warning("PyMuPDF found no text blocks in the specified region.")
            return None, "failure"

        parsed_rows = []
        page_width = table_box_pdf_coords.width
        page_num_column_start = table_box_pdf_coords.x0 + page_width * PYMUPDF_TOC_PAGE_NUM_COL_START_RATIO
        # Regex to identify typical page numbers (digits, dots, hyphens) and Roman numerals (lower/upper)
        page_num_regex = re.compile(r'^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$|^[\d.\-]+$', re.IGNORECASE)

        current_row_lines_data = [] # Stores {'text': str, 'page_num': str, 'y1': float} for lines in current conceptual row
        last_line_y1 = -1.0 # Y-coordinate of the bottom of the previous line processed

        for block in blocks:
            if block["type"] == 0: # Text block
                for line in block["lines"]:
                    line_bbox = fitz.Rect(line["bbox"])
                    line_y0 = line_bbox.y0
                    line_y1 = line_bbox.y1

                    # Check if it's a new conceptual row based on vertical gap
                    is_new_row = last_line_y1 > 0 and (line_y0 - last_line_y1) > PYMUPDF_TOC_LINE_GAP_THRESHOLD

                    if is_new_row and current_row_lines_data:
                        # Process the completed row
                        combined_text = " ".join(line_data['text'] for line_data in current_row_lines_data if line_data['text']).strip()
                        # Assume page number belongs to the last line of the conceptual row
                        final_page_num = current_row_lines_data[-1]['page_num']
                        # Clean up text and add row if content exists
                        cleaned_text = re.sub(r'\s+', ' ', combined_text).strip()
                        if cleaned_text or final_page_num:
                            parsed_rows.append({"text": cleaned_text, "page": final_page_num})
                        current_row_lines_data = [] # Reset for the new row

                    # Process spans in the current line to find text and potential page number
                    line_text_parts = []
                    potential_page_num = ""
                    has_text_content = False
                    for span in line["spans"]:
                        span_bbox = fitz.Rect(span["bbox"])
                        span_text = span["text"].strip()

                        if not span_text: continue # Skip empty spans

                        # Check if span is likely a page number
                        if span_bbox.x0 >= page_num_column_start and page_num_regex.match(span_text):
                            potential_page_num = span_text # Keep the rightmost match on the line
                        else:
                            line_text_parts.append(span_text)
                            has_text_content = True

                    # Add processed line info to current row data
                    if has_text_content or potential_page_num:
                         current_row_lines_data.append({
                             'text': " ".join(line_text_parts).strip(),
                             'page_num': potential_page_num,
                             'y1': line_y1 # Store bottom coordinate for gap calculation
                         })

                    last_line_y1 = line_y1 # Update y-coordinate of the bottom of the last processed line

        # Process the very last accumulated row after the loop
        if current_row_lines_data:
            combined_text = " ".join(line_data['text'] for line_data in current_row_lines_data if line_data['text']).strip()
            final_page_num = current_row_lines_data[-1]['page_num']
            cleaned_text = re.sub(r'\s+', ' ', combined_text).strip()
            if cleaned_text or final_page_num:
                parsed_rows.append({"text": cleaned_text, "page": final_page_num})

        if not parsed_rows:
            log.warning("Could not parse any valid rows from PyMuPDF text blocks.")
            return None, "failure"

        # Create DataFrame and format as Markdown
        try:
            df = pd.DataFrame(parsed_rows)
            # Ensure required columns exist, even if empty
            if "text" not in df.columns: df["text"] = ""
            if "page" not in df.columns: df["page"] = ""
            # Rename columns for standard output
            df.rename(columns={'text': 'Description', 'page': 'Page'}, inplace=True)
            # Reorder columns for consistency
            df = df[['Description', 'Page']]

            markdown_string = _df_to_markdown(df)
            if markdown_string:
                 log.info(f"Successfully parsed ToC-like table ({len(df)} rows) using {method_used}.")
                 return markdown_string, method_used
            else:
                 log.warning(f"Parsed {method_used} data resulted in an empty Markdown table.")
                 return None, "failure"
        except Exception as df_e:
            log.error(f"Error converting {method_used} parsed data to DataFrame/Markdown: {df_e}", exc_info=True)
            return None, "failure"

    except Exception as e:
        log.error(f"Error during PyMuPDF table parsing: {e}", exc_info=True)
        return None, "failure"

# --- Camelot Table Extraction ---
def extract_table_camelot(single_page_pdf_path: str, pdf_coords_str: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts table(s) from a specific region of a single-page PDF using Camelot.
    Tries 'lattice' mode first, then falls back to 'stream' mode.

    Args:
        single_page_pdf_path: Path to the single-page PDF file.
        pdf_coords_str: Comma-separated string of PDF coordinates (x0,y1,x1,y0).

    Returns:
        Tuple: (markdown_string, method_used) or (None, None) if failed/empty.
               method_used will be 'camelot-lattice' or 'camelot-stream'.
    """
    method_tag = "camelot"
    log.info(f"Attempting table extraction using Camelot (Lattice -> Stream).")
    log.debug(f"  PDF Path: {single_page_pdf_path}")
    log.debug(f"  Target Coords (pdf_coords_str): '{pdf_coords_str}'")

    if not CAMELOT_AVAILABLE:
        log.warning("Camelot library not available, skipping Camelot extraction.")
        return None, None # Return None for method too

    # Input validation
    pdf_path_obj = Path(single_page_pdf_path)
    if not pdf_path_obj.is_file():
        log.error(f"Camelot Error: PDF file not found or is not a file: {single_page_pdf_path}")
        return None, None
    if not pdf_coords_str or not isinstance(pdf_coords_str, str) or len(pdf_coords_str.split(',')) != 4:
        log.error(f"Camelot Error: Invalid or missing PDF coordinates string (expecting 'x0,y1,x1,y0'): {pdf_coords_str}")
        return None, None

    tables_to_process = None # Will hold camelot.TableList if successful
    used_flavor: Optional[str] = None

    # Attempt 1: Lattice
    try:
        log.info("Trying Camelot with flavor='lattice' (tuned)...")
        # Explicitly provide pdf_path as string
        tables_lattice = camelot.read_pdf(
            filepath=str(pdf_path_obj),
            pages='1', # Always page 1 of the single-page PDF
            flavor='lattice',
            table_regions=[pdf_coords_str],
            suppress_stdout=True,
            line_scale=CAMELOT_LATTICE_LINE_SCALE,
            process_background=CAMELOT_PROCESS_BACKGROUND
            # Note: Camelot might have GS path issues internally on some systems
        )
        log.info(f"Camelot (lattice) raw extraction found {tables_lattice.n} table(s).")
        if tables_lattice.n > 0:
            tables_to_process = tables_lattice
            used_flavor = 'lattice'
        else:
            log.info("Lattice found 0 tables.")

    except ImportError as import_e: # Catch potential Ghostscript errors more specifically if possible
         if "Ghostscript" in str(import_e) or "gs" in str(import_e).lower():
             log.error("Ghostscript not found or not in PATH. Camelot's 'lattice' mode requires Ghostscript. Trying 'stream'.", exc_info=False)
         else:
             log.error(f"Camelot dependency import error during lattice read: {import_e}", exc_info=True)
         # Proceed to try stream
    except Exception as lattice_e:
        log.error(f"Camelot (lattice) processing failed: {lattice_e}", exc_info=True)
        # Proceed to try stream

    # Attempt 2: Stream (Fallback)
    if tables_to_process is None: # Only if lattice didn't succeed
        log.info("Trying Camelot with flavor='stream'.")
        try:
            tables_stream = camelot.read_pdf(
                filepath=str(pdf_path_obj),
                pages='1',
                flavor='stream',
                table_regions=[pdf_coords_str],
                suppress_stdout=True
                # Add stream-specific tuning if needed: edge_tol, row_tol, etc.
            )
            log.info(f"Camelot (stream) raw extraction found {tables_stream.n} table(s).")
            if tables_stream.n > 0:
                tables_to_process = tables_stream
                used_flavor = 'stream'
            else:
                log.info("Stream also found 0 tables.")
                # Both methods failed to find tables
                return None, None
        except Exception as stream_e:
            log.error(f"Camelot (stream) processing failed: {stream_e}", exc_info=True)
            # Both methods failed
            return None, None

    # Process the tables found (either lattice or stream)
    if tables_to_process is None or used_flavor is None:
        # This case should ideally be caught above, but acts as a safeguard
        log.warning("No tables were successfully extracted by Camelot.")
        return None, None

    markdown_tables = []
    empty_table_count = 0
    for i, table in enumerate(tables_to_process):
        df = table.df
        log.debug(f"Processing Camelot table {i+1}/{tables_to_process.n} (flavor: {used_flavor}), Shape: {df.shape}")
        md_table = _df_to_markdown(df) # Use helper function
        if md_table:
            log.debug(f"Successfully converted Camelot table {i+1} to Markdown.")
            markdown_tables.append(md_table)
        else:
            empty_table_count += 1
            log.info(f"Skipping empty/trivial Camelot table {i+1} ({used_flavor}) after Markdown conversion.")

    if not markdown_tables:
        log.info(f"Camelot ({used_flavor}) found {tables_to_process.n} raw tables, but all were empty or failed Markdown conversion.")
        return None, None # Return None for method too

    # Combine multiple tables if found in the same region
    final_markdown = "\n\n".join(markdown_tables)
    full_method_tag = f"{method_tag}-{used_flavor}"
    log.info(f"Successfully extracted {len(markdown_tables)} non-empty table(s) via Camelot (Method: {full_method_tag}).")
    if empty_table_count > 0:
        log.info(f"  ({empty_table_count} other raw tables from Camelot were empty/skipped)")
    return final_markdown, full_method_tag # Return method tag + flavor


# --- img2table Table Extraction (Image-based Fallback) ---
def extract_table_img2table(image_crop: Image.Image) -> Optional[str]:
    """
    Extracts table(s) from a cropped PIL Image using the img2table library.

    Args:
        image_crop: A PIL Image object containing the cropped table area.

    Returns:
        Markdown string of the extracted table(s), or None if failed/empty.
    """
    method_tag = "img2table"
    log.info("Attempting table extraction using img2table (image-based fallback)...")

    if not IMG2TABLE_AVAILABLE:
        log.error("img2table/TesseractOCR not available or not initialized. Skipping img2table extraction.")
        return None
    if not isinstance(image_crop, Image.Image):
        log.error(f"Invalid input: 'image_crop' must be a PIL Image object, got {type(image_crop)}.")
        return None

    img_byte_arr = io.BytesIO()
    try:
        # Save PIL Image to bytes in memory (PNG format is lossless and widely supported)
        image_crop.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0) # Rewind the stream to the beginning

        # Create an img2table Document object
        # Consider `detect_rotation=True` if table crops might be rotated, but it adds overhead
        doc_image = Img2TableImage(src=img_byte_arr, detect_rotation=False)

        # Extract tables using the pre-initialized OCR instance
        # Note: A single image crop might contain multiple distinct tables for img2table
        extracted_tables = doc_image.extract_tables(
            ocr=ocr_instance, # Use the shared instance
            implicit_rows=IMG2TABLE_IMPLICIT_ROWS,
            borderless_tables=IMG2TABLE_BORDERLESS,
            min_confidence=IMG2TABLE_MIN_CONFIDENCE
        )

        log.info(f"img2table raw extraction found {len(extracted_tables)} potential table(s) in the image crop.")
        if not extracted_tables:
            log.info("img2table did not detect any tables in the provided image crop.")
            return None

        markdown_tables = []
        empty_table_count = 0
        for i, table in enumerate(extracted_tables):
            df = table.df
            log.debug(f"Processing img2table table {i+1}/{len(extracted_tables)}, Shape: {df.shape}, Title: {table.title}")
            md_table = _df_to_markdown(df) # Use helper function
            if md_table:
                 # Add identifying header if multiple tables are found in the same single crop
                 # This helps distinguish them in the final markdown output
                 if len(extracted_tables) > 1:
                      table_header = f"**Sub-Table {i+1} (Detected by {method_tag}):**"
                      markdown_tables.append(f"{table_header}\n{md_table}")
                 else:
                      markdown_tables.append(md_table)
                 log.debug(f"Successfully converted img2table table {i+1} to Markdown.")
            else:
                 empty_table_count += 1
                 log.info(f"Skipping empty/trivial img2table table {i+1} after Markdown conversion.")

        if not markdown_tables:
            log.info(f"img2table found {len(extracted_tables)} raw tables, but all were empty or failed Markdown conversion.")
            return None

        # Combine multiple tables if found
        final_markdown = "\n\n".join(markdown_tables)
        log.info(f"Successfully extracted {len(markdown_tables)} non-empty table(s) using {method_tag}.")
        if empty_table_count > 0:
            log.info(f"  ({empty_table_count} other raw tables from {method_tag} were empty/skipped)")
        return final_markdown

    # Catch specific Tesseract error if it occurs during the call (init check should prevent this)
    except TesseractNotFoundError: # Check if this specific exception exists
        log.error("Tesseract executable not found during img2table call (should have been caught at init).", exc_info=False)
        return None
    except Exception as e:
        log.error(f"Unexpected error during img2table extraction: {e}", exc_info=True)
        return None
    finally:
        # Ensure the BytesIO object is closed to release memory
        img_byte_arr.close()


# --- Hybrid Table Extraction Orchestrator ---
def extract_table_hybrid(
    image_crop: Image.Image,
    image_box_px: List[int], # Expecting [x0, y0, x1, y1] in pixel coords
    img_width_px: int,
    img_height_px: int,
    pdf_page: fitz.Page,
    single_page_pdf_path: str,
    img_dpi: int = 300
) -> Tuple[str, str]:
    """
    Orchestrates table extraction: Maps coords, tries Camelot, falls back to img2table.

    Args:
        image_crop: PIL Image object of the cropped table area.
        image_box_px: Bounding box [x0, y0, x1, y1] in image pixel coordinates.
        img_width_px: Width of the original page image in pixels.
        img_height_px: Height of the original page image in pixels.
        pdf_page: fitz.Page object corresponding to the image.
        single_page_pdf_path: Path to the temporary single-page PDF for Camelot.
        img_dpi: DPI used for PDF-to-image rendering (for coord mapping).

    Returns:
        Tuple (result_markdown, method_used):
            - result_markdown (str): Markdown table(s) or TABLE_FAILURE_PLACEHOLDER.
            - method_used (str): Identifier (e.g., 'camelot-lattice', 'img2table', 'failure').
    """
    log.info("--- Starting Hybrid Table Extraction ---")
    log.debug(f"Input image crop size: {image_crop.size if image_crop else 'None'}")
    log.debug(f"Input image box (pixels): {image_box_px}")
    log.debug(f"Original image dimensions (pixels): {img_width_px}x{img_height_px}")
    log.debug(f"Source PDF page number: {pdf_page.number if pdf_page else 'N/A'}")
    log.debug(f"Single page PDF path: {single_page_pdf_path}")
    log.debug(f"Image DPI for mapping: {img_dpi}")

    # --- Input Validation ---
    if not isinstance(image_crop, Image.Image):
        log.error("Hybrid extraction failed: Invalid 'image_crop' input.")
        return TABLE_FAILURE_PLACEHOLDER, 'failure'
    if not (isinstance(image_box_px, list) and len(image_box_px) == 4):
        log.error("Hybrid extraction failed: Invalid 'image_box_px' input.")
        return TABLE_FAILURE_PLACEHOLDER, 'failure'
    if not isinstance(pdf_page, fitz.Page):
        log.error("Hybrid extraction failed: Invalid 'pdf_page' input.")
        return TABLE_FAILURE_PLACEHOLDER, 'failure'
    # ------------------------

    camelot_result_md: Optional[str] = None
    camelot_method: Optional[str] = None # e.g., 'camelot-lattice'
    pdf_coords_str: Optional[str] = None
    pdf_coords_rect: Optional[fitz.Rect] = None # Store the Rect object if needed later

    # 1. Map Image Coordinates to PDF Coordinates for Camelot
    if UTILS_AVAILABLE and map_image_box_to_pdf_coords: # Check if function exists
        try:
            pdf_coords_rect = map_image_box_to_pdf_coords(
                image_box_px=image_box_px,
                img_width_px=img_width_px,
                img_height_px=img_height_px,
                pdf_page=pdf_page,
                img_dpi=img_dpi
            )

            # **CRUCIAL FIX CHECK**: Verify the result is a valid fitz.Rect before proceeding
            if isinstance(pdf_coords_rect, fitz.Rect) and not pdf_coords_rect.is_empty:
                # Format coordinates string required by Camelot: "x0,y1,x1,y0"
                # PDF Y-coords increase upwards. Rect uses y0 < y1. Camelot needs top-y (y1), bottom-y (y0).
                pdf_coords_str = (
                    f"{pdf_coords_rect.x0:.2f},"  # x_min (left)
                    f"{pdf_coords_rect.y1:.2f},"  # y_max (top)
                    f"{pdf_coords_rect.x1:.2f},"  # x_max (right)
                    f"{pdf_coords_rect.y0:.2f}"   # y_min (bottom)
                )
                log.info(f"Successfully mapped image box to PDF coordinates: '{pdf_coords_str}' (Rect: {pdf_coords_rect})")
            elif pdf_coords_rect is None:
                 log.warning("Coordinate mapping function returned None. Cannot attempt Camelot.")
            elif isinstance(pdf_coords_rect, fitz.Rect) and pdf_coords_rect.is_empty:
                 log.warning(f"Coordinate mapping resulted in an empty rectangle ({pdf_coords_rect}). Skipping Camelot.")
            else:
                 # This handles the case where the mapping function might return an unexpected type
                 log.error(f"Coordinate mapping function returned unexpected type: {type(pdf_coords_rect)}. Expected fitz.Rect or None. Skipping Camelot.")
                 pdf_coords_rect = None # Ensure rect is None if invalid

        except Exception as map_e:
            log.error(f"Error occurred during coordinate mapping: {map_e}", exc_info=True)
            pdf_coords_str = None # Ensure coords are None on error
            pdf_coords_rect = None
    else:
        if not UTILS_AVAILABLE or not map_image_box_to_pdf_coords:
            log.warning("Coordinate mapping function not available (import failed?). Skipping Camelot.")
        # pdf_page validity checked earlier

    # 2. Attempt Camelot Extraction (if available, and coordinates are valid)
    if CAMELOT_AVAILABLE and pdf_coords_str:
        log.info(f"Attempting Camelot extraction with mapped PDF coords...")
        try:
            camelot_result_md, camelot_method = extract_table_camelot(
                single_page_pdf_path=single_page_pdf_path,
                pdf_coords_str=pdf_coords_str
            )
            # Check if Camelot succeeded and returned non-empty results
            if camelot_result_md is not None and camelot_method is not None:
                log.info(f"Camelot extraction successful (Method: {camelot_method}). Using this result.")
                return camelot_result_md, camelot_method # SUCCESS via Camelot
            else:
                log.info("Camelot ran but failed to extract a valid/non-empty table.")
        except Exception as camelot_call_e:
            log.error(f"Unexpected error calling extract_table_camelot function: {camelot_call_e}", exc_info=True)
            camelot_result_md = None # Ensure result is None on error
            camelot_method = None
    elif not CAMELOT_AVAILABLE:
        log.info("Camelot is not available. Proceeding to img2table fallback.")
    else: # pdf_coords_str was None
        log.info("Valid PDF coordinates not available. Skipping Camelot, proceeding to img2table fallback.")

    # 3. Fallback to img2table (if Camelot failed, was skipped, or unavailable)
    log.info("Proceeding with img2table fallback extraction on the image crop.")
    if IMG2TABLE_AVAILABLE:
        img2table_result_md: Optional[str] = None
        try:
            img2table_result_md = extract_table_img2table(image_crop)
        except Exception as img2table_call_e:
            log.error(f"Unexpected error calling extract_table_img2table function: {img2table_call_e}", exc_info=True)
            img2table_result_md = None

        if img2table_result_md is not None:
            log.info("img2table fallback extraction successful.")
            return img2table_result_md, 'img2table' # SUCCESS via img2table
        else:
            log.warning("img2table fallback also failed or returned no valid table.")
            return TABLE_FAILURE_PLACEHOLDER, 'failure' # BOTH methods failed
    else:
        log.error("img2table fallback failed: img2table or Tesseract is not available/initialized.")
        return TABLE_FAILURE_PLACEHOLDER, 'failure' # BOTH methods failed (img2table unavailable)


# --- Main Block for Standalone Testing ---
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG, # Set to DEBUG for detailed extractor logs
        format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s'
    )
    log.info("--- Running table_extractor.py Standalone Tests ---")

    # Test Case 1: img2table Fallback Simulation
    print("\n" + "="*10 + " Test Case 1: img2table Fallback Simulation " + "="*10)
    try:
        # Create a dummy image
        width, height = 400, 150
        dummy_image = Image.new('RGB', (width, height), color='white')
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(dummy_image)
        try:
            # Try common fonts, fallback gracefully
            font = ImageFont.truetype("DejaVuSans.ttf", 15)
            log.debug("Loaded DejaVuSans font.")
        except IOError:
            try:
                font = ImageFont.truetype("arial.ttf", 15)
                log.debug("Loaded Arial font.")
            except IOError:
                log.warning("Could not load DejaVuSans or Arial font, using PIL default.")
                font = ImageFont.load_default()

        draw.text((10, 10), "Header A | Header B | Header C", fill="black", font=font)
        draw.line([(0, 30), (width, 30)], fill="black", width=1)
        draw.text((10, 40), "Row1 ColA | 123.45   | Value X ", fill="black", font=font)
        draw.text((10, 60), "Row2 ColA | 67       | Value Y ", fill="black", font=font)
        draw.line([(0, 80), (width, 80)], fill="black", width=1)
        draw.text((10, 90), "Row3 ColA | 8.9      | Value Z ", fill="black", font=font)

        dummy_image_path = Path("./dummy_table_image.png")
        dummy_image.save(dummy_image_path)
        log.info(f"Saved dummy table image to {dummy_image_path}")

        # Simulate inputs for hybrid function, forcing fallback by using None for pdf_page
        mock_box = [5, 5, width - 5, height - 5]
        mock_orig_width, mock_orig_height = 600, 800
        mock_pdf_page = None # This forces Camelot path to be skipped
        mock_pdf_path = "non_existent_page.pdf"

        log.info("Simulating hybrid call where Camelot is skipped (no pdf_page)...")
        result_md, result_method = extract_table_hybrid(
            image_crop=dummy_image, image_box_px=mock_box,
            img_width_px=mock_orig_width, img_height_px=mock_orig_height,
            pdf_page=mock_pdf_page, # Pass None here
            single_page_pdf_path=mock_pdf_path, img_dpi=300
        )

        print(f"\nHybrid Result Method: {result_method}")
        print(f"Hybrid Result Markdown (expecting img2table or failure):\n{result_md}")

        if result_method == 'img2table' and result_md != TABLE_FAILURE_PLACEHOLDER:
             print("\nTest Verdict: PASSED (Hybrid function correctly fell back to img2table).")
        elif result_method == 'failure':
             print("\nTest Verdict: FAILED (Hybrid function failed).")
             if not IMG2TABLE_AVAILABLE:
                 print("  Reason: img2table/Tesseract is likely unavailable (check logs for 'Tesseract not found').")
             else:
                 print("  Reason: Check img2table logs for errors during extraction.")
        else:
             print(f"\nTest Verdict: WARNING (Unexpected method '{result_method}'). Check logs.")

    except ImportError as e:
        print(f"\nERROR: Missing import required for testing (e.g., PIL/Pillow): {e}")
        log.error(f"ImportError during testing: {e}", exc_info=True)
    except TesseractNotFoundError: # Catch specific error if test setup itself fails
        print("\nERROR: Tesseract not found by system. Cannot perform img2table tests.")
        log.error("Tesseract not found during test setup - img2table tests skipped.", exc_info=False)
    except Exception as e:
        print(f"\nUNEXPECTED ERROR during Test Case 1: {e}")
        log.error(f"Error during Test Case 1: {e}", exc_info=True)

    # Test Case 2: Direct ToC Extraction (Placeholder)
    print("\n" + "="*10 + " Test Case 2: Direct ToC Extraction (Requires PDF) " + "="*10)
    # This requires a real PDF file with a ToC structure.
    # pdf_path_toc_test = Path("YOUR_PDF_WITH_TOC.pdf") # Replace with actual path
    # toc_page_number = 1 # Example: assuming ToC is on page 2 (index 1)
    # if pdf_path_toc_test.exists():
    #     log.info(f"Attempting ToC test on {pdf_path_toc_test}, page {toc_page_number + 1}")
    #     doc = None
    #     try:
    #         doc = fitz.open(str(pdf_path_toc_test))
    #         if toc_page_number < len(doc):
    #             toc_page = doc[toc_page_number]
    #             # Use full page rectangle for simplicity, or define a specific ToC area
    #             toc_coords = toc_page.rect
    #             log.info(f"Testing extract_toc_like_table_pymupdf on Page {toc_page_number + 1}, Rect: {toc_coords}")
    #             toc_md, toc_method = extract_toc_like_table_pymupdf(toc_page, toc_coords)
    #             print(f"\nToC Extraction Method: {toc_method}")
    #             print(f"ToC Markdown:\n{toc_md if toc_md else '[ToC Extraction Failed]'}")
    #         else:
    #             print(f"Error: PDF has less than {toc_page_number + 1} pages.")
    #     except Exception as toc_e:
    #         print(f"\nERROR during ToC test: {toc_e}")
    #         log.error(f"Error during ToC test case: {toc_e}", exc_info=True)
    #     finally:
    #         if doc:
    #             doc.close()
    # else:
    #      print(f"Skipping ToC test: PDF not found at '{pdf_path_toc_test}'")
    print("Skipping direct ToC test (requires manual setup with a suitable PDF).")

    # --- End of Tests ---
    log.info("--- Finished table_extractor.py Standalone Tests ---")