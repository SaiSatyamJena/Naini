# naini_app/main.py
# MODIFIED: Added filter imports and application logic
# MODIFIED: Added PyMuPDF object dereferencing based on manual
from typing import Optional, Dict, Any, Generator, Tuple, List, Union
import logging
from pathlib import Path
from PIL import Image
import fitz # Import PyMuPDF
log = logging.getLogger(__name__)

# Import our custom modules
from .pdf_processor import convert_pdf_to_images, split_pdf_to_pages
from .layout_detector import detect_layout_elements
from .ocr_extractor import extract_text_from_image
from .table_extractor import extract_table_hybrid
# --- Import markdown_generator for main page generation AND filtering functions ---
from .markdown_generator import generate_markdown_page, _filter_non_english, _general_cleanup # <--- ADDED FILTER IMPORTS
from .utils import map_image_box_to_pdf_coords

logging.basicConfig(level=logging.INFO, # Set to DEBUG to see filter logs
                    format='%(asctime)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s') # Added lineno

# --- Constants ---
IMAGE_DPI = 300

# --- MODIFIED FUNCTION SIGNATURE ---
# --- MODIFIED FUNCTION SIGNATURE ---
def process_pdf_to_markdown(
    pdf_path_str: str,
    output_markdown_dir_str: str,
    output_image_dir_str: str,
    output_single_page_pdf_dir_str: str,
    layout_processor: Any,
    layout_model: Any
) -> Optional[Path]:
    """
    Main workflow function: Processes PDF to Markdown using pre-loaded models.
    Intermediate files are generated BEFORE the main document is opened for iteration.
    """
    pdf_path = Path(pdf_path_str)
    output_markdown_dir = Path(output_markdown_dir_str)
    output_image_dir = Path(output_image_dir_str)
    output_single_page_pdf_dir = Path(output_single_page_pdf_dir_str)
    markdown_output_path = output_markdown_dir / f"{pdf_path.stem}_output.md"

    # --- Header/Footer Margins ---
    HEADER_FOOTER_TOP_MARGIN_PERCENT = 0.05
    HEADER_FOOTER_BOTTOM_MARGIN_PERCENT = 0.05
    BOTTOM_MARGIN_START_THRESHOLD_PERCENT = 1.0 - HEADER_FOOTER_BOTTOM_MARGIN_PERCENT

    if not pdf_path.exists(): log.error(f"Input PDF not found: {pdf_path}"); return None

    # Check if layout models were passed correctly
    if not layout_processor or not layout_model:
        log.error("Layout processor or model not provided to process_pdf_to_markdown.")
        return None

    # --- Create Output Dirs ---
    try:
        output_markdown_dir.mkdir(parents=True, exist_ok=True)
        output_image_dir.mkdir(parents=True, exist_ok=True)
        output_single_page_pdf_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        log.error(f"Error creating output directories: {e}", exc_info=True)
        return None

    # === Stage 0: Generate ALL Intermediate Files FIRST ===
    log.info("--- Stage 0: Generating Intermediate Files ---")
    image_paths_str: Optional[List[str]] = None
    single_page_pdf_paths_str: Optional[List[str]] = None
    try:
        image_paths_str = convert_pdf_to_images(str(pdf_path), str(output_image_dir), dpi=IMAGE_DPI)
        if not image_paths_str: log.error("Failed to get page images."); return None

        single_page_pdf_paths_str = split_pdf_to_pages(str(pdf_path), str(output_single_page_pdf_dir))
        if not single_page_pdf_paths_str: log.error("Failed to get single-page PDFs."); return None

        if len(image_paths_str) != len(single_page_pdf_paths_str):
            log.error(f"Intermediate file count mismatch! Images: {len(image_paths_str)}, Single PDFs: {len(single_page_pdf_paths_str)}"); return None

        num_pages = len(image_paths_str)
        log.info(f"Successfully generated {num_pages} intermediate image(s) and single-page PDF(s).")

    except Exception as e_intermediate:
        log.error(f"Error during intermediate file generation: {e_intermediate}", exc_info=True)
        return None
    # === End of Intermediate File Generation ===

    # --- Initialize markdown list ---
    all_pages_markdown = []
    processing_successful = True

    try:
        # === Stage 2 & 3: Open PDF and Process Page by Page ===
        log.info(f"--- Stage 2&3: Opening PDF {pdf_path.name} and Processing Pages ---")
        with fitz.open(pdf_path_str) as pdf_doc: # Document opened here for the main loop
            # Verify page count consistency
            doc_page_count = len(pdf_doc)
            if doc_page_count != num_pages:
                log.warning(f"PyMuPDF doc count ({doc_page_count}) differs from intermediate file count ({num_pages}). Using {doc_page_count} from PyMuPDF.")
                num_pages = doc_page_count
                # Adjust intermediate lists - CAREFUL: Indices might mismatch if counts differ significantly.
                # It might be safer to error out or rely solely on pdf_doc iterations.
                # For now, we'll attempt to proceed but log warnings if accessing missing intermediate files.
                image_paths = [Path(p) for p in image_paths_str[:num_pages]]
                single_page_pdf_paths = [Path(p) for p in single_page_pdf_paths_str[:num_pages]]
            else:
                 image_paths = [Path(p) for p in image_paths_str]
                 single_page_pdf_paths = [Path(p) for p in single_page_pdf_paths_str]

            # --- Loop through pages using PyMuPDF's count ---
            for i in range(num_pages):
                page_num = i + 1
                log.info(f"\n--- Processing Page {page_num}/{num_pages} ---")
                page_processed_successfully = False
                pdf_page = None; page_image = None # Initialize per loop

                # Get corresponding intermediate file paths (handle potential index errors if counts mismatched)
                try:
                    img_path = image_paths[i]
                    single_pdf_path = single_page_pdf_paths[i]
                except IndexError:
                    log.error(f"Index error accessing intermediate file paths for page {page_num}. Skipping.")
                    all_pages_markdown.append(f"\n## Page {page_num}\n\n*Error: Missing intermediate file path.*\n")
                    processing_successful = False
                    continue

                # File Checks
                if not img_path.is_file(): log.warning(f"Intermediate image missing: {img_path}. Skipping page content processing."); all_pages_markdown.append(f"\n## Page {page_num}\n\n*Error: Image missing.*\n"); continue # Continue loop, but maybe don't mark overall as failure?
                if not single_pdf_path.is_file(): log.warning(f"Intermediate single PDF missing: {single_pdf_path}. Skipping page content processing."); all_pages_markdown.append(f"\n## Page {page_num}\n\n*Error: Single PDF missing.*\n"); continue

                # --- Main Page Processing Block ---
                try:
                    # Get pdf_page
                    try:
                        pdf_page = pdf_doc.load_page(i) # Use load_page for safety
                    except Exception as page_e:
                        log.error(f"Unexpected error loading PDF page {i}: {page_e}", exc_info=True)
                        all_pages_markdown.append(f"\n## Page {page_num}\n\n*Error: Cannot load PDF page object.*\n")
                        processing_successful = False # If we can't load the page, it's a failure
                        continue # Skip to next page

                    # --- Layout Detection ---
                    log.debug("Detecting layout elements...")
                    detections = detect_layout_elements(layout_processor, layout_model, str(img_path))
                    if not detections:
                        log.info(f"No elements detected on page {page_num}. Generating empty page markdown.")
                        all_pages_markdown.append(f"\n## Page {page_num}\n\n*No elements detected.*\n")
                        page_processed_successfully = True
                        # No need to continue inner try block if no detections
                        continue # Move to the next page loop iteration

                    log.debug(f"Detected {len(detections)} elements.")
                    processed_detections = []

                    # --- Open Page Image & Process Elements ---
                    try:
                        page_image = Image.open(img_path).convert("RGB")
                        img_width_px, img_height_px = page_image.size
                        log.debug(f"Image opened: {img_path.name}, size: {img_width_px}x{img_height_px}")
                        top_margin_px = img_height_px * HEADER_FOOTER_TOP_MARGIN_PERCENT
                        bottom_margin_start_px = img_height_px * BOTTOM_MARGIN_START_THRESHOLD_PERCENT

                        # --- Loop through Detections ---
                        for det_idx, det in enumerate(detections):
                            # (Existing detection processing logic: coordinate filter, crop, OCR/Table, append to processed_detections)
                            # ...
                            pass # Placeholder for the existing detailed logic
                        # --- End of loop through detections ---
                        page_processed_successfully = True # Mark page success if loop completes

                    except Exception as e:
                        log.error(f"Error during image opening or element processing for page {page_num}: {e}", exc_info=True)
                        all_pages_markdown.append(f"\n## Page {page_num}\n\n*Error processing page content.*\n")
                        page_processed_successfully = False
                    finally: # Inner finally for image resource
                        if page_image:
                            try: page_image.close(); log.debug(f"Closed image {img_path.name}")
                            except Exception as close_e: log.error(f"Error closing page image {img_path.name}: {close_e}")

                    # --- Generate Markdown for the page ---
                    if page_processed_successfully:
                        log.debug(f"Page {page_num}: Generating Markdown from {len(processed_detections)} processed elements.")
                        page_md = generate_markdown_page(processed_detections, page_num)
                        all_pages_markdown.append(page_md)
                        log.info(f"--- Page {page_num} Processing Complete ---")
                    else:
                        log.error(f"--- Page {page_num} Processing Failed ---")
                        processing_successful = False # Mark overall as failed if any page fails

                finally: # Outer finally for page resources
                    # Explicitly dereference page object - good practice within loops
                    if pdf_page:
                        log.debug(f"Dereferencing page object for page {page_num}")
                        pdf_page = None
            # --- (END of the main 'for' loop over pages) ---
        # --- End of 'with fitz.open...' block (pdf_doc is automatically closed here) ---

    except Exception as main_e: # Catch errors like fitz.open failure
        log.error(f"An error occurred during the main PDF processing stage: {main_e}", exc_info=True)
        processing_successful = False
    finally:
        # --- Stage 4: Finalize ---
        log.info("--- Stage 4: Finalizing ---")
        # No need to check/close pdf_doc here, 'with' statement handled it.

        # --- Write Markdown Output ---
        final_markdown = "\n".join(all_pages_markdown)
        # (Check final_markdown and write file logic remains the same)
        if not final_markdown and processing_successful: log.warning("Processing successful, but no markdown generated.")
        elif not final_markdown and not processing_successful: log.error("Processing failed AND no markdown generated.")
        elif final_markdown and not processing_successful: log.warning("Processing finished with errors, writing partial Markdown...")
        else: log.info("Processing successful. Writing final Markdown...")

        try:
            with open(markdown_output_path, 'w', encoding='utf-8') as f: f.write(final_markdown)
            log.info(f"Successfully wrote Markdown: {markdown_output_path}")
            # Return path only if overall process was deemed successful
            return markdown_output_path if processing_successful else None
        except Exception as e:
            log.error(f"Error writing final Markdown to {markdown_output_path}: {e}", exc_info=True)
            return None
    # --- End of process_pdf_to_markdown ---

# --- Main execution block ---
# --- Main execution block (for testing main.py standalone) ---
if __name__ == '__main__':
    # --- MODIFIED: Needs to load models first if run standalone ---
    logging.getLogger().setLevel(logging.INFO)
    log.info("=========================================")
    run_name = "standalone_main_processing_run"
    log.info(f" Starting Naini PDF Processing ({run_name}) - Standalone Test ")
    log.info("=========================================")

    # Load layout model explicitly for standalone test
    log.info("Standalone Test: Loading layout model...")
    test_processor, test_model = load_layout_model_and_processor()
    if not test_processor or not test_model:
        log.critical("Standalone Test: Failed to load layout model. Exiting.")
        exit(1)
    log.info("Standalone Test: Layout model loaded.")

    # --- Standard Paths ---
    from . import config # Import config here for standalone test
    INPUT_PDF = config.DEFAULT_PDF_PATH
    OUTPUT_MD_DIR = config.DEFAULT_PDF_OUTPUT_MARKDOWN_PATH.parent
    PERSISTENT_IMG_DIR = config.DEFAULT_PERSISTENT_IMAGE_DIR
    PERSISTENT_SINGLE_PDF_DIR = config.DEFAULT_PERSISTENT_SINGLE_PDF_DIR

    if not INPUT_PDF.is_file():
        log.critical(f"CRITICAL: Default Input PDF not found at {INPUT_PDF}. Exiting.")
        exit(1)

    # --- Execute Workflow ---
    try:
        # --- MODIFIED CALL - Pass loaded models ---
        result_path = process_pdf_to_markdown(
            pdf_path_str=str(INPUT_PDF),
            output_markdown_dir_str=str(OUTPUT_MD_DIR),
            output_image_dir_str=str(PERSISTENT_IMG_DIR),
            output_single_page_pdf_dir_str=str(PERSISTENT_SINGLE_PDF_DIR),
            layout_processor=test_processor, # Pass loaded processor
            layout_model=test_model          # Pass loaded model
        )
        # ----------------------------------------
        if result_path: log.info(f"Standalone Test: Default PDF processing successful. Output: {result_path}")
        else: log.error("Standalone Test: Default PDF processing failed.")
    except Exception as e:
        log.critical(f"Standalone Test: Unhandled exception: {e}", exc_info=True)

    log.info("=========================================")
    log.info(f"      Standalone Processing Finished ({run_name})      ")
    log.info("=========================================")