# /workspace/naini_app/main.py
# VERSION WITH DETAILED ELEMENT PROCESSING LOGGING

import logging
from pathlib import Path
from typing import Optional, Any, List # Ensure List is imported
from PIL import Image
import fitz # Import PyMuPDF

# --- Import custom modules ---
# Note: Assuming these exist and are correct relative to main.py
from .pdf_processor import convert_pdf_to_images, split_pdf_to_pages
from .layout_detector import detect_layout_elements # Keep detect_layout_elements import
from .ocr_extractor import extract_text_from_image
from .table_extractor import extract_table_hybrid,TABLE_FAILURE_PLACEHOLDER
from .markdown_generator import generate_markdown_page, _filter_non_english, _general_cleanup # Keep generate_markdown_page import
from .utils import map_image_box_to_pdf_coords

# --- Basic Logging Setup ---
# Set level to INFO or DEBUG in main app or here for testing
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s')
log = logging.getLogger(__name__) # Use module-level logger


# --- Constants ---
IMAGE_DPI = 300

def process_pdf_to_markdown(
    pdf_path_str: str,
    output_markdown_dir_str: str,
    output_image_dir_str: str,
    output_single_page_pdf_dir_str: str,
    layout_processor: Any, # Pre-loaded layout processor
    layout_model: Any      # Pre-loaded layout model
) -> Optional[Path]:
    """
    Main workflow: Processes PDF to Markdown using pre-loaded models.
    Generates intermediates first, then processes pages with detailed logging.

    Args:
        pdf_path_str: Path to the input PDF file.
        output_markdown_dir_str: Directory where the final Markdown file will be saved.
        output_image_dir_str: Directory where intermediate page images will be saved.
        output_single_page_pdf_dir_str: Directory where intermediate single-page PDFs will be saved.
        layout_processor: The pre-loaded layout detection processor.
        layout_model: The pre-loaded layout detection model.

    Returns:
        Path to the generated Markdown file if successful, otherwise None.
    """
    pdf_path = Path(pdf_path_str)
    output_markdown_dir = Path(output_markdown_dir_str)
    output_image_dir = Path(output_image_dir_str)
    output_single_page_pdf_dir = Path(output_single_page_pdf_dir_str)
    markdown_output_path = output_markdown_dir / f"{pdf_path.stem}_output.md"

    log.info(f"Starting Markdown generation for '{pdf_path.name}'. Output: {markdown_output_path}")

    # --- Header/Footer Margins ---
    HEADER_FOOTER_TOP_MARGIN_PERCENT = 0.01
    HEADER_FOOTER_BOTTOM_MARGIN_PERCENT = 0.01
    BOTTOM_MARGIN_START_THRESHOLD_PERCENT = 1.0 - HEADER_FOOTER_BOTTOM_MARGIN_PERCENT

    # --- Input Checks ---
    if not pdf_path.is_file():
        log.error(f"Input PDF not found: {pdf_path}")
        return None
    if not layout_processor or not layout_model:
        log.error("Layout processor or model not provided to process_pdf_to_markdown.")
        return None

    # --- Create Output Dirs ---
    try:
        output_markdown_dir.mkdir(parents=True, exist_ok=True)
        output_image_dir.mkdir(parents=True, exist_ok=True)
        output_single_page_pdf_dir.mkdir(parents=True, exist_ok=True)
        log.debug("Output directories ensured.")
    except Exception as e:
        log.error(f"Error creating output directories: {e}", exc_info=True)
        return None

    # === Stage 0: Generate ALL Intermediate Files FIRST ===
    log.info("--- Stage 0: Generating Intermediate Files ---")
    image_paths_str: Optional[List[str]] = None
    single_page_pdf_paths_str: Optional[List[str]] = None
    num_pages = 0
    try:
        image_paths_str = convert_pdf_to_images(str(pdf_path), str(output_image_dir), dpi=IMAGE_DPI)
        if not image_paths_str:
            log.error("Failed to generate page images.")
            return None

        single_page_pdf_paths_str = split_pdf_to_pages(str(pdf_path), str(output_single_page_pdf_dir))
        if not single_page_pdf_paths_str:
            log.error("Failed to generate single-page PDFs.")
            return None

        if len(image_paths_str) != len(single_page_pdf_paths_str):
            log.error(f"Intermediate file count mismatch! Images: {len(image_paths_str)}, Single PDFs: {len(single_page_pdf_paths_str)}")
            return None

        num_pages = len(image_paths_str)
        if num_pages == 0:
            log.warning(f"Zero pages found during intermediate file generation for {pdf_path.name}")
            return None
        log.info(f"Stage 0: Successfully generated {num_pages} intermediate image(s) and single-page PDF(s).")

    except Exception as e_intermediate:
        log.error(f"Error during intermediate file generation: {e_intermediate}", exc_info=True)
        return None
    # === End of Intermediate File Generation ===

    all_pages_markdown = []
    processing_successful = True

    try:
        log.info(f"--- Stage 2&3: Opening PDF {pdf_path.name} and Processing {num_pages} Pages ---")
        with fitz.open(pdf_path_str) as pdf_doc:
            doc_page_count = len(pdf_doc)
            if doc_page_count != num_pages:
                log.warning(f"PyMuPDF doc count ({doc_page_count}) differs from intermediate file count ({num_pages}). Using PyMuPDF count: {doc_page_count}.")
                num_pages = doc_page_count

            image_paths = [Path(p) for p in image_paths_str[:num_pages]]
            single_page_pdf_paths = [Path(p) for p in single_page_pdf_paths_str[:num_pages]]

            for i in range(num_pages):
                page_num = i + 1
                log.info(f"--- Processing Page {page_num}/{num_pages} ---")
                page_processed_successfully = False
                pdf_page = None
                page_image = None

                img_path: Optional[Path] = image_paths[i] if i < len(image_paths) else None
                single_pdf_path: Optional[Path] = single_page_pdf_paths[i] if i < len(single_page_pdf_paths) else None

                if not img_path or not img_path.is_file():
                    log.warning(f"Intermediate image missing or path error for page {page_num} (Path: {img_path}). Skipping content processing.")
                    all_pages_markdown.append(f"\n## Page {page_num}\n\n*Error: Associated image file missing.*\n")
                    continue
                # Only warn if single_pdf is missing, might impact Camelot but img2table can proceed
                if not single_pdf_path or not single_pdf_path.is_file():
                    log.warning(f"Intermediate single PDF missing or path error for page {page_num} (Path: {single_pdf_path}). Might affect Camelot table extraction.")


                page_content_processed = False
                processed_detections = [] # Reset for THIS PAGE

                try:
                    try:
                        pdf_page = pdf_doc.load_page(i)
                    except Exception as page_e:
                        log.error(f"FATAL: Cannot load PDF page object {i}: {page_e}", exc_info=True)
                        all_pages_markdown.append(f"\n## Page {page_num}\n\n*FATAL Error: Cannot load PDF page object.*\n")
                        processing_successful = False
                        break

                    log.info(f"Page {page_num}: Running layout detection on {img_path.name}...")
                    detections = detect_layout_elements(layout_processor, layout_model, str(img_path))
                    total_detected_count = len(detections)
                    log.info(f"Page {page_num}: Detected {total_detected_count} raw elements.")
                    # <<< ADDED LOG >>> - Log raw detections (limit output if too verbose?)
                    # Let's log up to first 5 for diagnosis, then a summary
                    log.debug(f"Page {page_num}: Raw Detections (showing up to 5):")
                    for raw_idx, raw_det in enumerate(detections[:5]):
                        log.debug(f"  Raw Elem {raw_idx}: Label={raw_det.get('label')}, Score={raw_det.get('score', -1):.2f}, Box={raw_det.get('box')}")
                    if total_detected_count > 5:
                        log.debug(f"  ... ({total_detected_count - 5} more raw detections)")

                    if not detections:
                        log.info(f"Page {page_num}: No elements detected by layout model.")
                        all_pages_markdown.append(f"\n## Page {page_num}\n\n*No elements detected.*\n")
                        page_processed_successfully = True
                        continue

                    try:
                        page_image = Image.open(img_path).convert("RGB")
                        img_width_px, img_height_px = page_image.size
                        log.debug(f"Page {page_num}: Image opened: {img_path.name}, size: {img_width_px}x{img_height_px}")
                    except Exception as img_e:
                         log.error(f"FATAL: Cannot open page image {img_path}: {img_e}", exc_info=True)
                         all_pages_markdown.append(f"\n## Page {page_num}\n\n*FATAL Error: Cannot open page image.*\n")
                         processing_successful = False
                         break

                    top_margin_px = img_height_px * HEADER_FOOTER_TOP_MARGIN_PERCENT
                    bottom_margin_start_px = img_height_px * BOTTOM_MARGIN_START_THRESHOLD_PERCENT
                    log.debug(f"Page {page_num}: Image Height={img_height_px}px. Filter Margins: Top<{top_margin_px:.1f}, Bottom>{bottom_margin_start_px:.1f}")

                    text_labels = ['Text', 'Title', 'List-item', 'Caption', 'Section-header', 'Footnote', 'Formula']
                    table_labels = ['Table']
                    figure_labels = ['Picture']
                    skip_labels = ['Page-header', 'Page-footer']

                    elements_passed_filter_count = 0 # Track elements before content extraction attempts

                    for det_idx, det in enumerate(detections):
                        # Make sure 'box', 'label', 'score' exist before trying to access them
                        if not all(k in det for k in ['box', 'label', 'score']):
                             log.warning(f"Page {page_num}, Elem {det_idx+1}/{total_detected_count}: SKIPPED missing key(s) (box/label/score). Data: {det}")
                             continue

                        box = det['box']; label = det['label']; score = det['score']
                        log_prefix = f"Page {page_num}, Elem {det_idx+1}/{total_detected_count} ('{label}', score={score:.2f}, box={box})" # Include box in prefix
                        element_crop = None

                        # --- Coordinate Filtering ---
                        elem_y1, elem_y2 = box[1], box[3]
                        if elem_y2 <= top_margin_px or elem_y1 >= bottom_margin_start_px:
                            log.info(f"{log_prefix}: SKIPPED by coordinate filter. Y-coords: [{elem_y1:.1f}, {elem_y2:.1f}], Margins: [Top < {top_margin_px:.1f}, Bottom > {bottom_margin_start_px:.1f}]")
                            continue

                        # --- Explicit Label Skip ---
                        if label in skip_labels:
                            log.info(f"{log_prefix}: SKIPPED by label filter (Label: '{label}')")
                            continue

                        # <<< ADDED LOG >>> - Element passed initial filters
                        log.debug(f"{log_prefix}: Passed initial filters.")
                        elements_passed_filter_count += 1 # Increment here after passing initial filters

                        # --- Inner Try Block for Content Extraction ---
                        # Initialize flags/variables for this element's processing
                        content_extracted_successfully = False # Did the extraction function succeed?
                        content_is_valid = False # Is the extracted content non-trivial?
                        processed_data = None # Store extracted text/table result

                        try:
                            x1 = max(0, int(box[0])); y1 = max(0, int(elem_y1))
                            x2 = min(img_width_px, int(box[2])); y2 = min(img_height_px, int(elem_y2))
                            if x1 >= x2 or y1 >= y2:
                                log.warning(f"{log_prefix}: SKIPPED invalid/zero-area box [{x1},{y1},{x2},{y2}] after clipping.")
                                continue

                            # <<< ADDED LOG >>> - Cropping element
                            log.debug(f"{log_prefix}: Cropping image element at [{x1},{y1},{x2},{y2}]")
                            element_crop = page_image.crop((x1, y1, x2, y2))
                            current_det = det.copy() # Work on a copy for modification

                            # --- Table Processing ---
                            if label in table_labels:
                                log.info(f"{log_prefix}: --> Attempting TABLE extraction.")
                                table_content_tuple = extract_table_hybrid(
                                    image_crop=element_crop, image_box_px=[x1, y1, x2, y2],
                                    img_width_px=img_width_px, img_height_px=img_height_px,
                                    pdf_page=pdf_page, single_page_pdf_path=str(single_pdf_path) if single_pdf_path else None,
                                    img_dpi=IMAGE_DPI
                                )
                                # <<< ADDED LOG >>> - Log the result from table extractor
                                log.info(f"{log_prefix}: <-- TABLE extraction result: Method='{table_content_tuple[1]}', Content (first 100 chars)='{str(table_content_tuple[0])[:100] if table_content_tuple[0] else 'None'}'")
                                current_det['table_data'] = table_content_tuple # Store the tuple
                                # Mark success IF the method is not 'failure'
                                if table_content_tuple and table_content_tuple[1] != 'failure':
                                     content_extracted_successfully = True
                                     # Check if content is non-empty/not placeholder (can be None from extractor)
                                     if table_content_tuple[0] is not None and table_content_tuple[0] != TABLE_FAILURE_PLACEHOLDER:
                                          content_is_valid = True
                                processed_data = table_content_tuple # Store the tuple as data for later check

                            # --- Text Processing ---
                            elif label in text_labels or label not in figure_labels + table_labels + skip_labels: # Handle known text + fallback unknown
                                if label not in text_labels:
                                    log.warning(f"{log_prefix}: Processing as FALLBACK TEXT (Unknown Label '{label}')")
                                else:
                                     log.info(f"{log_prefix}: --> Attempting TEXT extraction.")

                                # <<< MODIFIED FOR HINDI >>> Add 'hin' language
                                # Change lang='eng' to lang='eng+hin'
                                extracted_text = extract_text_from_image(element_crop, lang='eng+hin') # TODO: Make language configurable
                                log.debug(f"{log_prefix}: Raw OCR Text (first 100 chars): '{extracted_text[:100]}'")

                                # Filter and clean (as before, filters now defined in markdown_generator.py)
                                filtered_text = _filter_non_english(extracted_text)
                                cleaned_text = _general_cleanup(filtered_text)
                                log.debug(f"{log_prefix}: Cleaned OCR Text (first 100 chars): '{cleaned_text[:100]}'")

                                current_det['text'] = cleaned_text # Store cleaned text
                                content_extracted_successfully = True # OCR itself ran
                                if cleaned_text: content_is_valid = True # Considered valid if non-empty after cleaning
                                processed_data = cleaned_text # Store string as data

                            # --- Figure Processing ---
                            elif label in figure_labels:
                                log.info(f"{log_prefix}: Marking as FIGURE (no extraction needed).")
                                current_det['label'] = 'Picture' # Standardize
                                current_det['text'] = None # Ensure no text field interferes
                                content_extracted_successfully = True # Considered 'successful' as it's marked
                                content_is_valid = True # Figures are valid content
                                processed_data = "[FIGURE]" # Store marker for logging

                            # <<< ADDED CHECK & LOG >>> - Decision to append element
                            if content_extracted_successfully and content_is_valid:
                                 log.info(f"{log_prefix}: <<< PASSED EXTRACTION & VALIDATION >>> Appending to processed_detections. Label='{current_det['label']}', Data Summary='{str(processed_data)[:100] if processed_data else 'N/A'}'")
                                 processed_detections.append(current_det)
                            else:
                                 log.warning(f"{log_prefix}: <<< FAILED / SKIPPED >>> Not appending. Success={content_extracted_successfully}, Valid={content_is_valid}, Label='{current_det.get('label')}', Data='{str(processed_data)[:100] if processed_data else 'N/A'}'")

                        except Exception as content_e:
                            log.error(f"{log_prefix}: Error during content extraction/processing for this element: {content_e}", exc_info=True)
                            # Do not append if an error occurred during extraction
                        finally:
                            if element_crop:
                                try: element_crop.close()
                                except Exception as close_e: log.warning(f"{log_prefix}: Error closing crop image: {close_e}")

                    # --- End of Detections Loop ---
                    log.info(f"Page {page_num}: Finished filtering/processing loop.")
                    log.info(f"Page {page_num}: Initial Filter Pass Count = {elements_passed_filter_count} / Total Detected = {total_detected_count}")
                    log.info(f"Page {page_num}: Final Count Appended to 'processed_detections' = {len(processed_detections)}")
                    page_content_processed = True

                except Exception as outer_content_e:
                    log.error(f"Error during page content processing block for page {page_num}: {outer_content_e}", exc_info=True)
                    page_content_processed = False
                    processing_successful = False
                    all_pages_markdown.append(f"\n## Page {page_num}\n\n*FATAL Error processing page content block.*\n")
                    break

                finally:
                    if page_image:
                        try: page_image.close(); log.debug(f"Page {page_num}: Closed page image.")
                        except Exception as close_e: log.warning(f"Page {page_num}: Error closing page image: {close_e}")
                    if pdf_page:
                        try: log.debug(f"Page {page_num}: Dereferencing page object."); pdf_page = None
                        except Exception: pass

                if page_content_processed:
                    # <<< RE-CONFIRM LOG >>> - Reconfirm size just before calling generate_markdown_page
                    final_elements_count = len(processed_detections)
                    log.info(f"Page {page_num}: ---> Calling generate_markdown_page with {final_elements_count} elements.")
                    if final_elements_count == 0 and elements_passed_filter_count > 0:
                         log.warning(f"Page {page_num}: Passed initial filters ({elements_passed_filter_count}), but final list for MD gen is empty. Check extraction/validation failures.")

                    try:
                        page_md = generate_markdown_page(processed_detections, page_num)
                        all_pages_markdown.append(page_md)
                        page_processed_successfully = True
                        log.info(f"--- Page {page_num} Processing Successfully Completed ---")
                    except Exception as md_gen_e:
                         log.error(f"Page {page_num}: Error during Markdown generation step: {md_gen_e}", exc_info=True)
                         all_pages_markdown.append(f"\n## Page {page_num}\n\n*Error generating page Markdown.*\n")
                         # Keep processing successful unless fatal? Or mark fail? Let's mark failure for MD gen error.
                         processing_successful = False

                if not page_processed_successfully:
                     log.error(f"--- Page {page_num} FAILED Processing ---")
                     # Mark overall process as failed if any page fails its primary processing.
                     processing_successful = False # Redundant? Maybe not if content_processed was False. Safety check.

            # --- End of Page Loop ---

        log.info("Finished processing all pages.")

    except Exception as main_e:
        log.error(f"FATAL error during main PDF processing stage: {main_e}", exc_info=True)
        processing_successful = False

    log.info("--- Stage 4: Finalizing ---")

    final_markdown = "\n".join(all_pages_markdown).strip()
    # Final logging based on success and content presence remains the same
    if not final_markdown and processing_successful: log.warning("Processing marked successful, but final markdown string is empty.")
    elif not final_markdown and not processing_successful: log.error("Processing failed AND final markdown string is empty.")
    elif final_markdown and not processing_successful: log.warning("Processing finished with errors, writing partial Markdown...")
    else: log.info("Processing seems complete. Writing final Markdown...")


    try:
        # Ensure we have *some* content or placeholder if successful, before writing.
        # Avoid writing empty file if truly nothing was ever generated.
        # Add a master placeholder if completely empty but process technically didn't fatal error?
        # Example: If all pages were skipped or resulted in empty MD.
        if not final_markdown and processing_successful:
             log.warning("Writing placeholder for successful process with no content.")
             final_markdown = "# Processing Completed\n\nNo content was successfully extracted or formatted into Markdown."

        # Only attempt write if processing_successful is True (or we have some content despite errors)
        # We might want partial output even on page errors. Let's write if there's *anything*.
        if final_markdown:
             with open(markdown_output_path, 'w', encoding='utf-8') as f:
                 f.write(final_markdown)
             log.info(f"Successfully wrote final Markdown ({len(final_markdown)} chars): {markdown_output_path}")
             # Decide return value based on overall success flag now
             return markdown_output_path if processing_successful else None
        else: # No markdown and processing likely failed
             log.error(f"Processing failed and resulted in no Markdown content. No file written to {markdown_output_path}")
             return None

    except Exception as write_e:
        log.error(f"Error writing final Markdown to {markdown_output_path}: {write_e}", exc_info=True)
        return None


# --- Standalone test block `if __name__ == '__main__':` ---
# (This block remains unchanged from the previous version)
if __name__ == '__main__':
    # --- MODIFIED: Needs to load models first if run standalone ---
    logging.getLogger().setLevel(logging.INFO) # Or DEBUG for very verbose output
    log.info("=========================================")
    run_name = "standalone_main_processing_run_with_logs"
    log.info(f" Starting Naini PDF Processing ({run_name}) - Standalone Test ")
    log.info("=========================================")

    # Load layout model explicitly for standalone test
    # Need to import the loader function if running standalone
    from .layout_detector import load_layout_model_and_processor
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
        else: log.error("Standalone Test: Default PDF processing failed or produced no output.")
    except Exception as e:
        log.critical(f"Standalone Test: Unhandled exception: {e}", exc_info=True)

    log.info("=========================================")
    log.info(f"      Standalone Processing Finished ({run_name})      ")
    log.info("=========================================")