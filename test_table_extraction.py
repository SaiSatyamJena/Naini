# /workspace/test_table_extraction.py
import logging
import argparse
from pathlib import Path
import sys
from PIL import Image
import fitz  # PyMuPDF
import pandas as pd

# Configure logging first
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - [%(module)s.%(funcName)s] - %(message)s')
log = logging.getLogger(__name__)

# --- Dynamically add naini_app to Python path ---
# This allows running the script from /workspace
script_dir = Path(__file__).parent.resolve()
app_dir = script_dir / "Naini" / "naini_app" # Assumes script is in /workspace and app is in /workspace/Naini/naini_app
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(script_dir / "Naini")) # Add Naini parent dir to path

try:
    # --- Import Naini modules AFTER adjusting path ---
    from naini_app.pdf_processor import convert_pdf_to_images, split_pdf_to_pages
    from naini_app.layout_detector import load_layout_model_and_processor, detect_layout_elements
    from naini_app.table_extractor import extract_table_camelot, extract_table_img2table
    from naini_app.utils import map_image_box_to_pdf_coords
except ImportError as e:
    log.critical(f"Failed to import Naini modules. Ensure the script is in the correct location relative to the 'Naini' directory and all dependencies are installed. Error: {e}", exc_info=True)
    sys.exit(1)

# --- Constants ---
IMAGE_DPI = 300 # Should match the DPI used in main pipeline

def test_page_extraction(pdf_path: Path, page_num: int, persistent_image_base_dir: Path, persistent_single_pdf_base_dir: Path):
    """
    Tests table extraction for a specific page using both Camelot and img2table.
    """
    log.info(f"--- Starting Test for Page {page_num} of {pdf_path.name} ---")

    # --- 1. Verify Input PDF ---
    if not pdf_path.is_file():
        log.error(f"Input PDF not found: {pdf_path}")
        return

    pdf_stem = pdf_path.stem

    # --- 2. Prepare Persistent Data (Get paths, generate if needed) ---
    log.info("Preparing persistent image and single-page PDF paths...")
    try:
        # Ensure images are generated (call generates if needed)
        all_image_paths = convert_pdf_to_images(str(pdf_path), str(persistent_image_base_dir), dpi=IMAGE_DPI)
        if not all_image_paths:
            log.error("Failed to get page images.")
            return

        # Ensure single-page PDFs are generated
        all_single_pdf_paths = split_pdf_to_pages(str(pdf_path), str(persistent_single_pdf_base_dir))
        if not all_single_pdf_paths:
            log.error("Failed to get single-page PDFs.")
            return

        # Get the specific paths for the target page (adjusting for 0-based index)
        if page_num < 1 or page_num > len(all_image_paths):
            log.error(f"Page number {page_num} is out of range (1-{len(all_image_paths)}).")
            return
        img_path = Path(all_image_paths[page_num - 1])
        single_page_pdf_path = Path(all_single_pdf_paths[page_num - 1])

        if not img_path.is_file():
            log.error(f"Image file for page {page_num} not found at expected path: {img_path}")
            return
        if not single_page_pdf_path.is_file():
            log.error(f"Single-page PDF for page {page_num} not found at expected path: {single_page_pdf_path}")
            return

        log.info(f"Using Image: {img_path.name}")
        log.info(f"Using Single PDF: {single_page_pdf_path.name}")

    except Exception as e:
        log.error(f"Error preparing persistent data: {e}", exc_info=True)
        return

    # --- 3. Load Models ---
    log.info("Loading Layout Model...")
    processor, model = load_layout_model_and_processor()
    if not processor or not model:
        log.error("Failed to load DETR model/processor.")
        return
    # Note: img2table OCR instance is initialized globally within table_extractor

    # --- 4. Get fitz.Page Object ---
    pdf_doc = None
    pdf_page = None
    try:
        log.info("Opening source PDF with PyMuPDF...")
        pdf_doc = fitz.open(str(pdf_path))
        if page_num > len(pdf_doc):
             log.error(f"PyMuPDF page count ({len(pdf_doc)}) is less than requested page {page_num}.")
             pdf_doc.close()
             return
        pdf_page = pdf_doc[page_num - 1] # 0-based index
        log.info(f"Successfully accessed fitz.Page object for page {page_num}.")
    except Exception as e:
        log.error(f"Failed to open source PDF or get page {page_num} with PyMuPDF: {e}", exc_info=True)
        if pdf_doc: pdf_doc.close()
        return

    # --- 5. Run Layout Detection ---
    log.info(f"Running layout detection on {img_path.name}...")
    detections = detect_layout_elements(processor, model, str(img_path))
    if not detections:
        log.warning(f"No elements detected for page {page_num}.")
        if pdf_doc: pdf_doc.close()
        return

    table_detections = [det for det in detections if det['label'] == 'Table']
    log.info(f"Found {len(table_detections)} 'Table' regions on page {page_num}.")

    if not table_detections:
        log.info("No tables detected on this page to test.")
        if pdf_doc: pdf_doc.close()
        return

    # --- 6. Open Page Image ---
    page_image = None
    try:
        page_image = Image.open(img_path).convert("RGB")
        img_width_px, img_height_px = page_image.size
        log.info(f"Opened page image: {img_width_px}x{img_height_px} pixels.")
    except Exception as e:
        log.error(f"Failed to open page image {img_path}: {e}", exc_info=True)
        if pdf_doc: pdf_doc.close()
        return

    # --- 7. Process Each Detected Table ---
    for i, det in enumerate(table_detections):
        log.info(f"\n===== Processing Detected Table {i+1} =====")
        box_px = det['box'] # Pixel coordinates [x1, y1, x2, y2]
        log.info(f"  DETR Box (pixels): {box_px}")

        # --- 7a. Crop Image ---
        try:
            x1, y1, x2, y2 = map(int, [
                max(0, box_px[0]), max(0, box_px[1]),
                min(img_width_px, box_px[2]), min(img_height_px, box_px[3])
            ])
            if x1 >= x2 or y1 >= y2:
                log.warning(f"  Skipping invalid box coordinates.")
                continue
            element_crop = page_image.crop((x1, y1, x2, y2))
            log.info(f"  Cropped image area: [{x1},{y1},{x2},{y2}]")
        except Exception as e:
            log.error(f"  Error cropping image: {e}", exc_info=True)
            continue # Skip to next detected table

        # --- 7b. Map Coordinates ---
        log.info("  Mapping image coordinates to PDF coordinates...")
        pdf_coords_str = map_image_box_to_pdf_coords(
            image_box_px=box_px,
            img_width_px=img_width_px,
            img_height_px=img_height_px,
            pdf_page=pdf_page,
            img_dpi=IMAGE_DPI
        )
        if pdf_coords_str:
            log.info(f"  Mapped PDF Coords: '{pdf_coords_str}'")
            #  # --- START INSERTED BLOCK ---
            # # Manually override/expand coordinates for Page 107, Table 1 for debugging
            # if page_num == 107 and i == 0: # Assuming page 107 and it's the first table detected
            #     original_coords = pdf_coords_str
            #     try:
            #         x1, y1, x2, y2_old = map(float, original_coords.split(','))
            #         # Increase y2 slightly (move bottom edge down) - Adjust value as needed
            #         y2_new = y2_old + 10.0 # Add 10 points (approx 3-4 rows?)
            #         # Ensure it doesn't exceed page height (though unlikely here)
            #         pdf_height = pdf_page.rect.height
            #         y2_new = min(y2_new, pdf_height)
                    
            #         pdf_coords_str = f"{x1:.2f},{y1:.2f},{x2:.2f},{y2_new:.2f}"
            #         log.warning(f"  OVERRIDING Coords for Page 107 Debug: '{pdf_coords_str}' (Original y2: {y2_old:.2f})")
            #     except Exception as e:
            #         log.error(f"  Failed to parse/adjust original coords '{original_coords}': {e}")
            #         pdf_coords_str = original_coords # Revert on error
            # # --- END INSERTED BLOCK ---
            
        else:
            log.warning("  Coordinate mapping failed. Cannot test Camelot.")

        # --- 7c. Test Camelot (Lattice then Stream) ---
        log.info("  --- Testing Camelot ---")
        camelot_result_md = None
        if pdf_coords_str:
            try:
                # This function tries lattice then stream
                camelot_result_md = extract_table_camelot(str(single_page_pdf_path), pdf_coords_str)
                if camelot_result_md:
                    log.info("  Camelot extraction SUCCEEDED.")
                    # Print limited output to avoid flooding terminal
                    print("\n  CAMELOT RESULT (Markdown Head):")
                    print('\n'.join(camelot_result_md.splitlines()[:10])) # Print first 10 lines
                    if len(camelot_result_md.splitlines()) > 10: print("  ...")
                else:
                    log.info("  Camelot extraction FAILED (returned None).")
            except Exception as e:
                log.error(f"  Camelot extraction raised an EXCEPTION: {e}", exc_info=True)
        else:
             log.info("  Skipping Camelot test due to coordinate mapping failure.")
        print("-" * 30)

        # --- 7d. Test img2table (Standalone) ---
        log.info("  --- Testing img2table ---")
        img2table_result_md = None
        try:
            img2table_result_md = extract_table_img2table(element_crop)
            if img2table_result_md:
                log.info("  img2table extraction SUCCEEDED.")
                print("\n  IMG2TABLE RESULT (Markdown Head):")
                print('\n'.join(img2table_result_md.splitlines()[:10])) # Print first 10 lines
                if len(img2table_result_md.splitlines()) > 10: print("  ...")
            else:
                log.info("  img2table extraction FAILED (returned None).")
        except Exception as e:
            log.error(f"  img2table extraction raised an EXCEPTION: {e}", exc_info=True)
        print("-" * 30)

        # --- (Optional) Save crop for manual inspection ---
        # crop_save_path = Path(f"/workspace/data/output_markdown/debug_page{page_num}_table{i+1}_crop.png")
        # try:
        #     element_crop.save(crop_save_path)
        #     log.info(f"  Saved image crop to {crop_save_path}")
        # except Exception as e:
        #     log.error(f"  Failed to save image crop: {e}")

        # Close the crop image
        element_crop.close()

    # --- 8. Cleanup ---
    log.info("Cleaning up resources...")
    if page_image:
        page_image.close()
    if pdf_doc:
        pdf_doc.close()
    log.info(f"--- Finished Test for Page {page_num} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test table extraction for a specific page of a PDF.")
    parser.add_argument("pdf_file", help="Name of the PDF file in data/persistent_pdfs/")
    parser.add_argument("page_number", type=int, help="Page number to test (1-based index).")
    parser.add_argument("--img_dir", default="data/persistent_images", help="Base directory for persistent images.")
    parser.add_argument("--pdf_dir", default="data/persistent_single_pdfs", help="Base directory for persistent single-page PDFs.")

    args = parser.parse_args()

    # Construct full paths based on /workspace assumption
    workspace_root = Path("/workspace")
    input_pdf_path = workspace_root / "data"/"persistent_pdfs"/ args.pdf_file
    persistent_img_path = workspace_root / args.img_dir
    persistent_pdf_path = workspace_root / args.pdf_dir

    # Run the test
    test_page_extraction(
        pdf_path=input_pdf_path,
        page_num=args.page_number,
        persistent_image_base_dir=persistent_img_path,
        persistent_single_pdf_base_dir=persistent_pdf_path
    )