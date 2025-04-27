# naini_app/pdf_processor.py

from pdf2image import convert_from_path
from pathlib import Path
import logging
import glob
import fitz # PyMuPDF
import os # Needed for removing temp files if split fails

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# convert_pdf_to_images function remains the same...
# Placeholder: Add error handling, configuration for DPI, etc.
def convert_pdf_to_images(pdf_path_str: str, persistent_output_folder_str: str, format='png', dpi=300):
    """
    Converts each page of a PDF file into an image, saving to a persistent folder.
    If images already exist in the folder, it skips conversion and returns existing paths.

    Args:
        pdf_path_str: Path to the input PDF file.
        persistent_output_folder_str: Base directory for persistent images.
                                        A subfolder named after the PDF stem will be used/created.
        format: Image format (png, jpeg, etc.).
        dpi: Dots per inch resolution for the output images.

    Returns:
        A list of paths (strings) to the generated or existing images, sorted numerically.
        Returns an empty list on critical error.
    """
    pdf_path = Path(pdf_path_str)
    if not pdf_path.is_file():
        logging.error(f"Input PDF not found: {pdf_path_str}")
        return []

    pdf_stem = pdf_path.stem
    specific_output_folder = Path(persistent_output_folder_str) / pdf_stem
    logging.info(f"Target image folder: {specific_output_folder}")

    # --- Check if images already exist ---
    specific_output_folder.mkdir(parents=True, exist_ok=True)
    # Use a pattern that matches the pdf2image output format more reliably
    existing_images = sorted(
        glob.glob(str(specific_output_folder / f"page_*-*.{format}")), # Adjusted glob pattern
        key=lambda x: int(Path(x).stem.split('-')[-1]) # Sort by the number after the last hyphen
    )


    # Check if the number of existing images matches the PDF page count
    expected_page_count = 0
    try:
        with fitz.open(pdf_path_str) as doc:
            expected_page_count = len(doc)
    except Exception as e:
        logging.error(f"Could not open PDF {pdf_path_str} to get page count: {e}")
        # Proceed with image check, but might be incomplete

    if existing_images and expected_page_count > 0 and len(existing_images) == expected_page_count:
        logging.info(f"Found {len(existing_images)} existing images (matching page count) in {specific_output_folder}. Skipping conversion.")
        return [str(p) for p in existing_images]
    elif existing_images:
         logging.warning(f"Found {len(existing_images)} existing images, but expected {expected_page_count} pages. Re-generating images.")
         # Optionally remove existing images before regenerating
         for img_path in existing_images:
             try:
                 os.remove(img_path)
             except OSError as e:
                 logging.error(f"Error removing existing image {img_path}: {e}")
    # --- End check ---

    logging.info(f"Starting PDF to image conversion for: {pdf_path_str} ({expected_page_count} pages)")
    image_paths = []
    try:
        # Use a more structured output file naming convention consistent with sorting
        output_file_prefix = f"page_{pdf_stem}" # Example: page_reportFS2021

        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            output_folder=str(specific_output_folder),
            fmt=format,
            output_file=output_file_prefix, # Use prefix
            thread_count=4 # Optional: Use multiple threads if beneficial
        )

        # pdf2image generates names like prefix-0001-1.png, prefix-0001-2.png etc.
        # We need to rename them to a simpler format like page_001.png, page_002.png
        final_image_paths = []
        for i, img in enumerate(images):
            page_num = i + 1
            # Define the desired new filename format (e.g., page_001.png)
            new_filename = specific_output_folder / f"page_{page_num:03d}.{format}"
            # Rename the generated file
            original_path = Path(img.filename)
            if original_path.exists():
                original_path.rename(new_filename)
                final_image_paths.append(str(new_filename))
            else:
                logging.warning(f"Expected image file {original_path} not found after conversion.")

        # Sort the final paths numerically
        final_image_paths.sort(key=lambda x: int(Path(x).stem.split('_')[-1]))

        logging.info(f"Successfully converted and renamed {len(final_image_paths)} pages to images in {specific_output_folder}")
        return final_image_paths

    except Exception as e:
        logging.error(f"Error converting PDF {pdf_path_str} to images: {e}", exc_info=True)
        return []

# --- NEW FUNCTION ---
def split_pdf_to_pages(pdf_path_str: str, persistent_output_folder_str: str):
    """
    Splits a PDF into single-page PDF files, saving to a persistent folder.
    If single-page PDFs already exist and match the page count, skips splitting.

    Args:
        pdf_path_str: Path to the input PDF file.
        persistent_output_folder_str: Base directory for persistent single-page PDFs.
                                        A subfolder named after the PDF stem will be used/created.

    Returns:
        A list of paths (strings) to the generated or existing single-page PDFs, sorted numerically.
        Returns an empty list on critical error.
    """
    pdf_path = Path(pdf_path_str)
    if not pdf_path.is_file():
        logging.error(f"Input PDF not found: {pdf_path_str}")
        return []

    pdf_stem = pdf_path.stem
    specific_output_folder = Path(persistent_output_folder_str) / pdf_stem
    logging.info(f"Target single-page PDF folder: {specific_output_folder}")

    expected_page_count = 0
    try:
        with fitz.open(pdf_path_str) as doc:
            expected_page_count = len(doc)
    except Exception as e:
        logging.error(f"Could not open PDF {pdf_path_str} to get page count: {e}")
        return [] # Cannot proceed without page count

    if expected_page_count == 0:
        logging.warning(f"PDF {pdf_path_str} reported 0 pages. Skipping split.")
        return []

    # --- Check if single-page PDFs already exist ---
    specific_output_folder.mkdir(parents=True, exist_ok=True)
    existing_pdfs = sorted(
        glob.glob(str(specific_output_folder / "page_*.pdf")),
        key=lambda x: int(Path(x).stem.split('_')[-1]) # Sort by page number
    )

    if existing_pdfs and len(existing_pdfs) == expected_page_count:
        logging.info(f"Found {len(existing_pdfs)} existing single-page PDFs (matching page count) in {specific_output_folder}. Skipping split.")
        return [str(p) for p in existing_pdfs]
    elif existing_pdfs:
        logging.warning(f"Found {len(existing_pdfs)} existing single-page PDFs, but expected {expected_page_count}. Re-splitting PDF.")
        # Clean up potentially incomplete set before regenerating
        for pdf_file in existing_pdfs:
            try:
                os.remove(pdf_file)
            except OSError as e:
                 logging.error(f"Error removing existing single-page PDF {pdf_file}: {e}")
    # --- End check ---

    logging.info(f"Starting PDF split for: {pdf_path_str} ({expected_page_count} pages)")
    output_pdf_paths = []
    try:
        with fitz.open(pdf_path_str) as source_doc:
            for i in range(expected_page_count):
                page_num = i + 1
                output_filename = specific_output_folder / f"page_{page_num:03d}.pdf"
                with fitz.open() as single_page_doc: # Create a new empty PDF
                    single_page_doc.insert_pdf(source_doc, from_page=i, to_page=i) # Insert the single page
                    single_page_doc.save(str(output_filename))
                output_pdf_paths.append(str(output_filename))
                if page_num % 50 == 0: # Log progress periodically
                     logging.info(f"  Split page {page_num}/{expected_page_count}")

        # Verify count matches
        if len(output_pdf_paths) == expected_page_count:
            logging.info(f"Successfully split PDF into {len(output_pdf_paths)} single-page files in {specific_output_folder}")
            # Sort just in case list building wasn't sequential (though it should be)
            output_pdf_paths.sort(key=lambda x: int(Path(x).stem.split('_')[-1]))
            return output_pdf_paths
        else:
             logging.error(f"PDF split process finished, but generated {len(output_pdf_paths)} files instead of expected {expected_page_count}.")
             # Clean up potentially incomplete/failed split
             for pdf_file in output_pdf_paths:
                 if Path(pdf_file).exists():
                     os.remove(pdf_file)
             return []

    except Exception as e:
        logging.error(f"Error splitting PDF {pdf_path_str}: {e}", exc_info=True)
        # Attempt to clean up any files created during the failed split
        for pdf_file in output_pdf_paths:
             if Path(pdf_file).exists():
                 try:
                     os.remove(pdf_file)
                 except OSError as rm_e:
                     logging.error(f"Error removing partial file {pdf_file} after split failure: {rm_e}")
        return []


if __name__ == '__main__':
    # Example usage (for testing this module directly)
    print("Testing pdf_processor.py...")
    # Assumes the script is run from the project root (Naini/) or container root (/workspace)
    test_pdf = "/workspace/data/input_pdfs/Copy of report for FS 2020-21.pdf" # Make sure this file exists
    persistent_image_output = "data/persistent_images" # Base directory for images
    persistent_single_pdf_output = "data/persistent_single_pdfs" # Base directory for single PDFs

    if Path(test_pdf).exists():
        print("\n--- Testing Image Conversion (with persistence) ---")
        print(f"First run for images of {test_pdf}:")
        paths1_img = convert_pdf_to_images(test_pdf, persistent_image_output)
        if paths1_img:
            print(f"  Generated/Found {len(paths1_img)} images: {paths1_img[:2]}...")
        else:
            print("  Image generation failed.")

        print(f"\nSecond run for images of {test_pdf} (should skip conversion):")
        paths2_img = convert_pdf_to_images(test_pdf, persistent_image_output)
        if paths2_img:
            print(f"  Found existing images: {paths2_img[:2]}...")
            print(f"  Total found: {len(paths2_img)}")
        else:
             print("  Image finding failed.")

        print("\n--- Testing PDF Splitting (with persistence) ---")
        print(f"First run for splitting {test_pdf}:")
        paths1_pdf = split_pdf_to_pages(test_pdf, persistent_single_pdf_output)
        if paths1_pdf:
            print(f"  Generated/Found {len(paths1_pdf)} single-page PDFs: {paths1_pdf[:2]}...")
        else:
            print("  PDF splitting failed.")

        print(f"\nSecond run for splitting {test_pdf} (should skip split):")
        paths2_pdf = split_pdf_to_pages(test_pdf, persistent_single_pdf_output)
        if paths2_pdf:
            print(f"  Found existing single-page PDFs: {paths2_pdf[:2]}...")
            print(f"  Total found: {len(paths2_pdf)}")
        else:
            print("  Single-page PDF finding failed.")

    else:
        print(f"Test PDF not found at: {test_pdf}")