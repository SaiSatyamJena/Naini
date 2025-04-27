# naini_app/utils.py
import logging
import fitz # PyMuPDF
from typing import Optional, List, Tuple # Import necessary types

log = logging.getLogger(__name__)

def map_image_box_to_pdf_coords(
    image_box_px: List[int] | Tuple[int, int, int, int],
    img_width_px: int,
    img_height_px: int,
    pdf_page: fitz.Page,
    img_dpi: int = 300
) -> Optional[fitz.Rect]:
    """
    Maps a bounding box from image pixel coordinates to PDF point coordinates.

    Args:
        image_box_px: List or tuple [x1, y1, x2, y2] in pixel coordinates
                      relative to the top-left corner of the image.
        img_width_px: Width of the image in pixels.
        img_height_px: Height of the image in pixels.
        pdf_page: The fitz.Page object corresponding to the image.
                  Used to get PDF dimensions (points) and rotation.
        img_dpi: The DPI used when converting the PDF page to an image. (Currently unused but kept for signature).

    Returns:
        A fitz.Rect object representing the bounding box in PDF point coordinates
        (origin top-left, y increases downwards), or None if mapping fails.
    """
    try:
        # --- Input Validation ---
        if not isinstance(pdf_page, fitz.Page):
            log.error("Invalid fitz.Page object provided for coordinate mapping.")
            return None
        if not (isinstance(image_box_px, (list, tuple)) and len(image_box_px) == 4):
            log.error(f"Invalid image_box_px format: {image_box_px}. Expected list/tuple of 4 numbers.")
            return None
        if not (img_width_px > 0 and img_height_px > 0):
             log.error(f"Invalid image dimensions: {img_width_px}x{img_height_px}. Must be > 0.")
             return None

        # --- Get PDF page dimensions and rotation ---
        pdf_rect = pdf_page.rect # Returns fitz.Rect(x0, y0, x1, y1) in points, origin top-left
        pdf_width_pt = pdf_rect.width
        pdf_height_pt = pdf_rect.height
        rotation = pdf_page.rotation # Usually 0, 90, 180, 270

        log.debug(f"PDF Page Info: Rect={pdf_rect}, Width={pdf_width_pt:.2f}pt, Height={pdf_height_pt:.2f}pt, Rotation={rotation}")
        log.debug(f"Image Info: Dims=({img_width_px}x{img_height_px})px")
        log.debug(f"Input Image Box (pixels): {image_box_px}")

        # --- Calculate scaling factors ---
        # Determine the correct PDF dimension corresponding to the image width/height based on rotation
        if rotation == 0 or rotation == 180:
            # No swap
            pdf_dim_for_img_width = pdf_width_pt
            pdf_dim_for_img_height = pdf_height_pt
        elif rotation == 90 or rotation == 270:
            # Dimensions are swapped
            pdf_dim_for_img_width = pdf_height_pt # Image width corresponds to PDF height
            pdf_dim_for_img_height = pdf_width_pt  # Image height corresponds to PDF width
        else:
            # Should not happen with valid PDFs, but handle defensively
            log.error(f"Unsupported PDF page rotation encountered: {rotation}")
            return None

        # Avoid division by zero if image dimensions are somehow invalid despite checks
        if img_width_px <= 0 or img_height_px <= 0:
             log.error(f"Cannot calculate scale factors with non-positive image dimensions: {img_width_px}x{img_height_px}")
             return None

        scale_x = pdf_dim_for_img_width / img_width_px
        scale_y = pdf_dim_for_img_height / img_height_px

        log.debug(f"Calculated Scaling Factors: scale_x={scale_x:.4f}, scale_y={scale_y:.4f}")

        # --- Convert image pixel coordinates to PDF point coordinates ---
        img_x1, img_y1, img_x2, img_y2 = image_box_px

        # Simple scaling based on corresponding dimensions
        # Remember: image origin (0,0) is top-left, y increases downwards
        # fitz.Rect origin (0,0) is top-left, y increases downwards
        pdf_x1 = img_x1 * scale_x
        pdf_y1 = img_y1 * scale_y
        pdf_x2 = img_x2 * scale_x
        pdf_y2 = img_y2 * scale_y

        log.debug(f"Scaled PDF Box (points, pre-rotation adjustment): [{pdf_x1:.2f}, {pdf_y1:.2f}, {pdf_x2:.2f}, {pdf_y2:.2f}]")

        # --- Adjust coordinates based on rotation ---
        # The goal is to get the rectangle in the *unrotated* page's coordinate system.
        # PyMuPDF's transformation matrix handles this.

        # Create the rectangle in the *apparent* coordinate system (scaled from image)
        apparent_rect = fitz.Rect(pdf_x1, pdf_y1, pdf_x2, pdf_y2)

        # Get the transformation matrix that converts from the rotated page back to unrotated
        # Use page.transformation_matrix for coords ON the page -> default device space
        # Use page.rotation_matrix for angles
        # Use fitz.Matrix(rotation=-rotation) seems more direct for undoing rotation
        # Let's use the page's method to get the matrix that accounts for CropBox etc.
        # page.derotation_matrix undoes rotation relative to top-left
        transform_matrix = pdf_page.derotation_matrix

        # Apply the inverse rotation matrix to the scaled rectangle corners
        # This transforms the coordinates FROM the rotated view TO the unrotated view
        final_rect = apparent_rect * transform_matrix

        log.debug(f"Final Mapped PDF Rect (points, unrotated system): {final_rect}")

        # --- Final Clipping and Validation ---
        # Clip the resulting rectangle to the actual page boundaries (unrotated)
        # Use intersection operator '&'
        final_rect = final_rect & pdf_page.rect # pdf_page.rect is the unrotated MediaBox/CropBox

        # Ensure coordinates are valid (x1 <= x2, y1 <= y2) and the rect has area
        # The '&' operation should handle this, but double-check
        if final_rect.is_empty or final_rect.width <= 0 or final_rect.height <= 0:
             log.warning(f"Mapped rectangle resulted in empty or invalid dimensions after transformation/clipping: {final_rect}. Original box: {image_box_px}")
             # Consider if returning None is better here, depends on downstream tolerance
             # For now, return the potentially invalid rect and let caller handle?
             # Let's return None if it's invalid after clipping.
             return None

        # --- Return the fitz.Rect object ---
        log.info(f"Successfully mapped image box {image_box_px} to PDF Rect: {final_rect}")
        return final_rect # Return the fitz.Rect object

    except Exception as e:
        log.error(f"Unexpected error during coordinate mapping: {e}", exc_info=True)
        return None # Return None on any unexpected exception

# --- Standalone Testing Block ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s')
    log.info("--- Testing coordinate mapping function (utils.py) ---")

    # --- Mock Inputs ---
    mock_image_box = [100, 50, 700, 550] # x1, y1, x2, y2 in pixels (large box in middle)
    mock_img_width = 800 # Assume image width matches PDF width for rotation 0
    mock_img_height = 1000 # Assume image height matches PDF height for rotation 0
    mock_dpi = 300 # DPI value (not used in current calculation logic)

    # Simulate a PDF page (e.g., slightly larger than image dims in points)
    class MockFitzPage:
        def __init__(self, width_pt, height_pt, rot):
            # MediaBox/CropBox (unrotated dimensions)
            self.rect = fitz.Rect(0, 0, width_pt, height_pt)
            self.rotation = rot
            # Calculate derotation matrix based on rotation and page center
            self._update_matrices()

        def _update_matrices(self):
            """Calculate transformation matrices (simplified)."""
            center = self.rect.center
            # Matrix to undo rotation around the center
            self.derotation_matrix = fitz.Matrix(1, 1).pretranslate(-center.x, -center.y) \
                                           .prerotate(-self.rotation) \
                                           .pretranslate(center.x, center.y)
            # Matrix to apply rotation
            self.rotation_matrix = fitz.Matrix(1, 1).pretranslate(-center.x, -center.y) \
                                           .prerotate(self.rotation) \
                                           .pretranslate(center.x, center.y)

        @property
        def width(self): return self.rect.width
        @property
        def height(self): return self.rect.height

    pdf_w_pt, pdf_h_pt = 800, 1000 # Let PDF points match image pixels for simplicity here

    # Test Case 1: No rotation
    print("\n--- Test Case 1: Rotation 0 ---")
    mock_page_0 = MockFitzPage(pdf_w_pt, pdf_h_pt, 0)
    coords_0 = map_image_box_to_pdf_coords(
        mock_image_box, mock_img_width, mock_img_height, mock_page_0, mock_dpi
    )
    print(f"Result (Rotation 0): {coords_0}")
    # Expected: fitz.Rect(100.0, 50.0, 700.0, 550.0) because scale is 1:1 and no rotation adjustment

    # Test Case 2: Rotation 90
    print("\n--- Test Case 2: Rotation 90 ---")
    mock_page_90 = MockFitzPage(pdf_w_pt, pdf_h_pt, 90) # PDF page is 800w x 1000h (unrotated)
    # Image generated from this would be 1000w x 800h
    img_width_rot90 = 1000
    img_height_rot90 = 800
    # Assume the *same pixel box* was detected on the rotated image view
    coords_90 = map_image_box_to_pdf_coords(
        mock_image_box, img_width_rot90, img_height_rot90, mock_page_90, mock_dpi
    )
    print(f"Result (Rotation 90): {coords_90}")
    # Expected: The box (100,50,700,550) on the 1000x800 image corresponds to a
    # transformed rectangle on the 800x1000 unrotated page.
    # Check calculation manually:
    # scale_x = pdf_h / img_w = 1000 / 1000 = 1
    # scale_y = pdf_w / img_h = 800 / 800 = 1
    # Apparent rect = fitz.Rect(100, 50, 700, 550)
    # Apply derotation matrix for -90 degrees around center (400, 500)
    # Point (100, 50) -> Rotate -90 -> Becomes relative coords (-450, -300) -> Add center -> (-50, 200)? No, calculation is complex.
    # Let's trust the matrix math produces a valid rect within the original 800x1000 bounds. Example output might be Rect(50.0, 100.0, 550.0, 700.0) if rotation is perfect center.

    # Test Case 3: Invalid Page
    print("\n--- Test Case 3: Invalid Page ---")
    coords_invalid = map_image_box_to_pdf_coords(
         mock_image_box, mock_img_width, mock_img_height, None, mock_dpi
    )
    print(f"Result (Invalid Page): {coords_invalid}") # Expected: None

    # Test Case 4: Invalid Image Dimensions
    print("\n--- Test Case 4: Invalid Image Dims ---")
    coords_invalid_dims = map_image_box_to_pdf_coords(
         mock_image_box, 0, 0, mock_page_0, mock_dpi
    )
    print(f"Result (Invalid Dims): {coords_invalid_dims}") # Expected: None

    # Test Case 5: Invalid Box Format
    print("\n--- Test Case 5: Invalid Box Format ---")
    coords_invalid_box = map_image_box_to_pdf_coords(
         [10, 20], mock_img_width, mock_img_height, mock_page_0, mock_dpi
    )
    print(f"Result (Invalid Box): {coords_invalid_box}") # Expected: None

    log.info("--- Finished coordinate mapping tests ---")